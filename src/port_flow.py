import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso, LinearRegression, ElasticNet, ElasticNetCV
from .transfer_lasso import TransferLasso, _zscore_jit
from . import cond_norm_flows as cnf
from .cond_norm_flows import Data
import os
import pickle
from scipy import stats

def get_overlapping_features(target_features, source_features):
    overlap = list(set(source_features).intersection(target_features))
    overlap.sort()
    target_only = [x for x in target_features if x not in overlap]
    source_only = [x for x in source_features if x not in overlap]
    
    return {'overlap' : overlap, 'target_only' : target_only, 'source_only' : source_only}
    
class PortFlow():
    def __init__(self, 
        target_feats,
        source_feats,
        n_feats_impute = None,
        n_feats_predict = None,
        linear_model = 'lasso',
        n_folds_linear = None,
        n_flow_steps = 4,
        n_layers_flow = 3,
        share_layer = 'simple',
        num_hf = 3, 
        optimizer = 'adam',
        out_dir = './port_flow_output/',
        lr_initial = 1e-3,
        scheduler_steps = 200,
        scheduler_gamma = .5,
        lambda_lm = 1e-3,
        enet_l1_ratio = .5,
        lambda_trans = 1,
        alpha_trans = 0,
        fit_intercept = True,
        eps_cv = 1e-3):
        
        
        self.out_dir = out_dir
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        ### save original features again for easy import when transfering the model ######  
        
        feats =  get_overlapping_features(target_feats, source_feats)
        feats['target'] = target_feats
        feats['source'] = source_feats
        self.features = feats
        with open(out_dir + 'features.pkl', 'wb') as f:
            pickle.dump(feats, f)
            

        
        
        ### initialize source linear model ######
        if linear_model == 'lasso':
            if n_folds_linear is None:
                self.linear_model = Lasso(alpha = lambda_lm, fit_intercept = fit_intercept) 
            else:
                self.linear_cv = True
                self.linear_model = LassoCV(alphas = np.linspace(1e-4,.1,100), cv = n_folds_linear, fit_intercept = fit_intercept, eps = eps_cv)
        elif linear_model == 'e_net':
            if n_folds_linear is None:
                self.linear_model = ElasticNet(alpha = lambda_lm, l1_ratio = enet_l1_ratio, fit_intercept = fit_intercept) 
            else:
                self.linear_cv = True
                self.linear_model = ElasticNetCV(alphas = np.linspace(1e-4,.1,100), l1_ratio = enet_l1_ratio, cv = n_folds_linear, fit_intercept = fit_intercept, eps = eps_cv)
        else:
            self.linear_model = LinearRegression(fit_intercept = fit_intercept)
        
        if n_feats_impute is None:
            in_shape = len(self.features['source_only'])
        else:
            in_shape = n_feats_impute
            
        if n_feats_predict is None:
            n_cond = len(self.features['overlap'])
        
        
        ##### initialize conditional normalizing flow model #######
        self.cnf_model = cnf.FlowModel(in_shape, n_layers_flow, n_flow_steps, n_cond, sharing_layer = share_layer, num_hf = num_hf)

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.cnf_model.parameters(), lr=lr_initial, amsgrad=True)
        else:
            raise ValueError('Only Adam optimizer is currently supported. Support for other optimizers coming soon!')
            
        if scheduler_steps is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = scheduler_steps, gamma = scheduler_gamma)
            
            
        
        self.lambda_trans = lambda_trans
        self.alpha_trans = alpha_trans
        self.fit_intercept = True
        

    def train_cond_norm_flow(self, 
                             data, 
                             n_epochs = 10, 
                             test_size = .2, 
                             split_col = None, 
                             validation_size = .2, 
                             outfile = 'trained_model.pth', 
                             keys = {'train' : 'train', 'validate' : 'valid', 'test' : 'test'}, 
                             plot_training = True, 
                             clip_value = 25, 
                             num_workers = 0,
                             batch_size = 64):

        model = self.cnf_model
        optimizer = self.optimizer
        scheduler = self.scheduler
        feats = self.features
        
        over_feats = feats['overlap']
        impute_feats = feats['source_only']
        feats_tot = impute_feats + over_feats
        
        if split_col is not None:
            feats_tot += [split_col]
            
        ## NOTE: PLEASE ENSURE THAT THE INDEX IS PROPERLY SET. 
        ## THAT IS, INDEX 0 SHOULD BE THE 0TH FEATURE, NOT THE INDEX
        data = data[feats_tot]
        
        means = data[over_feats].mean(axis = 0)
        var = data[over_feats].var(axis = 0)
        
        self.source_means_overlap = means
        self.source_var_overlap = var
        
        #idx_split = np.arange()
        data = cnf.load_data(data, len(impute_feats), split_col = split_col, test_size = test_size, validation_size = validation_size, batch_size = 64)
        self.data = data
        model = cnf.train(model, optimizer, scheduler, data, return_model = True, n_epochs = n_epochs, out_dir = self.out_dir, outfile = outfile, keys = keys, plot_training = plot_training, clip_value = clip_value, num_workers = num_workers)
        
        self.cnf_model = model
        
        print('Conditional Normalizing Flow Training complete')
    
    def train_source_lasso(self, 
                           data, 
                           predict_col, 
                           test_size = .2, 
                           outfile = 'source_lm_params.csv',
                           seed = None):
        
        self.predict_col = predict_col
        feats = self.features
        X_feats = feats['source_only'] + feats['overlap']
        
        if predict_col in X_feats:
            X_feats.remove(predict_col)
        
        X = data[X_feats].values
        Y = data[predict_col].values
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = seed, shuffle = True)
        
        X_train = stats.zscore(X_train, axis = 0)
        X_test = stats.zscore(X_test, axis = 0)
        model = self.linear_model
        model.fit(X_train, Y_train)
            
        R2 = model.score(X_test, Y_test)
        
        MSE = (1-R2)*Y_test.var()
        
        coefs = model.coef_
        intercept = model.intercept_
        
        vals = np.concatenate([np.array([R2, MSE, intercept, np.nan]),coefs])
        
        params_df = pd.DataFrame(vals, index = ['R2', 'mse', 'intercept', predict_col] + X_feats).rename({0 : 'params'}, axis = 1)
        
        params_df['means'] = np.concatenate([np.zeros(3), np.array([Y.mean()]),  X.mean(axis = 0)])
        params_df['std'] = np.concatenate([np.zeros(3),np.array([Y.std()]), X.std(axis = 0)])
        
        
        params_df.to_csv(self.out_dir + outfile)
        
        self.source_lm_params = params_df
        
        
    def load(self, 
             out_dir = None, 
             cnf_model_path = 'trained_model.pth', 
             lm_params_path = 'source_lm_params.csv', 
             feats_path = 'features.pkl'):
        
        if out_dir is None:
            out_dir = self.out_dir
        self.source_lm_params = pd.read_csv(out_dir + lm_params_path, index_col = 0)
        self.cnf_model = torch.load(out_dir + cnf_model_path, weights_only = False)
        
        self.predict_col = self.source_lm_params.index.tolist()[3]
        
        with open(out_dir + feats_path, 'rb') as f:
            self.features = pickle.load(f)
        
    def impute_cnf(self, 
                   df, 
                   n_samples = 1, 
                   shift = True, 
                   scale = False, 
                   include_target_only = False):
        
        feats_cond = self.features['overlap']
        feats_impute = self.features['source_only']
        feats_targ = self.features['target_only']

        feats_src = self.source_lm_params.index.tolist()[4:]

        feats_cond = [x for x in feats_src if x in feats_cond]
        feats_impute = [x for x in feats_src if x in feats_impute]
        
        X = df[feats_cond].values
    
        if shift:
            mean = self.source_lm_params.loc[feats_cond, 'means'].values
            X = X - X.mean(axis = 0) + mean
        if scale:
            std = self.source_lm_params.loc[feats_cond, 'std'].values
            X = X*(std/X.std(axis = 0))
            
        
        model = self.cnf_model
        model.eval()
        
        if include_target_only:
            X_targ = df[feats_targ].values
            X = np.hstack([X, X_targ])
            offset = len(feats_targ)
        else:
            offset = 0
        
        loader = DataLoader(cnf.Data(X, idx_split = 0), batch_size = 64)
        if n_samples == 1:
            data_samps = []
            print('Sampling source only features on target data')
            print(' ')
            for batch_idx, item in enumerate(tqdm(loader)):
                x = item[1]
                Z_samp = model._sample(x)
                dat = np.hstack([Z_samp, x])
                data_samps.append(dat)
            X_imp = np.concatenate(data_samps, axis = 0)
            
        else:
            X_imp = np.zeros(shape = (n_samples, X.shape[0], len(feats_impute) + len(feats_cond) + offset))
            for i in range(n_samples):
                data_samps = []
                for batch_idx, item in enumerate(tqdm(loader)):
                    x = item[1]
                    Z_samp = model._sample(x)
                    dat = np.hstack([Z_samp, x])
                    data_samps.append(dat)
                X_imp[i,:,:] = np.concatenate(data_samps, axis = 0)
                             
        self.imputed_data = X_imp
            
        
    def transfer_lasso(self, 
                       X, 
                       predict_col, 
                       outfile = 'target_lm_params.csv', 
                       return_df = False,
                       initialize = 'zeros', 
                       copy_X = False, 
                       l = 1, 
                       a = 0, 
                       tol = 1e-4, 
                       max_iter = 1000,
                       n_cpus = 1,
                       include_target_only = False, 
                       test_size = .2,
                       seed = None):
        
        features = self.features
        feats_use = features['source_only'] + features['overlap']
        
        source_params = self.source_lm_params
        beta_t = source_params.T[feats_use].values[0,:]
        
        if include_target_only:
            feats_targ_only = features['target_only']
            feats_use += feats_targ_only
            beta_t = np.concatenate([beta_t, np.zeros_like(feats_targ_only)])
        
        Y = X[predict_col].values
        X = X[feats_use].values
        
        X, Y, beta_t = X.astype(np.float32), Y.astype(np.float32), beta_t.astype(np.float32)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2,random_state = seed, shuffle = True)
        
        X_test = _zscore_jit(X_test, 0)
        
        
        model_trans = TransferLasso(
                                    beta_t, 
                                    l = l, 
                                    a = a, 
                                    tol = tol, 
                                    max_iter = max_iter, 
                                    n_cpus = 1)
        model_trans.fit(X_train, Y_train)
        
        R2 = model_trans.score(X_test, Y_test)
        MSE = (1-R2)*Y_test.var()
        intercept = model_trans.alpha
        coefs = model_trans.beta
        
        vals = np.concatenate([np.array([R2, MSE, intercept]),coefs])
        
        params_df = pd.DataFrame(vals, index = ['R2', 'mse', 'intercept'] + feats_use).rename({0 : 'params'}, axis = 1)
        
        self.target_lm = model_trans
        self.target_lm_params = params_df
        
        if outfile is not None:
            params_df.to_csv(self.out_dir + outfile)
        if return_df:
            return params_df
        
        
    def fit_source(self, 
                   df, 
                   predict_col, 
                   lm_outfile = 'source_lm_params.csv', 
                   cnf_outfile = 'trained_model.pth',
                   split_col = None, 
                   test_size = .2, 
                   seed = None,
                   **kwargs):
        
        if split_col is not None:
            self.train_cond_norm_flow(df, split_col = split_col, **kwargs)
            df = df.drop(split_col, axis = 1)
        else:
            self.train_cond_norm_flow(df, **kwargs)
        self.train_source_lasso(df, predict_col = predict_col)
        
        print('Source model trained and ready to download')
        print(' ')
        
        
    def fit_target_single(self, 
                   df,
                   predict_col,
                   n_samples = 1, 
                   seed = None,
                   shift = True, 
                   scale = False,  
                   outfile = 'target_lm_params.csv', 
                   initialize = 'zeros', 
                   copy_X = False, 
                   l = 1, 
                   a = 0, 
                   tol = 1e-4, 
                   max_iter = 1000,
                   n_cpus = 1,
                   include_target_only = False, 
                   test_size = .2):
        
        features = self.features
        feats_use = features['source_only'] + features['overlap']
        
        predict_col = self.predict_col
        
        if include_target_only:
            feats_targ_only = features['target_only']
            feats_use += feats_targ_only
        
        self.impute_cnf(df, 
                        n_samples = n_samples, 
                        shift = shift, 
                        scale = scale, 
                        include_target_only = include_target_only)
        
        
        X_imp = self.imputed_data
        
        df_imp = pd.DataFrame(X_imp, columns = feats_use, index = df.index)
        df_imp[predict_col] = df[predict_col].values
                
        self.transfer_lasso(df_imp, 
                            predict_col, 
                            outfile = outfile, 
                            initialize = initialize, 
                            copy_X = copy_X, 
                            l = l, 
                            a = a, 
                            tol = tol, 
                            max_iter = max_iter,
                            n_cpus = n_cpus,
                            include_target_only = include_target_only, 
                            test_size = test_size,
                            seed = seed)
        
        print('Model successfully ported!! Great job!!')
        print(' ')


    def fit_target_multi(self, 
                   df,
                   predict_col,
                   n_samples = 10, 
                   shift = True, 
                   scale = False,  
                   outfile = 'target_lm_params.csv', 
                   initialize = 'zeros', 
                   copy_X = False, 
                   l = 1, 
                   a = 0, 
                   tol = 1e-4, 
                   max_iter = 1000,
                   n_cpus = 1,
                   include_target_only = False, 
                   test_size = .2,
                   seed = None):
        print('Running multiple imputation. Note: This may take a while. ')
        features = self.features
        feats_use = features['source_only'] + features['overlap']
        
        predict_col = self.predict_col
        
        if include_target_only:
            feats_targ_only = features['target_only']
            feats_use += feats_targ_only
        
        self.impute_cnf(df, 
                        n_samples = n_samples, 
                        shift = shift, 
                        scale = scale, 
                        include_target_only = include_target_only)
        
        
        X_imp = self.imputed_data

        to_concat = []
        for X in X_imp:
            df_imp = pd.DataFrame(X, columns = feats_use, index = df.index)
            df_imp[predict_col] = df[predict_col].values
                    
            df_out = self.transfer_lasso(df_imp, 
                                predict_col, 
                                outfile = outfile, 
                                initialize = initialize, 
                                return_df = True,
                                copy_X = copy_X, 
                                l = l, 
                                a = a, 
                                tol = tol, 
                                max_iter = max_iter,
                                n_cpus = n_cpus,
                                include_target_only = include_target_only, 
                                test_size = test_size,
                                seed = None)

            to_concat.append(df_out)

        df_full = pd.concat(to_concat, axis = 1)
        self.target_lm_params = df_full

        if outfile is not None:
            df_full.to_csv(self.out_dir + outfile)

    def fit_target(self, 
                    df,
                    predict_col,
                    n_samples = 1, 
                    seed = None,
                    shift = True, 
                    scale = False,  
                    outfile = 'target_lm_params.csv', 
                    initialize = 'zeros', 
                    copy_X = False, 
                    l = 1, 
                    a = 0, 
                    tol = 1e-4, 
                    max_iter = 1000,
                    n_cpus = 1,
                    include_target_only = False, 
                    test_size = .2):

        if n_samples == 1:
            self.fit_target_single(
                   df,
                   predict_col,
                   n_samples = n_samples, 
                   seed = seed,
                   shift = shift, 
                   scale = scale,  
                   outfile = outfile, 
                   initialize = initialize, 
                   copy_X = copy_X, 
                   l = l, 
                   a = a, 
                   tol = tol, 
                   max_iter = max_iter,
                   n_cpus = n_cpus,
                   include_target_only = include_target_only, 
                   test_size = test_size)
        else:
            self.fit_target_multi(
                   df,
                   predict_col,
                   n_samples = n_samples, 
                   seed = seed,
                   shift = shift, 
                   scale = scale,  
                   outfile = outfile, 
                   initialize = initialize, 
                   copy_X = copy_X, 
                   l = l, 
                   a = a, 
                   tol = tol, 
                   max_iter = max_iter,
                   n_cpus = n_cpus,
                   include_target_only = include_target_only, 
                   test_size = test_size)


    ####### UNNECESSARY FOR NOW, BUT KEEPING IN CASE WE NEED LATER #########
    # def predict(self, X, beta, alpha):
    #     return alpha + np.dot(X, beta)

    # def mse(self, Y, Y_pred):
    #     return ((Y-Y_pred)**2).mean()

    # def score(self, X, Y):
    #     Y_pred = self.predict(X)
    #     mse = self.mse(Y, Y_pred)
    #     return 1- mse/Y.var()
    
    
    # def predict_multi(self, df, predict_col = None, average_params = False, impute = True):
    #     params = self.target_lm_params
    #     feats = params.index.tolist()[3:]
    #     X = df[feats].values
    #     if predict_col is not None:
    #         Y = df[predict_col].values

    #     if not average_params:
    #         Y_pred = []
    #         alphas = params.loc['intercept',:].values
    #         betas = params.iloc[3:, :].values
    #         for i, b in enumerate(betas):
    #             y = self.predict(X, b, alpha[i])
    #             Y_pred.append(y)
    #         Y_pred = np.array(Y_pred)
    #         #Y_mean = Y_pred.mean(axis = 1)
    #         #Y_var = Y_pred.var(axis = 1)
    #         df_pred = pd.DataFrame(Y_pred, index = df.index)
    #         return df_pred
    #     else:
    #         alpha = params.loc['intercept',:].values.mean()
    #         beta = params.iloc[3:, :].values.mean(axis = 1)
    #         Y_pred = self.predict(X, beta, alpha)
    #         df_pred = pd.DataFrame(Y_pred, index = df.index)
    #         return df_pred

    # def predict_single(self, df, predict_col = None, impute = True):
    #     params = self.target_lm_params
    #     feats = params.index.tolist()[3:]
    #     X = df[feats].values
    #     if predict_col is not None:
    #         Y = df[predict_col].values

    #     Y_pred = self.target_lm.predict(X)
    #     df_pred = pd.DataFrame(Y_pred, index = df.index)
    #     return df_pred

            



