# PortFlow
A small toolkit for model portability. Includes Conditional Normalizing Flows for fast imputation for blocks of correlated missing features, Transfer Lasso for transfer learning of linear models, and PortFlow which combines both into a single, simple workflow. On the source data, the user initializes the model with the sets of features of both the target and source dataset. PortFlow then simultaneously learns a Conditional Normalizig Flow for imputing source-only features based on overlapping features, and trains a linear regression model on a user-defined outcome. The architecutre is summarzied in the following figure:

<img width = "400" src = https://github.com/user-attachments/assets/193ff33e-f9e7-4cba-adec-2c2ed2044d4d />

# Modules
Each module can be used as standalone classes, or combined into the PortFlow class.

## PortFlow
The ```PortFlow``` class is an all-in-one imputer and transfer class for imputing blocks of missing features and transfering models from a large source data set to a small target data set. After the source data has been trained, the output folder should be downloaded and reuploaded to the target data set. 

```
import numpy as np
import pandas as pd
from src.port_flow import PortFlow
import pickle

target_feats = [LIST OF FEATURES OF TARGET DATA]
source_feats = [LIST OF FEATURES ON SOURCE DATA]
#### PORTFLOW WILL IMPUTE SOURCE ONLY FEATURES BASED ON OVERLAPPING FEATURES.
#### FOR BEST RESULTS, ENSURE THIS NUMBER OF SOURCE ONLY FEATS IS NOT MORE THAN 40% OF THE ENTIRE DATASET
#### ALSO MAKE SURE THE STRING TYPES ARE CONSITINET, I.E. ALL UPPER OR ALL LOWERCASE. DEALERS CHOICE

model = PortFlow(
        target_feats,
        source_feats,
        linear_model = 'lasso',
        n_folds_linear = None,
        lambda_lm = 1e-3,
        enet_l1_ratio = .5,
        n_flow_steps = 4,
        n_layers_flow = 3,
        share_layer = 'simple',
        optimizer = 'adam',
        out_dir = './port_flow_output/',
        lr_initial = 1e-3,
        scheduler_steps = 200,
        scheduler_gamma = .5,
        lambda_trans = 1,
        alpha_trans = 0,
        fit_intercept = True,
        eps_cv = 1e-3
                )

################ ON SOURCE DATA ONLY ###################
source_data = pd.read_csv('/PATH/TO/SOURCE/DATA')

model.fit_source(data, 
           predict_col, ## column with outcome for the linear model 
           lm_outfile = 'source_lm_params.csv', ## save the parameters of the linear model
           cnf_outfile = 'trained_model.pth', ## save the CondNormFlow model
           split_col = None, ## if train_test_split has already been done, split_col is the columns which labels each set
           test_size = .2, ## size of test set of linear model
           seed = None, ## random seed for train_test_split
           **kwargs ## keywork arguments fed into conditional normalizing flow (see below)
                )

################ ON TARGET DATA ONLY ###################
target_data = pd.read_csv('/PATH/TO/TARGET/DATA')
#### REINITIALIZE MODEL ####
model = PortFlow(
        target_feats,
        source_feats,
        linear_model = 'lasso',
        n_folds_linear = None,
        lambda_lm = 1e-3,
        enet_l1_ratio = .5,
        n_flow_steps = 4,
        n_layers_flow = 3,
        share_layer = 'simple',
        optimizer = 'adam',
        out_dir = './port_flow_output/',
        lr_initial = 1e-3,
        scheduler_steps = 200,
        scheduler_gamma = .5,
        lambda_trans = 1,
        alpha_trans = 0,
        fit_intercept = True,
        eps_cv = 1e-3
                )

model.fit_target(data, 
            predict_col, ## same as source data
            n_samples = 1, ## denote the number of imputed samples to keep (for multiple imputation)
            seed = None, ## random seed for train_test_split
            shift = True, ## shift target data mean to match that of the source
            scale = False, ## scale target data to have same variance as source
            outfile = 'target_lm_params.csv', ## to save the parameters of the target model
            initialize = 'zeros', ## how to initialize the TransferLasso parameters
            l = 1, ## regularization parameters for TransferLasso
            a = 0, ## strenght of regular Lasso term in the TransferLasso loss function
            tol = 1e-4, ## tolerance for stopper criteria (NOT FINISHED YET)
            max_iter = 1000, ## number of iterations for coordinate descent
            include_target_only = False, ## whether or not to include target_only features in the transfer lasso 
            test_size = .2 ## test size for train_test_split
                )
```


## CondNormFlow

The ```CondNormFlow``` class is a pytorch implementation of the Conditional Normalzing Flows architecture. The code is largerly borrowed from [Winlkler et al. (2019)](https://arxiv.org/abs/1912.00042) but has been adapted for simpler tabular data as opposed to 2D image data. The main idea is to learn a cooridante transformation which maps the complicated data distribution of a subset of features to a Gaussian distribution where sampling is easy to perform. The details of the mapping and the final Gaussian distribution depends on a set of features on which your model will be conditioned. This gives a unique distribution for each set of conditioned features, and the resulting imputed data the preserves the covariance (and higher order moment) structure of the initial, full data distribution. 

Using ```CondNormFlow``` for imputation can be thought of as a higher-order regression. That is, while standard regression analysis aims to learn a conditional mean of the predictor $\mathbb{E}[Y|X]$, ```CondNormFlow``` learns and samples from the entire conditional distribution, where the conditional mean as well as the higher-order conditional moments are learned. Operationally ```CondNormFlows``` is comprised of a number of neural networks, all of which have at most 2 hidden layers, with ReLu activation function and 25% dropout layers. These networks are for learning the specific coordinate transformation, as well as the means and variances of a number of internal Gaussian distributions. The loss function is the negative log-likelihood of the distribution including the Jacobian determinant of the transformation. 

<<< ADD FIGURE HERE SHOWING THE DETAILS OF EACH LAYER >>>>>

### Usage
Hyperparameters include the number of flow steps, the number of data splits (see Winkler et al. and references therein for details on the data splitting procedures), the type of sharing layer. The specifics of the internal neural networks are fixed, but can be changed by users who are so inclined by hardwiring it into the code. 

A minimal example is 
```
from src.cond_norm_flows import FlowModel, Data, load_data, train
from torch.utils.data import Dataset, DataLoader
from torch import optim

data = pd.read_csv('/PATH/TO/DATA')
idx_split = [THE INDICIES OF THE FEATURES YOU WANT TO MODEL]
input_shape = len(idx_split)
n_cond = num_features - input_shape

data = load_data(data) # performs train-test-validation split and loads data into a pytorch DataLoader object/

model = FlowModel(
        input_shape, ## number of features to impute
        L, # number of splits in the data
        K, # number of flows per split
        n_cond, # size of conditional features
        sharing_layer = 'simple', # Options for other sharing layers coming soon.
        norm_layer = 'act')

optimizer = optim.Adam(model.parameters(), lr = .001, amsgrad = True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = .5)
model = train(model, optimizer, scheduler, data, n_epochs = 10, eturn_model = True, out_dir = './', outfile = 'trained_model.pth', plot_training = True, clip_value = 25)
```

For sampling from the model, again we utilize pytorch's DataLoader object and tqdm. 

```
loader = DataLoader(Data(your_data_here), batch_size = 64)
for batch_idx, item in enumerate(tqdm(loader)):
    x = item[1]
    Z_samp = model._sample(x)
    dat = np.hstack([Z_samp, x])
    data_samps.append(dat)
X_imp = np.concatenate(data_samps, axis = 0)
```
