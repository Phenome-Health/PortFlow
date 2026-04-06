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

def split(feature):
    """
    Splits the input feature tensor into two halves along the channel dimension.
    Channel-wise masking.
    Args:
        feature: Input tensor to be split.
    Returns:
        Two output tensors resulting from splitting the input tensor into half
        along the channel dimension.
    """
    C = feature.size(1)
    return feature[:, : C // 2, ...], feature[:, C // 2:, ...]

def cross(feature):
    """
    Performs two different slicing operations along the channel dimension.
    Args:
        feature: PyTorch Tensor.
    Returns:
        feature[:, 0::2, ...]: Selects every feature with even channel dimensions index.
        feature[:, 1::2, ...]: Selects every feature with uneven channel dimension index.
    """
    return feature[:, 0::2, ...], feature[:, 1::2, ...]

def flatten_sum(logps):
    while len(logps.size()) > 1:
        logps = logps.sum(dim=-1)
    return logps

class ActNorm(nn.Module):

    """
    Activation Normalization layer which normalizes the activation
    values of a batch by their mean and variance. The activations of
    each channel then should have zero mean and unit variance. This
    layer ensure more stable parameter updates during training as it reduces
    the variance over the samples in a mini batch.
    from: https://github.com/pclucas14/pytorch-glow/blob/master/invertible_layers.py
    """

    def __init__(self, num_features, logscale_factor=1.0, scale=1.0):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.num_features = num_features
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter("b", nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(1, num_features, 1)))

    def forward(self, input, logdet, reverse=False):

        if not reverse:
            ##print(input)

            input_shape = input.size()
            input = input.view(input_shape[0], input_shape[1], -1)
            #print('ActNorm: ', input_shape, input.size())

            if not self.initialized:
                #print('ok', self.num_features)
                self.initialized = True
                unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach() #idk if this is needed! but it works regardless.

                # Compute the mean and variance
                sum_size = input.size(0) * input.size(-1)
                b = -torch.sum(input, dim=(0, -1)) / sum_size #extra minus sign comes from x-mu. not sure why they did it this way
                vars = unsqueeze(
                    torch.sum((input + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size
                )
                logs = (
                    torch.log(self.scale / (torch.sqrt(vars) + 1e-6)) 
                    / self.logscale_factor
                )
                #print('ActNorm: ', b.size(), vars.size(), self.b.size())
                self.b.data.copy_(unsqueeze(b).data)
                self.logs.data.copy_(logs.data)
            logs = self.logs * self.logscale_factor
            b = self.b

            output = (input + b) * torch.exp(logs)
            dlogdet = torch.sum(logs) * input.size(-1)  # c x h
            logdet += dlogdet
            #print('ActNorm Forward: ', logdet)
            return output.view(input_shape), logdet

        elif reverse == True:
            # assert self.initialized
            input_shape = input.size()
            input = input.view(input_shape[0], input_shape[1], -1)
            logs = self.logs * self.logscale_factor
            b = self.b
            output = input * torch.exp(-logs) - b
            dlogdet = torch.sum(logs) * input.size(-1)  # c x h
            logdet -= dlogdet
            #print('ActNorm Reverse: ', logdet)
            return output.view(input_shape),logdet 

class QRFlow(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.from_numpy(np.linalg.qr(np.random.randn(self.num_features, self.num_features))[0])) # initialize weight with a random rotation

    def forward(self, x, logdet, reverse=False):
        n_samps, n_feats = x.size()

        if not reverse:
            Q,R = torch.linalg.qr(self.weight)
            #MAYBE COMPUTE THE LOG SUM AND THEN EXPONENTIAL. 
            dlogdet = torch.sum(torch.log(torch.diagonal(R))) # determinant is easy to calculate since Q is orthgonal #
            logdet += dlogdet
            output = (self.weight @ x.T).T
        else:
            Q,R =  torch.linalg.qr(self.weight)
            x=x.double()

            dlogdet = torch.sum(torch.log(torch.diagonal(R)))
            logdet -= dlogdet
            output = torch.linalg.solve(R, Q.T @ x.T).T # system is easy to solve since R is upper triangular. 
        return output, logdet

class PermuteSimple(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, x, logdet, reverse=False):
        
        if not reverse:
            output = torch.flip(x, [1])
            logdet += 0
            #print('Share forward: ', logdet)
        else:
            output = torch.flip(x,[1])
            logdet += 0
            #print('Share reverse: ', logdet)
        return output, logdet
    
    
class HF(nn.Module):
    def __init__(self, num_features, n_cond, num_hf = 2):
        super().__init__()
        out_feats = num_hf*num_features
        int_size_1 = int(np.mean([n_cond, out_feats]))
        int_size_2 = int(np.mean([int_size_1, out_feats]))
        int_size_3 = int(np.mean([int_size_2, out_feats]))
        
        self.num_hf = num_hf
        self.num_features = num_features
        self.net = nn.Sequential(
            nn.Linear(n_cond, int_size_1),
            nn.ReLU(), 
            nn.Linear(int_size_1, out_feats),
        )
        
    def forward(self, z, cond_x, logdet = 0, reverse = False):
        cond_x = cond_x.float()
        V = self.net(cond_x)
        V = torch.split(V, self.num_features, dim = 1)
        
        if reverse:
            V = tuple(reversed(V))
        for v in V:
            v2 = torch.einsum('bd,bd->b',v,v)
            vz = torch.einsum('bd,bd->b',v,z.float())
            z = z - (vz/v2)[:,None]*v
        
        return z, logdet
            

class Net(nn.Module):
    def __init__(
        self,
        input_shape, 
        n_cond,
        delta = 0,
        noscale=False,
        noscaletest=False
        ):
        super().__init__()

        d = 1 if noscale else 2
        int_size_1 = int(np.mean([input_shape+n_cond, d*input_shape]))
        int_size_2 = int(np.mean([int_size_1, d*input_shape]))
        int_size_3 = int(np.mean([int_size_2, d*input_shape]))
        #print(input_shape, n_cond,int_size_1, int_size_2)
        
        self.Net = nn.Sequential(
            nn.Linear(input_shape + n_cond ,int_size_1),
            nn.Dropout(p = .25),
            nn.ReLU(),
            nn.Linear(int_size_1, int_size_2),
            nn.Dropout(p = .25),
            nn.ReLU(),
            nn.Linear(int_size_2, int_size_3),
            nn.ReLU(),
            nn.Linear(int_size_3, d*input_shape + delta)
        )

    def forward(self, input, cond_x):
        h = torch.cat((input,cond_x), 1).to(torch.float32)
        #print(h.size())
        out = self.Net(h)
        
        return out

class ConditionalCoupling(nn.Module):
    def __init__(
        self,
        input_shape,
        n_cond,
        delta = 0,
        noscale = False,
        noscaletest = False,
    ):
        super().__init__()
        self.Net = Net(input_shape//2, n_cond, delta = delta,noscale = noscale, noscaletest = noscaletest)  ## architecture for learning the couplings 
        self.noscale = noscale
        self.noscaletest = noscaletest
        self.delta = delta

    def forward(self, z, cond_x = None, logdet=0, logpz=0, reverse=False):
        

        z1, z2 = split(z) # first initial split
        h = self.Net(z1, cond_x) # run half through the network

        if self.noscale:
            # print("Scale disabled")
            t, scale = h, torch.ones_like(h)
        else:
            # print("Scale enabled")
            t, h_scale = cross(h) # further split up the output into shift and scale 
            scale = torch.nn.functional.softplus(h_scale) #enforce positivity
            logscale = torch.log(scale) #take log for the log determinant

            if self.noscaletest:
                # print("Scale disabled for sampling")
                scale = torch.ones_like(scale)

        
        # add if testnocsale then t, scale = h, torch.ones_like(h)

        if not reverse:
            #print('Cond Couple: ', z2.size(), z1.size(), scale.size(), t.size(), self.delta)
            y2 = (z2 * scale) + t
            y1 = z1
            #print('scale: ', scale)
            #print('shift: ', t)
            logdet += 0 if self.noscale else flatten_sum(logscale)
            #print("Conditional Forward:" ,logdet )

        else:
            # reg = 1e-5
            # scale[scale == 0] = reg
            y2 = (z2 - t) / scale
            y1 = z1
            logdet -= 0 if self.noscale else flatten_sum(logscale)
            # print("Conditional reverse:" ,z) 
            # print('Scale: ', scale)
            # print("Shift: ", t)

        y = torch.cat((y1, y2), 1)
        return y, logdet

class FlowStep(nn.Module):
    def __init__(
        self,
        input_shape,
        n_cond,
        delta = 0,
        num_hf = 2,
        noscale = False,
        noscaletest = False,
        sharing_layer = "simple",
        norm_layer = "act"
    ):
        super().__init__()

        # 1. Activation Normalization
        if norm_layer == "act":
            self.norm =ActNorm(input_shape)
        elif norm_layer == 'batch':
            self.norm = nn.BatchNorm1d(input_shape)
        # 2. sharing layer definitions!
        if sharing_layer == 'simple':
            self.share = PermuteSimple(input_shape)
        elif sharing_layer == 'qr':
            self.share = QRFlow(input_shape)
        elif sharing_layer == 'hf':
            self.share = HF(input_shape, n_cond, num_hf = num_hf)
        # 3. Conditional Coupling layer
        self.conditionalCoupling = ConditionalCoupling(
        input_shape,
        n_cond,
        delta = delta,
        noscale = noscale,
        noscaletest = noscaletest,
        )
        self.sharing_layer = sharing_layer

    def forward(self, z, cond_x = None,logdet = 0, reverse=False):
        if type(logdet) == int:
            logdet = torch.from_numpy(np.array([logdet]))

        if (len(logdet.size()) == 1)&(logdet.size()[0] == 1):
            logdet = logdet * torch.ones_like(z[:,0])

        if not reverse:
            # 1. Activation normalization layer
            # add normalization for the conditioned features

            z, logdet = self.norm(z, logdet=logdet, reverse=False)
            # 2. Sharing layer (replaces the 1x1 invertible convolution)
            if self.sharing_layer == 'hf':
                z, logdet = self.share.forward(z, cond_x = cond_x, logdet=logdet, reverse=False)
            if self.sharing_layer == 'simplie':
                z, logdet = self.share.forward(z, logdet=logdet, reverse=False)
            # 3. Conditional Coupling Operation
            z, logdet = self.conditionalCoupling(
                z, cond_x = cond_x, logdet=logdet, reverse=False
            )
            #print('Flow Step: ', logdet.mean())
            return z, logdet

        else:
            #normalize the conditional features. 
            # need to normalize the half of the features which are being fed into the net
            ###################################
            # zz = z.detach().clone()
            # zz, _ = self.norm(zz, logdet = logdet, reverse = False)
            # zz1, zz2 = split(zz)
            # z1, z2 = split(z)
            # z = torch.concat([zz1, z2], axis = 1)
            ###############################
            #z = torch.cat([z1,z2])
            # 1. Conditional Coupling
            z, logdet = self.conditionalCoupling(
                z, cond_x = cond_x, logdet=logdet, reverse=True
            )
            # zz1, z2 = split(z)
            # z = torch.cat([z1, z2], axis=1)
            #z1, _ = self.norm(z1, logdet = 0, reverse = True) # unnormalize the half that was previously normalized
            #z = torch.cat([z1, z2])
            # 2. Sharing layer
            if self.sharing_layer == 'hf':
                z, logdet = self.share.forward(z, cond_x = cond_x, logdet=logdet, reverse=True)
            if self.sharing_layer == 'simplie':
                z, logdet = self.share.forward(z, logdet=logdet, reverse=True)
            # 3. final act norm layer
            z, logdet = self.norm(z, logdet=logdet, reverse=True)
            #print('Z mean:' z.mean(axis=0), z.std(axis=0))
        
            
            return z, logdet

class Gaussian_Diag(object):
    def __init__(self):
        super().__init__()
        pass

    def logp(self, x, mean, sigma):
        ones = torch.ones_like(x)
        #print('Gauss Diag: ',x.size(), mean.size(), sigma.size())
        ll = -0.5 * (x - mean) ** 2 / (sigma ** 2) - 0.5 * torch.log(
            2 * np.pi * (sigma ** 2) * ones
        )
        return ll.sum(axis=1)

    def sample(self, mean, sigma, eps=0):
        noise = torch.randn_like(mean)
        return mean + eps * sigma * noise

class MLP(nn.Module):
    def __init__(
        self,
        input_shape, 
        n_cond,
        delta = 0,
        delta_int = 0,
        final = False
        ):
        super().__init__()
       
        self.final = final
        if final:
            int_size_1 = int(np.mean([n_cond, 2*input_shape]))
            int_size_2 = int(np.mean([int_size_1, 2*input_shape]))
            int_size_3 = int(np.mean([int_size_2, 2*input_shape]))
            self.Net = nn.Sequential(
                nn.Linear(n_cond ,int_size_1),
                nn.Dropout(p = .25),
                nn.ReLU(),
                nn.Linear(int_size_1, int_size_2),
                nn.Dropout(p = .25),
                nn.ReLU(),
                nn.Linear(int_size_2, int_size_3),
                nn.ReLU(),
                nn.Linear(int_size_3, 2*input_shape) 
                )
        else:
            int_size_1 = int(np.mean([input_shape//2+n_cond, input_shape]))
            int_size_2 = int(np.mean([int_size_1, input_shape]))
            int_size_3 = int(np.mean([int_size_2, input_shape]))
            self.Net = nn.Sequential(
                nn.Linear(n_cond + input_shape//2 + delta_int ,int_size_1),
                nn.Dropout(p = .25),
                nn.ReLU(),
                nn.Linear(int_size_1, int_size_2),
                nn.Dropout(p = .25),
                nn.ReLU(),
                nn.Linear(int_size_2, int_size_3),
                nn.ReLU(),
                nn.Linear(int_size_3, input_shape+delta) 
                )


    def forward(self, input, cond_x):
        if self.final:
            h = cond_x.to(torch.float32)
        else:
            h = torch.cat((input,cond_x), 1).to(torch.float32)
        #print(h.size())
        out = self.Net(h)
        
        return out

class GaussianPrior(nn.Module):
    def __init__(self, input_shape, n_cond, delta, final=False):
        super(GaussianPrior, self).__init__() # old syntax, i believe? 
        self.input_shape = input_shape
        self.n_cond = n_cond
        self.final = final
        self.prior = Gaussian_Diag()
        self.delta = delta

        if input_shape%2 == 0:
            delta_int = 0
        else:
            delta_int = 1

        if final: 
            self.net = MLP(self.input_shape, self.n_cond, delta = 0, final = True)
        else:
            self.net  = MLP(self.input_shape, self.n_cond,delta = delta, delta_int = delta_int, final = False)

    def split_prior(self, z, cond_x):
        h = self.net(z, cond_x)
        mean, sigma = h[:, 0::2], nn.functional.softplus(h[:, 1::2]) # each needs to be same dimensionality as z ??
        #print(mean, sigma)
        return mean, sigma

    def final_prior(self, z, cond_x):
        h = self.net(z, cond_x)
        mean, sigma = h[:, 0::2], nn.functional.softplus(h[:, 1::2])
        return mean, sigma

    def forward(
        self, x, cond_x, eps, reverse = False, logpz=0, logdet=0, use_stored=False
    ):

        if not reverse:
            if not self.final:
                z, y = torch.chunk(x, 2, 1)
                #print('Gauss', z.shape, y.shape)
                mean, sigma = self.split_prior(z, cond_x)
                logpz += self.prior.logp(y, mean, sigma)
            else:
                # final prior computation
                z = x
                mean, sigma = self.final_prior(z, cond_x)
                logpz += self.prior.logp(x, mean, sigma)
            
            #print(f'Gaussian {self.final}:', logdet.mean(), logpz.mean())
                
        else:
            if not self.final:
                mean, sigma = self.split_prior(x, cond_x)
                z2 = self.prior.sample(mean, sigma, eps=eps)
                z = torch.cat((x, z2), 1)

            else:
                # final prior computation
                mean, sigma = self.final_prior(x, cond_x)
                z = self.prior.sample(mean, sigma, eps=eps)
        
        return z, logdet, logpz

class NormFlowNet(nn.Module):
    def __init__(
        self,
        input_shape,
        L,
        K,
        n_cond,
        num_hf = None,
        noscale = False,
        noscaletest = False,
        sharing_layer = "simple",
        norm_layer = "act",
        split_layers = True
    ):

        super().__init__()
        self.input_shape = input_shape
        self.L = L
        self.K = K
        self.output_shapes = [] 
        self.layers = nn.ModuleList()

        
        offsets = {}
        odd_layers = []
        idx = 0
        in_shapes = [input_shape]
        while idx < L:
            if input_shape%2 != 0:

                offset = 1
                odd_layers.append(idx)
            else:
                offset = 0

            offsets[idx] = offset
            input_shape = input_shape//2 + offset
            in_shapes.append(input_shape)
            idx += 1

        #offsets[odd_layers[0]] = 0
        # if odd_layers[0] == 0:
        #     offsets[odd_layers[0]+1] = 1

        input_shape = self.input_shape
        self.offsets = offsets
        #print(self.offsets)
        self.odd_layers = odd_layers
        #print(odd_layers)


        # Build Normalizing Flow
        self.level_modules = torch.nn.ModuleList()
        for i in range(self.L):
            self.level_modules.append(nn.ModuleList())

        for i in range(self.L):
            # delta = 0
            d = 0
            if i in odd_layers:
                delta = 2
                d = -1
            # elif i-1 in odd_layers:
            #     if input_shape%2 == 0:
            #         delta = 2
            #         d = -1
            else:
                delta = 0
                d = 0
            # if i-1 in odd_layers:
            #     dd = 1
            # else: 
            #     dd = 0
            # 1. Flow Steps
            for k in range(K):
                #print(in_shapes[i])
                self.level_modules[i].append(
                    FlowStep(
                        in_shapes[i], #input_shape + off,
                        n_cond,
                        delta = delta,
                        noscale = noscale,
                        noscaletest = noscaletest,
                        sharing_layer = sharing_layer,
                        norm_layer = norm_layer, 
                        num_hf = num_hf
                    )
                )
            
            if split_layers:
                if i < L - 1:
                    # 3.Split Prior for intermediate latent variables
                    #if input_shape%2 == 0:

                    self.level_modules[i].append(
                        GaussianPrior(in_shapes[i], n_cond, delta = d)
                    )
                    
                    #input_shape = input_shape//2 + off
                    
                    self.output_shapes.append(in_shapes[i])


        #print(self.output_shapes)
        self.level_modules[-1].append(
           GaussianPrior(in_shapes[i] , n_cond, delta = 0, final=True)
        )


    def forward(
        self, z, cond_x, logdet=0, logpz=0, eps=1, reverse=False, use_stored=False
    ):
        # Encode
        if not reverse:
            for i in range(self.L):
                for layer in self.level_modules[i]:
                    if isinstance(layer, FlowStep):
                        z, logdet = layer(
                            z,
                            cond_x = cond_x,
                            logdet=logdet,
                            reverse=False,
                        )
                    elif isinstance(layer, GaussianPrior):
                        z, logdet, logpz = layer(
                            z,
                            cond_x,
                            logdet=logdet,
                            logpz=logpz,
                            eps=eps,
                            reverse=False,
                        )
        else:
            # Decode
            for i in reversed(range(self.L)):
                for layer in reversed(self.level_modules[i]):
                    if isinstance(layer, GaussianPrior):
                        z, logdet, logpz = layer(
                            z,
                            cond_x,
                            logdet = logdet,
                            logpz = logpz,
                            eps = eps,
                            reverse = True,
                            use_stored = use_stored,
                        )
                        #print('Gauss reverse: ', z)
                    elif isinstance(layer, FlowStep):
                        z, logdet = layer(
                            z,
                            cond_x = cond_x,
                            logdet = logdet,
                            reverse = True,
                        )
                        #print('Flow reverse: ', z)
                    

        return z, logdet, logpz

class FlowModel(nn.Module):
    def __init__(
        self,
        input_shape,
        L,
        K,
        n_cond,
        num_hf = None,
        noscale = False,
        noscaletest = False,
        split_layers = True, 
        sharing_layer = 'simple',
        norm_layer = 'act'
    ):

        super().__init__()

        self.flow = NormFlowNet(
            input_shape=input_shape,
            L=L,
            K=K,
            n_cond = n_cond,
            noscale=noscale,
            noscaletest=noscaletest,
            split_layers = split_layers,
            sharing_layer = sharing_layer,
            norm_layer = norm_layer, 
            num_hf = num_hf
        )
        self.norm_cond = nn.BatchNorm1d(n_cond)

    def forward(
        self,
        z=None,
        cond_x=None,
        logdet=0,
        eps=1,
        reverse=False,
        use_stored=False,
    ):
        #print("initial:", logdet)
        if not reverse:
            return self.normalizing_flow(z, cond_x)

        else:
            return self.inverse_flow(
                z=z, cond_x=cond_x, logdet=logdet, eps=eps, use_stored=use_stored
            )

    def normalizing_flow(self, z, cond_x, logdet=0):
        cond_x = self.norm_cond(cond_x.to(torch.float))
        cond_x = cond_x.double()
        # Push z through flow
        z, logdet, logp_z = self.flow.forward(z=z, cond_x=cond_x, logdet=logdet)

        # Loss: Z'ks under Gaussian + sum_logdet
        D = float(np.log(2) * np.prod(z.size()[1:]))
        x_bpd = -(logdet + logp_z) #/ D
        # loss = x_bpd + 0.001 * l2_scale
        return z, x_bpd

    def inverse_flow(self, z, cond_x, eps = 1, logdet=0, use_stored=False):
        cond_x = self.norm_cond(cond_x.to(torch.float))
        cond_x = cond_x.double()
        if logdet == 0:
            if z is not None:
                logdet = torch.zeros_like(z[:,0])
        #print("inverse flow: ", logdet)
        x = self.flow.forward(
            z, logdet=logdet, cond_x=cond_x, eps=eps, reverse=True, use_stored=use_stored
        )
        return x

    def _sample(self, x, eps=1):
        """
        Super-resolves a low-resolution image with estimated params.
        """
        # Draw samples from model
        with torch.no_grad():
            samples = self.inverse_flow(z=None, cond_x=x, eps=eps)[0]
            return samples#.clamp(min=0, max=float(self.nbins - 1) / float(self.nbins))

class ToyData(Dataset):
    def __init__(self, X, idx_split = None):
        if idx_split == None:
            idx_split = len(X[0,:])//2
        self.z = X[:,:idx_split]
        self.cond_x= X[:, idx_split:]

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        z = self.z[idx,:]
        cond_x = self.cond_x[idx,:]
        return z, cond_x

class ProteomicsData(Dataset):
    def __init__(self, X, idx_split = None):
        if idx_split is None:
            idx_split = int(len(X[0,:])//2)
        if type(idx_split) == list:
            idx_no_split = [x for x in range(len(X.T)) if x not in idx_split]
            self.z = X[:,idx_split]
            self.cond_x= X[:, idx_no_split]
        elif type(idx_split) == int:
            self.z = X[:,:idx_split]
            self.cond_x= X[:, idx_split:]
            
    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        z = self.z[idx,:]
        cond_x = self.cond_x[idx,:]
        return z, cond_x
    
class Data(Dataset):
    def __init__(self, X, idx_split = None):
        if idx_split is None:
            idx_split = int(len(X[0,:])//2)
        if type(idx_split) == list:
            idx_no_split = [x for x in range(len(X.T)) if x not in idx_split]
            self.z = X[:,idx_split]
            self.cond_x= X[:, idx_no_split]
        elif type(idx_split) == int:
            self.z = X[:,:idx_split]
            self.cond_x= X[:, idx_split:]
            
    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        z = self.z[idx,:]
        cond_x = self.cond_x[idx,:]
        return z, cond_x
    
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
    
def train(model, optimizer, scheduler, data, n_epochs = 10, return_model = True, out_dir = './', outfile = 'trained_model.pth', keys = {'train' : 'train', 'validate' : 'valid', 'test' : 'test'}, plot_training = True, clip_value = 25, num_workers = 0):
    
    train_loader, valid_loader = data[keys['train']], data[keys['validate']]
    loss_p_epoch = []
    valid_loss = []
    step = 0
    # break_it = False
    for epoch in range(n_epochs): ### we can change the number of epochs LATER
        print(epoch)
        print(' ')
        running_loss = 0
        running_loss_val = 0

        for batch_idx, item in enumerate(tqdm(train_loader)):
            #print(batch_idx)
            z = item[0]
            x = item[1]
            #y, x = y.to(device), x.to(device)
            # if break_it:
            #     break
            model.train()
            optimizer.zero_grad()


            # Forward pass
            z, bpd = model.forward(z=z, cond_x=x, logdet=0)
            loss = bpd

            # Compute gradients
            running_loss+= loss.mean()

            loss.mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
#             grads = []
#             for param in model.parameters():
#                 grads.append(param.grad.view(-1))
                
#             for i, x in enumerate(grads):
#                 print((i, x.max(), x.min()))
#                 if max([torch.abs(x.max()), torch.abs(x.min())]) > 1000:
#                     break_it = True
#                     break
                
            

            # Update model parameters using calculated gradients
            optimizer.step()
            scheduler.step()
            step += 1

#         if break_it:
#             break
        model.eval()
        
        with torch.no_grad():
            for batch_idx, item in enumerate(tqdm(valid_loader)):
                z = item[0]
                x = item[1]

                z, bpd = model.forward(z = z, cond_x = x, logdet = 0)

                running_loss_val += bpd.mean()


        if np.isnan(running_loss_val):
            print(running_loss_val, running_loss)
            raise ValueError('Loss function is undefined!!!!')
            break

        loss_p_epoch.append(running_loss/len(train_loader.dataset))
        valid_loss.append(running_loss_val/len(valid_loader.dataset))
        
    xx = [x.item() for x in loss_p_epoch]
    yy = [i for i in range(len(loss_p_epoch))]

    xx_val = [x.item() for x in valid_loss]
    yy_val = [i for i in range(len(valid_loss))]

        
    dfp = pd.DataFrame()
    dfp['training_loss'] = xx
    dfp['valid_loss'] = xx_val
    dfp['epoch'] = yy
    dfp.to_csv(f'{out_dir}/loss_p_epoch.csv')
    
    #torch.save(model, f'{out_dir}{outfile}')
    torch.save(model.state_dict(), out_dir + outfile)
    
    if plot_training: 
        plt.plot(yy, xx, label = 'training')
        plt.plot(yy_val, xx_val, label = 'validation')
        plt.ylabel('-log-likelihood', fontsize = 18)
        plt.xlabel('epoch', fontsize = 18)
        plt.legend()
        plt.savefig(f'{out_dir}/loss_p_epoch.png')
        plt.show()
        plt.clf()
        
    if return_model:
        return model
        
        
def load_data(data, idx_split, test_size = .2, validation_size = .2, split_col = None, **kwargs):
    
    if split_col is not None:
        data_dict = {}
        grps = data.groupby(split_col)
        
        for key, vals in grps:
            data_dict[key] = DataLoader(Data(vals.drop(split_col, axis = 1).values, idx_split = idx_split), **kwargs)
    else:
        train, test = train_test_split(data.values, test_size = test_size + validation_size)
        
        val_split = validation_size/(test_size + validation_size)
        
        test, valid = train_test_split(test, test_size = val_split)
        
        #print(type(train), type(test), type(valid))
        data_dict = {'train' : train, 'test' : test, 'valid' : valid}
        
        data_dict = {key : DataLoader(Data(item, idx_split = idx_split), **kwargs) for key, item in data_dict.items()}
        
        
    return data_dict

    
        