# PortFlow
A small toolkit for model portability. Includes Conditional Normalizing Flows for fast imputation for blocks of correlated missing features, Transfer Lasso for transfer learning of linear models, and PortFlow which combines both into a single, simple workflow. 


# Modules
Each module can be used as standalone classes, or combined into the PortFlow class.

## PortFlow

## CondNormFlow

The ```CondNormFlow``` class is a pytorch implementation of the Conditional Normalzing Flows architecture. The code is largerly borrowed from [Winlkler et al. (2019)](https://arxiv.org/abs/1912.00042) but has been adapted for simpler tabular data as opposed to 2D image data. The main idea is to learn a cooridante transformation which maps the complicated data distribution of a subset of features to a Gaussian distribution where sampling is easy to perform. The details of the mapping and the final Gaussian distribution depends on a set of features on which your model will be conditioned. This gives a unique distribution for each set of conditioned features, and the resulting imputed data the preserves the covariance (and higher order moment) structure of the initial, full data distribution. 

Using ```CondNormFlow``` for imputation can be thought of as a higher-order regression. That is, while standard regression analysis aims to learn a conditional mean of the predictor $\mathbb{E}[Y|X]$, ```CondNormFlow``` learns and samples from the entire conditional distribution, where the conditional mean as well as the higher-order conditional moments are learned. 

<img width="299" height="126" alt="Screenshot 2026-03-26 at 2 44 56 PM" src="https://github.com/user-attachments/assets/eb6bf4d5-154b-465a-9667-8ea7e0ceab8d" /> 


Image adapted from [the original publication](https://arxiv.org/abs/1912.00042). 

Operationally ```CondNormFlows``` is comprised of a number of neural networks, all of which have at most 2 hidden layers, with ReLu activation function and 25% dropout layers. These networks are for learning the specific coordinate transformation, as well as the means and variances of a number of internal Gaussian distributions. The loss function is the negative log-likelihood of the distribution including the Jacobian determinant of the transformation. 

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
