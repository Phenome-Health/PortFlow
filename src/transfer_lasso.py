#Optional numba JIT compilation for additional speedup
import numpy as np
from scipy import stats
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator




@jit(nopython=True)
def _compute_gamma_jit(X_col, residual_i, N):
    """JIT-compiled gamma computation"""
    return X_col.T @ residual_i / N

@jit(nopython=True)
def _compute_residual_i_jit(cached_residual, X_col, beta_i):
    """JIT-compiled residual adjustment"""
    return cached_residual + X_col * beta_i

@jit(nopython=True)
def _check_bounds_jit(gamma, lower, upper):
    """JIT-compiled bounds checking"""
    lower_check = gamma > lower
    upper_check = gamma < upper
    return upper_check * lower_check

class TransferLasso():
    def __init__(self,X,Y,beta_t, fit_intercept = True, initialize = 'zeros', copy_X = False, l = 1, a = 0, tol = 1e-4, max_iter = 1000,n_cpus = 1):
        if copy_X == True:
            X = stats.zscore(X, axis = 0)
            self.X = X.copy()
        else:
            X = stats.zscore(X, axis = 0)
            self.X = X
        self.X2 = (X**2).mean(axis = 0)
        self.Y = Y
        self.beta_t = beta_t
        self.l = l
        self.fit_intercept = fit_intercept
        self.a = a
        self.tol = 1e-4
        self.max_iter = max_iter

        if initialize == 'zeros':
            self.beta = np.zeros_like(X[0])#np.random.randint(-2,high = 2, size = len(X.T))*np.random.rand(len(X.T))
        elif initialize == 'transfer':
            self.beta = beta_t
        else: 
            raise ValueError('Either initialize with "zero" or "transfer"')
        self.n_cpus = n_cpus

        if fit_intercept == True:
            self.alpha = 0#np.random.randint(-2, high = 2)
        
        # Notify about JIT compilation status
        if HAS_NUMBA:
            print("Numba JIT compilation enabled for additional speedup")

    def _gamma(self, i):
        X = self.X
        N = len(X)
        # Optimized: use cached residual from fit() and add back feature i's contribution
        # Use JIT-compiled helper if available
        residual_i = _compute_residual_i_jit(self._cached_residual, X[:, i], self.beta[i])
        return _compute_gamma_jit(X[:, i], residual_i, N)
        
    def _alpha(self):
        beta = self.beta
        X = self.X
        Y = self.Y
        return Y.mean() - X.mean(axis = 0) @ beta

    def _solutions(self,gamma,i):
        b = self.beta_t[i]
        a = self.a
        l = self.l
        X2 = self.X2[i]
        #return np.array([gamma - l, gamma-l*(2*a-1), gamma + l*(2*a-1), gamma+l, 0, X2*self.beta_t[i]])/X2
        return np.array([0, b, gamma - l*(2*a-1)*np.sign(b)])

    def _lower_pos(self, i):
        b = self.beta_t[i]
        a = self.a
        l = self.l
        X2 = self.X2[i]
        #return np.array([X2*b + l, l*(2*a-1), 0, -1e12, -l, X2*b + l*(2*a-1) ])
        return np.array([-l, l*(2*a-1) + b, l*(2*a-1)])

    def _upper_pos(self,i):

        b = self.beta_t[i]
        a = self.a
        l = self.l
        X2 = self.X2[i]
        #return np.array([1e12,X2*b + l*(2*a-1),-1, -l, X2*b + l*(2*a-1), X2*b + l])
        return np.array([l*(2*a-1), l + b, l*(2*a-1)])

    def _lower_neg(self,i):
        b = self.beta_t[i]
        a = self.a
        l = self.l
        X2 = self.X2[i]
        #return np.array([l, 0, X2*b - l*(2*a-1), -1e12, -l*(2*a-1), X2*b-l ])
        return np.array([-l*(2*a-1), -l+b, -l*(2*a-1)])

    def _upper_neg(self,i):
        b = self.beta_t[i]
        a = self.a
        l = self.l
        X2 = self.X2[i]
        #return np.array([1e12,-1, -l*(2*a-1), X2*b-l, l, X2*b - l*(2*a-1)])
        return np.array([l, -l*(2*a-1) + b, -l*(2*a-1)])


    def _update(self, i) :
        b = self.beta_t[i]
        l = self.l
        if b >= 0:
            _lower = self._lower_pos(i)
            _upper = self._upper_pos(i)
        else:
            _lower = self._lower_neg(i)
            _upper = self._upper_neg(i)

        gamma = self._gamma(i)
        # Use JIT-compiled bounds checking
        sol_mask = _check_bounds_jit(gamma, _lower, _upper)

        if True not in sol_mask:
            #raise ValueError('Waaaaaahhhhhhh')
            solution = [gamma - l*np.sign(gamma)]
        else:
            solution = self._solutions(gamma, i)[sol_mask]
        self.beta[i] = solution[0]
        return solution[0]

    def fit(self):
        beta_t = self.beta_t
        for _ in range(self.max_iter):
            if _%100 == 0:
                print(_)
            
            # Optimized: cache residual computation once per iteration
            N = len(self.X)
            self._cached_residual = self.Y - self.alpha - self.X @ self.beta
            
            beta_old = self.beta.copy()
            
            for i, b in enumerate(beta_t):
                self._cached_residual = self.Y - self.alpha - self.X @ self.beta
                self._update(i)
            
            # Optimized: early stopping when converged
            # if np.abs(self.beta - beta_old).max() <= self.tol:
            #     self.alpha = self._alpha()
            #     if _%100 != 0:
            #         print(f"Converged at iteration {_}")
            #     break
            
            self.alpha = self._alpha()

        # self.train_mse = .5*(Y - self.alpha - X @ self.beta).mean()
        # self.train_log_likelihood = self.train_mse + l*(a*np.abs(self.beta).sum() + (1-a)*np.abs(self.beta - self.beta_t).sum())
    
    def predict(self, X):
        return self.alpha + np.dot(X, self.beta)

    def mse(self, Y, Y_pred):
        return ((Y-Y_pred)**2).mean()

    def score(self, X, Y):
        Y_pred = self.predict(X)
        mse = self.mse(Y, Y_pred)
        return 1- mse/Y.var()