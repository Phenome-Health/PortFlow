import numpy as np
from scipy import stats

try:
    from numba import jit, njit
    from numba.experimental import jitclass
    from numba import float32, int32, boolean, char, float64
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def _solutions(gamma,b,l,a):
    return np.array([0, b, gamma - l*(2*a-1)*np.sign(b)], dtype = float64)

@jit(nopython=True)
def _compute_gamma_jit(X_col, residual_i, N):
    """JIT-compiled gamma computation"""
    return np.dot(X_col, residual_i) / N

@jit(nopython=True)
def _compute_residual_i_jit(cached_residual, X_col, beta_i):
    """JIT-compiled residual adjustment"""
    return np.ascontiguousarray(cached_residual + X_col * beta_i)

@jit(nopython=True)
def _check_bounds_jit(gamma, lower, upper):
    """JIT-compiled bounds checking"""
    lower_check = gamma > lower
    upper_check = gamma < upper
    return upper_check * lower_check

@jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1], dtype = arr.dtype)
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0], dtype = arr.dtype)
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@jit(nopython = True)
def _mean_and_std_jit(X):
    """Compute mean and std in a single pass per column"""
    N = X.shape[0]
    P = X.shape[1]
    mean = np.empty(P, dtype=X.dtype)
    std = np.empty(P, dtype=X.dtype)
    for j in range(P):
        col = X[:, j]
        m = col.mean()
        mean[j] = m
        std[j] = col.std()
    return mean, std

@jit(nopython=True)
def _zscore_jit(X, axis=0):
    mean, std = _mean_and_std_jit(X)
    return np.asfortranarray((X - mean) / std)
    
@jit(nopython=True)
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

# @jit(nopython=True)
# def np_std(array, axis):
#   return np_apply_along_axis(np.std, axis, array)


# @jit(nopython=True)
# def _zscore_jit(X, axis = 0):
#     mean = np_mean(X, axis)
#     std = np_std(X, axis)
#     return np.asfortranarray((X - mean)/std)


spec = [
    ('beta_t', float32[::1]),
    ('fit_intercept', boolean),
    ('beta', float32[::1]),
    ('alpha', float32),
    ('l', float32),
    ('a', float32),
    ('tol', float32),
    ('max_iter', int32),
    ('n_cpus', int32), 
    ('_cached_residual', float32[::1])
]

@jitclass(spec)
class _TransferLasso():
    def __init__(self,
                beta_t, 
                fit_intercept = True, 
                alpha = 0,
                l = 1.0, 
                a = 0.0, 
                tol = 1e-4, 
                max_iter = 1000,
                n_cpus = 1):
        self.beta_t = beta_t
        self.l = l
        self.fit_intercept = fit_intercept
        self.a = a
        self.tol = 1e-4
        self.max_iter = max_iter
        self.beta = np.zeros_like(beta_t)
        self.alpha = alpha
        self.n_cpus = n_cpus
        #self._cached_residual = np.zeros_like(beta_t)
        
        # # Notify about JIT compilation status
        # if HAS_NUMBA:
        #     print("Numba JIT compilation enabled for additional speedup")
        

    def _gamma(self, X_col, i):
        N = len(X_col)
        # Optimized: use cached residual from fit() and add back feature i's contribution
        # Use JIT-compiled helper if available
        residual_i = _compute_residual_i_jit(self._cached_residual, X_col, self.beta[i])
        return _compute_gamma_jit(X_col, residual_i, N)
        
    def _alpha(self, X, Y):
        beta = self.beta
        return Y.mean() - np_mean(X, 0) @ beta


    def _lower_pos(self, i):
        b = self.beta_t[i]
        a = self.a
        l = self.l
        return np.array([-l, l*(2*a-1) + b, l*(2*a-1)])

    def _upper_pos(self,i):

        b = self.beta_t[i]
        a = self.a
        l = self.l
        return np.array([l*(2*a-1), l + b, l*(2*a-1)])

    def _lower_neg(self,i):
        b = self.beta_t[i]
        a = self.a
        l = self.l
        return np.array([-l*(2*a-1), -l+b, -l*(2*a-1)])

    def _upper_neg(self,i):
        b = self.beta_t[i]
        a = self.a
        l = self.l
        #return np.array([1e12,-1, -l*(2*a-1), X2*b-l, l, X2*b - l*(2*a-1)])
        return np.array([l, -l*(2*a-1) + b, -l*(2*a-1)])


    def _update(self, X_col, i) :
        b = self.beta_t[i]
        l = self.l
        a = self.a
        if b >= 0:
            _lower = self._lower_pos(i)
            _upper = self._upper_pos(i)
        else:
            _lower = self._lower_neg(i)
            _upper = self._upper_neg(i)

        gamma = self._gamma(X_col, i)
        # Use JIT-compiled bounds checking
        sol_mask = _check_bounds_jit(gamma, _lower, _upper)

        if not np.any(sol_mask):
            #raise ValueError('Waaaaaahhhhhhh')
            solution = np.array([gamma - l*np.sign(gamma)])
        else:
            solution = _solutions(gamma, b,l,a)[sol_mask]

        self.beta[i] = solution[0]
        #return solution[0]

    def fit(self, X, Y):
        beta_t = self.beta_t

        X = np.asfortranarray(X.astype(np.float32))
        Y = np.asfortranarray(Y.astype(np.float32))

        for _ in range(self.max_iter):
            if _%100 == 0:
                print(_)

            X = _zscore_jit(X, axis = 0)
            # Optimized: cache residual computation once per iteration
            N = len(X)
            self._cached_residual = Y - self.alpha - np.dot(X ,self.beta)
            
            beta_old = self.beta.copy()
            
            for i in range(len(self.beta_t)):
                old_beta_i = self.beta[i]
                X_col = X[:,i]
                self._update(X_col, i)
                # Cheap O(N) rank-1 update instead of full O(N*P) recompute
                self._cached_residual -= X_col * (self.beta[i] - old_beta_i)
            
            # Optimized: early stopping when converged
            # if np.abs(self.beta - beta_old).max() <= self.tol:
            #     self.alpha = self._alpha()
            #     if _%100 != 0:
            #         print(f"Converged at iteration {_}")
            #     break
            
            self.alpha = self._alpha(X,Y)

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


def TransferLasso(beta_t, fit_intercept=True, alpha=0.0,
                  l=1.0, a=0.0, tol=1e-4, max_iter=1000, n_cpus=1):
    # Do type conversion in plain Python before entering njit
    # if X.dtype != np.float32:
    #     X = X.astype(np.float32)
    # if Y.dtype != np.float32:
    #     Y = Y.astype(np.float32)
    if beta_t.dtype != np.float32:
        beta_t = beta_t.astype(np.float32)
    return _TransferLasso_njit(beta_t, fit_intercept, alpha, l, a, tol, max_iter, n_cpus)

@njit
def _TransferLasso_njit(beta_t, fit_intercept, alpha, l, a, tol, max_iter, n_cpus):
    return _TransferLasso(beta_t, fit_intercept, alpha, l, a, tol, max_iter, n_cpus)

# @njit
# def TransferLasso(
#                 X,
#                 Y,
#                 beta_t,
#                 fit_intercept = True, 
#                 alpha = 0.0,
#                 copy_X = False, 
#                 l = 1.0, 
#                 a = 0.0, 
#                 tol = 1e-4, 
#                 max_iter = 1000,
#                 n_cpus = 1 ):
#     X = np.ascontiguousarray(X, dtype=np.float32) 
#     Y = np.ascontiguousarray(Y, dtype=np.float32)
#     beta_t = np.ascontiguousarray(beta_t, dtype=np.float32)
#     return _TransferLasso(X,Y,beta_t, fit_intercept , alpha, copy_X , l , a , tol , max_iter , n_cpus )