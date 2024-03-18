
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

### ~~~
## ~~~ Dependencies
### ~~~

import numpy as np



def par_grad_lacking_bias( a, W, x, sigma, sigma_prime, flatten_and_concatenate=True ):
    H = len(a)
    d = len(x)  # ~~~ note that x is assumed to have a length; this fails if, e.g., x=2.2; this function assumes x=[2.2] instead
    assert W.shape==(H,d)
    assert a.shape==(H,)
    assert x.shape==(d,)
    Wx = W@x    # ~~~ don't bother computing this twice
    #
    # ~~~ The partial derivatives of f(a,W) = a^T\sigma(Wx) with respect to a is \sigma(Wx)
    grad_a = sigma(Wx)
    #
    # ~~~ The partial derivative of f(a,W) = a^T\sigma\sigma(Wx) w.r.t. W_{i,j} is a_i\sigma'(\sum_jw_{i,f}x_f)x_j which equals the outer product:
    grad_W = np.outer( a*sigma_prime(Wx), x )
    #
    # ~~~ Return the things that you'd flatten and concatenate to get the gradient vector of f(a,W) = a^T\sigma\sigma(Wx)
    return np.concatenate(( grad_a, grad_W.flatten() )) if flatten_and_concatenate else (grad_a,grad_W)



def par_grad_with_bias( a, W, b, c, x, sigma, sigma_prime, flatten_and_concatenate=True ):
    x_tilde = np.concatenate((x,[1.]))
    W_tilde = np.column_stack((W,b))
    grad_a, grad_W_tilde = par_grad_lacking_bias( a, W_tilde, x_tilde, sigma, sigma_prime, flatten_and_concatenate=False )
    grad_W, grad_b = grad_W_tilde[:,:-1], grad_W_tilde[:,-1]
    grad_c = 1.
    return np.concatenate(( grad_a, grad_W.flatten(), grad_b, [grad_c] )) if flatten_and_concatenate else ( grad_a, grad_W, grad_b, grad_c )


def grad_of_item_loss( a, W, b, c, x, y, sigma, sigma_prime, ell_prime, flatten_and_concatenate=True ):
    x = np.array(x)
    assert len(x.squeeze().shape)<2 # ~~~ x is expected to be a single datapoint, *not* a list of several datapoints
    assert isinstance(y,float)      # ~~~ y is expected to be a single (float type) label, *not* a list of several labels
    x = x.reshape(-1)               # ~~~ turn 1.2 into [1.2] in order to enable the syntax W@x
    pred = c + np.inner(a,sigma(W@x+b))
    if flatten_and_concatenate:
        return ell_prime(y,pred)*par_grad_with_bias( a, W, b, c, x, sigma, sigma_prime )
    else:
        grad_a, grad_W, grad_b, grad_c = par_grad_with_bias( a, W, b, c, x, sigma, sigma_prime, flatten_and_concatenate=False )
        return ell_prime(y,pred)*grad_a, ell_prime(y,pred)*grad_W, ell_prime(y,pred)*grad_b, ell_prime(y,pred)*grad_c 

