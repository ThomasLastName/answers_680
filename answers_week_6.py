
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

### ~~~
## ~~~ Dependencies
### ~~~
import torch
torch.set_default_dtype(torch.double)



def mse( vector_of_target_values, vector_of_predicted_values ):
    return torch.mean(( vector_of_target_values - vector_of_predicted_values )**2)


def build_from_data( x_data, y_data, deg=1, loss_function=mse, dtype=torch.get_default_dtype() ):
    #
    # ~~~ Process the data
    x_data = x_data.reshape((-1,1)) # ~~~ cast as a proper column vector
    dim_H = deg+1                   # ~~~ the dimension of the hypothesis vector space H
    M = x_data**torch.arange(dim_H) # ~~~ the matrix where j-th column is x_train**j
    M = M.to(dtype)                 # ~~~ cast as the desired data type
    x_data = x_data.to(dtype)       # ~~~ cast as the desired data type
    y_data = y_data.to(dtype)       # ~~~ cast as the desired data type
    #
    # ~~~ Define the objective function in terms of the processed data
    def loss_of_residual(coefficients):
        assert coefficients.shape==(dim_H,)
        assert coefficients.dtype==dtype
        vector_of_target_values = y_data
        vector_of_predicted_values = M@coefficients
        return loss_function( vector_of_target_values, vector_of_predicted_values )
    #
    # ~~~ Assuming that `loss_function` is MSE, define a function the gradient of the function `loss_of_residual`
    def grad_of_mse_loss(coefficients):
        return 2*M.T@( M@coefficients - y_data ) / len(x_data)  # ~~~ the gradient of the quadratic function "coefficients\mapsto\|y-M@coefficients\|^2"
    #
    # ~~~ Assuming that `loss_function` is MSE, compute the (best) lambda such that `loss_of_residual` is lambda-smooth
    hessian = 2*M.T@M                                       # ~~~ the (constant) Hessian of the quadratic function "coefficients\mapsto\|y-M@coefficients\|^2
    L = torch.linalg.eigvalsh(hessian).max() / len(x_data)  # ~~~ the (\ell^2-\ell^2) operator norm of a symmetric matrix (such as the Hessian) is simply its largest singular value
    #
    return loss_of_residual, grad_of_mse_loss, L



def build_objective_from_data( *args, **kwargs ):
    return build_from_data( *args, **kwargs )[0]


def formula_for_the_gradient( *args, **kwargs ):
    return build_from_data( *args, **kwargs )[1]


def compute_lambda( x_data, y_data, deg=1, dtype=torch.get_default_dtype() ):
    return build_from_data( x_data=x_data, y_data=y_data, deg=deg, dtype=dtype )[2]

#