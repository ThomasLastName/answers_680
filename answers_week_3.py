
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680


#
# ~~~ Extra featuers if you have cvxpy
try:
    import cvxpy as cvx
    use_cvx = True
except Exception as probably_ModuleNotFoundError:
    if type(probably_ModuleNotFoundError) is ModuleNotFoundError:
        use_cvx = False
    else:
        raise


### ~~~
## ~~~ Actually, this code doesn't quite work as I'd hoped, and I haven't really tried to fix it. I'm just hiding it here indefinitely
### ~~~

def LASSO( A, b, lam=0, solver=cvx.ECOS ):
    m,n = A.shape           # ~~~ get the number of rows and columns of A
    assert b.shape==(m,1)   # ~~~ safety feature
    x = cvx.Variable((n,1)) # ~~~ define the optimization variable x
    objective = cvx.Minimize( cvx.norm(A@x-b,2) + lam*cvx.norm1(x) )
    problem = cvx.Problem(objective)    # ~~~ put it all together into a copmlete minimization program
    problem.solve(solver=solver)        # ~~~ try to solve it
    return x.value if problem.status==cvx.OPTIMAL else None


def SLASSO( A, b, lam=0, solver=cvx.ECOS ):
    m,n = A.shape           # ~~~ get the number of rows and columns of A
    assert b.shape==(m,1)   # ~~~ safety feature
    x = cvx.Variable((n,1)) # ~~~ define the optimization variable x
    objective = cvx.Minimize( cvx.norm(A@x-b,2) + lam**2*cvx.norm1(x)**2 )
    problem = cvx.Problem(objective)    # ~~~ put it all together into a copmlete minimization program
    problem.solve(solver=solver)        # ~~~ try to solve it
    return x.value if problem.status==cvx.OPTIMAL else None



def LASSO_polyfit( x_train, y_train, degree=1, lam=0, squared_ell1=True, solver=cvx.ECOS ):
    #
    # ~~~ Shape coercion
    y_train = y_train.reshape(-1,1)
    #
    # ~~~ Build the matrix for which least squares is polynomial regression
    model_matrix = np.vstack([ x_train**j for j in range(int(degree)+1) ]).T[::-1]
    #
    #~~~ An optional sanity check
    m,n = model_matrix.shape
    assert m==len(x_train)==len(y_train)
    assert n==degree+1
    coeffs = SLASSO( model_matrix, y_train, lam, solver ) if squared_ell1 else LASSO( model_matrix, y_train, lam, solver )
    return coeffs


#
# ~~~ Validate our code by checking that our routine implements polynomial regression correctly (compare to the numpy implementation of polynmoial regression)
x_train, y_train = Foucarts_training_data()
poly, coeffs = univar_poly_fit( x_train, y_train, degree=2 )
my_coeffs = LASSO_polyfit( x_train, y_train, degree=2 )
assert abs(coeffs-my_coeffs).max() < 1e-14    # ~~~ if this passes, it means that our implementation is equivalent to numpy's
