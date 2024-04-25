
# # ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

# ### ~~~
# ## ~~~ Here I'm just hiding some code which does not work
# ### ~~~


# from math import log, sqrt
# import numpy as np
# from sklearn.kernel_ridge import KernelRidge
# from quality_of_life.my_numpy_utils         import generate_random_1d_data, list_all_the_hat_functions
# from quality_of_life.my_visualization_utils import points_with_curves, buffer


# #
# # ~~~ Make up some fake data
# f = lambda x: np.sin(2*np.pi*x)
# n_train = 120
# noise = 0.
# x_train, y_train, x_test, y_test = generate_random_1d_data( f, n_train=n_train, noise=noise )


# ### ~~~
# ## ~~~ DEMONSTRATION 2 of n: Proposition 10.3 in the special case when F is as RKHS, and V=span({ kernel(\cdot,c_i) : i=1,...,N } where centers={c_1,...,c_N}
# ### ~~~

# #
# # ~~~ A function that applies Proposition 10.3 with dim(V)==deg
# def RKHS_V_spanned_by_kernels( x_train, y_train, kernel, deg ):
#     lo = min(x_train)
#     hi = max(x_train)
#     centers = np.linspace(lo,hi,deg)
#     G_u = kernel(x_train,x_train)
#     C   = kernel(x_train,centers)
#     G_u_inv = np.linalg.inv(G_u)
#     b = np.linalg.solve( C.T@G_u_inv@C, C.T@G_u_inv@y_train )
#     a = G_u_inv@y_train - G_u_inv@C@b
#     return lambda x, a=a, b=b: kernel(x,x_train)@a + kernel(x,centers)@b

# k = lambda x,y: np.exp( -(x.reshape(-1,1)-y.reshape(1,-1))**2/2 ) / log(sqrt(2*np.pi))  # ~~~ for 1d data, (x.reshape(-1,1)-y.reshape(1,-1)).abs() is the cdist matrix
# h = .01
# k_h = lambda x,y,h=h: k(x/h,y/h)/h
# Delta = RKHS_V_spanned_by_kernels( x_train, y_train, kernel=k_h, deg=5 )
# points_with_curves( x_train, y_train, (Delta,f), grid=x_test )




# #
# # ~~~ A function that applies Proposition 10.3 with dim(V)==deg
# def RKHS( x_train, y_train, kernel, V ):
#     #
#     # ~~~ Define the matrices G_u, G_v, and C from the Proposition
#     G_u = kernel(x_train,x_train) 
#     C = np.column_stack([ v_j(x_train) for v_j in V ])  # ~~~ for point-evaluations, C_{i,j} is sipmly v_j(x_i)
#     #
#     # ~~~ Apply the formulas (10.12) in the text
#     G_u_inv = np.linalg.inv(G_u)
#     b = np.linalg.solve( C.T@G_u_inv@C, C.T@G_u_inv@y_train )
#     a = G_u_inv@y_train - G_u_inv@C@b
#     #
#     # ~~~ Apply formula (10.11) in the text
#     return lambda x, a=a, b=b: kernel(x,x_train)@a + np.column_stack([ phi(x) for phi in V ])@b


# k = lambda x,y: np.exp( -(x.reshape(-1,1)-y.reshape(1,-1))**2/2 ) / log(sqrt(2*np.pi))  # ~~~ for 1d data, (x.reshape(-1,1)-y.reshape(1,-1)).abs() is the cdist matrix
# h = 0.01
# k_h = lambda x,y,h=h: k(x/h,y/h)/h
# a = min(x_train)
# b = max(x_train)
# deg = 5
# V = [
#         lambda x,n=n: np.sin(2*np.pi*n*(x-a)/(b-a)) for n in range(deg//2)
#     ] + [
#         lambda x,n=n: np.cos(2*np.pi*n*(x-a)/(b-a)) for n in range(1,deg//2+1)
#     ]
# Delta = RKHS( x_train, y_train, kernel=k_h, V=V )
# points_with_curves( x_train, y_train, (Delta,f), grid=x_test )




# #
# # ~~~ Univariate Nadaraya–Watson kernel regression (https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya%E2%80%93Watson_kernel_regression)
# def M(x_train,y_train,kernel):
#     def m(x):
#         K = kernel(x,x_train)
#         return K@y_train/K.sum(axis=1)
#     return m

# k = lambda x,y: np.exp( -(x.reshape(-1,1)-y.reshape(1,-1))**2/2 ) / log(sqrt(2*np.pi))  # ~~~ for 1d data, (x.reshape(-1,1)-y.reshape(1,-1)).abs() is the cdist matrix
# h = 0.1
# k_h = lambda x,y,h=h: k(x/h,y/h)/h
# m = M( x_train, y_train, kernel=k_h )
# points_with_curves( x_train, y_train, (m,f), grid=x_test )



# #
# # ~~~ Scikit-learn's implementation of kernel ridgre regression
# kernel_reg = KernelRidge( kernel='rbf', gamma=1/(2*h**2) )
# kernel_reg.fit( x_train.reshape(-1,1), y_train )
# predictor = lambda x: kernel_reg.predict(x.reshape(-1,1))
# points_with_curves( x_train, y_train, (predictor,f), grid=x_test )




# #
# # ~~~ Apply Proposition 10.3 in the special case when F is as RKHS, and V=span({ kernel(\cdot,c_i) : i=1,...,N } where centers={c_1,...,c_N}
# def RKHS_V_spanned_by_kernels( x_train, y_train, kernel, centers ):
#     centers = np.linspace(-1,1,501)
#     G_u = kernel(x_train,x_train)
#     G_v = kernel(centers,centers)
#     C   = kernel(x_train,centers)
#     G_u_inv = np.linalg.inv(G_u)
#     b = np.linalg.solve( C.T@G_u_inv@C, C.T@G_u_inv@y_train )
#     a = G_u_inv@y_train - G_u_inv@C@b
#     return lambda x, a=a, b=b: kernel(x,x_train)@a + kernel(x,centers)@b


# Delta = RKHS_V_spanned_by_kernels( x_train, y_train, kernel=k_h, centers=np.linspace(-1,1,501) )
# points_with_curves( x_train, y_train, (Delta,f), grid=x_test )


