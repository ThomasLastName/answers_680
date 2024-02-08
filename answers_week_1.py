
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

### ~~~
## ~~~ Dependencies
### ~~~

import numpy as np
from matplotlib import pyplot as plt
from quality_of_life.my_numpy_utils         import generate_random_1d_data, my_min, my_max
from quality_of_life.my_visualization_utils import buffer, points_with_curves

#
# ~~~ A helper function that prepares data identical to Fouract's in https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb
def Foucarts_training_data():
    # ~~~ equivalent to:
        # np.random.seed(12)
        # m = 15
        # x_train = np.random.uniform(-1,1,m)
        # x_train.sort()
        # x_train[0] = -1
        # x_train[-1] = 1
        # y_train = abs(x_train) + np.random.normal(0,1,m)
    x_train = np.array([-1.        , -0.97085008, -0.93315714, -0.72558136, -0.69167432,
                       -0.47336997, -0.43234329,  0.06747879,  0.21216637,  0.48009939,
                        0.70547108,  0.80142971,  0.83749402,  0.88845027,  1.        ])
    y_train = np.array([ 3.73781428,  2.08759803,  2.50769528,  0.63971456,  1.16841094,
                        -0.13801677,  0.08287235, -0.63793798, -0.12801989,  2.5073981 ,
                         0.12439097,  1.67456455,  1.7480593 ,  1.93609588, -0.18963857])
    return x_train, y_train

#
# ~~~ A helper function for polynomial regression
def univar_poly_fit( x, y, degree=1 ):
    coeffs = np.polyfit( x, y, degree )
    poly = np.poly1d(coeffs)
    return poly, coeffs

#
# ~~~ Measure how well the model does when certain subsets of the data are withheld from training (written to mimic the sklearn.model_selection function of the same name)
def cross_val_score( estimator, eventual_x_train, eventual_y_train, cv, scoring, shuffle=False, plot=False, ncol=None, nrow=None, f=None, grid=None ):
    #
    # ~~~ Boiler plate stuff, not important
    scores = []
    # models = []
    if plot:
        ncol = 1 if ncol is None else ncol
        nrow = cv if nrow is None else nrow
        fig,axs = plt.subplots(nrow,ncol)
        axs = axs.flatten()
        xlim = buffer(eventual_x_train)
        ylim = buffer(eventual_y_train)
        grid = np.linspace( min(xlim), max(xlim), 1001 )
    #
    # ~~~ Partition the training data
    if shuffle: # ~~~ shuffle the data before partitionint it
        reordered_indices = np.random.permutation( len(eventual_y_train) )
        eventual_x_train = eventual_x_train[reordered_indices]
        eventual_y_train = eventual_y_train[reordered_indices]
    x_val_sets = np.array_split( eventual_x_train, cv )     # ~~~ split `eventual_x_train` into `cv` different pieces
    y_val_sets = np.array_split( eventual_y_train, cv )     # ~~~ split `eventual_y_train` into `cv` different pieces
    #
    # ~~~ For each one of the pieces (say, the i-th piece) into which we split our data set...
    for i in range(cv):
        #
        # ~~~ Use the i-th subset of our data (which is 1/cv percent of our data) to train a model
        x_train = x_val_sets[i]
        y_train = y_val_sets[i]
        model = estimator( x_train, y_train )
        #
        # ~~~ Use the remaining cv-1 parts of our data (i.e., (cv-1)/cv percent of our data) to test the fit
        x_test = np.concatenate( x_val_sets[:i] + x_val_sets[(i+1):] )  # ~~~ all the data we didn't train on
        y_test = np.concatenate( y_val_sets[:i] + y_val_sets[(i+1):] )  # ~~~ all the data we didn't train on
        scores.append(scoring( y_test, model(x_test) ))
        #
        # ~~~ Plot the model that was trained on this piece of the data, if desired (this is mostly useful for building intuition)
        if plot:
            axs[i].plot( x_train, y_train, "o", color="blue", label="Training Data" )
            axs[i].plot( x_test, y_test, "o", color="green", label="Test Data" )
            axs[i].plot( grid, model(grid), "-", color="blue", label="Predictions" )
            if (f is not None and grid is not None):
                axs[i].plot( grid, f(grid), "--", color="green", label="Ground Truth" )
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            axs[i].grid()
            axs[i].legend()
    if plot:    # ~~~ after the loop is over, perform the final configuration of the plot, if applicable, and then render it
        fig.tight_layout()
        plt.show()
    return scores

#
# ~~~ Define the metric by which we will assess accurcay: mean squared error
def mean_squared_error( true, predicted ):
    return np.mean( (true-predicted)**2 )
    # ~~~ usually you'd load this or one of the other options from sklearn.meatrics (https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)
    # ~~~ we have defined it explicitly for transparency and simplicity



### ~~~
## ~~~ Possible answers to the first exercise
### ~~~

#
# ~~~ **When the data is not too noisy,** a complicated/aggressive/expressive model (i.e., higher degree of polynomial regression) might be appropriate
f_a = lambda x: np.abs(x)                   # ~~~ the so called "ground truth" by which x causes y
x_train_a, _ = Foucarts_training_data()     # ~~~ take only the x data
np.random.seed(123)                         # ~~~ for improved reproducibility
y_train_a = f_a(x_train_a) + (1/20)*np.random.random(size=x_train_a.size)   # ~~~ a less noisy version of Foucart's data
explanation_a_4 = "A Degree 4 Polynomial Can't Approximate the Corner Very Well"
explanation_a_10 = "A Degree 10 Polynomial May Fit Well when the Data is not Too Noisy"

#
# ~~~ **When the ground truth is complicated,** even if the data is fairly noisy, a more complicated/aggressive model might be appropriate
f_b = lambda x: np.exp(x)*np.cos(3*x*np.exp(x))     # ~~~ a more complicated choice of ground truth
np.random.seed(680)                                 # ~~~ for improved reproducibility
x_train_b, y_train_b, _, _ = generate_random_1d_data( ground_truth=f_b, n_train=50, noise=0.3 )
explanation_b_4 = "A Degree 4 Polynomial Can't Model a Complex Ground Truth"
explanation_b_10 = "A Degree 10 Polynomial Offers More Approximation Power"



### ~~~
## ~~~ Possible answers to the second exercise
### ~~~

#
# ~~~ Perform ERM for any vector space of functions H that the user supplies (recall: degree d polynomial regression is ERM with H={1,x,...,x^d})
def empirical_risk_minimization_with_linear_H( x_train, y_train, list_of_functions_that_span_H, ell_2_penalization_parameter=None ):
    #
    # ~~~ Shape coercion
    y_train = y_train.squeeze()
    #
    # ~~~ Enumerating list_of_functions_that_span_H = \{ \phi_1, \ldots, \phi_d \}, the model matrix is the matrix with j-th column \phi_j(x_train), i.e., (i,j)-th entry \phi_j(x^{(i)})
    model_matrix = np.column_stack([ phi(x_train) for phi in list_of_functions_that_span_H ])   # ~~~ for list_of_functions_that_span_H==[1,x] this coincides with ordinary least squares
    #
    #~~~ An optional sanity check
    m,n = model_matrix.shape
    assert m==len(x_train)==len(y_train)
    assert n==len(list_of_functions_that_span_H)
    #
    # ~~~ For regularization (used later in ch6), augment the problem as in equation (6.4) of the text
    if ell_2_penalization_parameter is not None:
        if not isinstance( ell_2_penalization_parameter, np.ndarray ):  # ~~~ scalar \lambda
            Lambda = ell_2_penalization_parameter**2 * np.eye(n)
        elif len(ell_2_penalization_parameter.shape)==1:                # ~~~ vectorial \lambda (not seen in lecture)
            Lambda = np.diag(ell_2_penalization_parameter)
        elif len(ell_2_penalization_parameter.shape)==2:                # ~~~ \lambda is a matrix (I think this is an exercise in the text?)
            Lambda = ell_2_penalization_parameter
        else:
            raise ValueError("Only a scalar, numpy vector, or numpy matrix may be passed as `ell_2_penalization_parameter`")
        model_matrix = np.vstack(( model_matrix, Lambda ))
        y_train = np.concatenate(( y_train, np.zeros(n) ))
    #
    #~~~ Find some coefficients which minimize MSE
    coeffs = np.linalg.lstsq( model_matrix, y_train, rcond=None )[0]
    #
    #~~~ A vectorized implementation of the MSE-minimizing function \widehat{\phi}(x) = \sum_j c_j\phi_j(x)
    empirical_risk_minimizer = lambda x, c=coeffs: np.column_stack([ phi(x) for phi in list_of_functions_that_span_H ]) @ c
    return empirical_risk_minimizer, coeffs

#
# ~~~ Wrap polynomial regression for convenience
def my_univar_poly_fit( x_train, y_train, degree, penalty=None ):
    monomials = [ (lambda x,j=j: x**j) for j in range(int(degree)+1) ]   # ~~~ define a list of functions which span the hypothesis class, in this case [ 1, x , x^2, ..., x^degree ]
    basis_for_H = monomials[::-1]   # ~~~ reverse the order in which they're listed; this is merely a convention adopted to be consistent with the convention used by np.polyfit
    return empirical_risk_minimization_with_linear_H( x_train, y_train, basis_for_H, ell_2_penalization_parameter=penalty )     # ~~~ that's it!

#
# ~~~ Define a routint that creates the list "H = [(j-th hat function) for j in range(n)]" where n is the length of a sequence of knots
def list_all_the_hat_functions(knots):
    knots = np.sort(knots)
    n = len(knots)
    hat_functions = []
    for j in range(n):
        midpoint = knots[j]
        if j==0:
            next_point = knots[j+1]
            hat_functions.append( 
                    lambda x, b=midpoint, c=next_point: my_max( 0, 1-(x-b)/(c-b) )
                )   # ~~~ the positive part of the the line with value 1 at b going down to value 0 at c
        if j==(n-1):
            prior_point = knots[j-1]
            hat_functions.append(
                    lambda x, a=prior_point, b=midpoint: my_max( 0, (x-a)/(b-a) )
                )   # ~~~ the positive part of the the line with value 0 at a going up to value 1 at b
        else:
            prior_point = knots[j-1]
            next_point = knots[j+1]
            hat_functions.append(
                    lambda x, a=prior_point, b=midpoint, c=next_point: my_max( 0, my_min(
                            (x-a) / (b-a),
                        1 - (x-b) / (c-b)
                        ))
                )
    return hat_functions

#
# ~~~ A wrapper for globally continuous linear spline regression
def univar_spline_fit( x_train, y_train, knots, penalty=None ):
    return empirical_risk_minimization_with_linear_H( x_train, y_train, list_all_the_hat_functions(knots), ell_2_penalization_parameter=penalty )



### ~~~
## ~~~ Answers to the third exercise
### ~~~

#
# ~~~ Make enough executive decisions that all that remains is for someone to supply us with the data
def Toms_example_of_the_cv_workflow( x_train, y_train, n_bins=2, plot=True, plot_like_Foucart=False, max_degree=20, ground_truth=None ):
    #
    # ~~~ Set hyperhyperparameters: those which will be used when determining the hyperparameters
    possible_hyperparameters = np.arange(max_degree)+1      # ~~~ i.e., np.array([1,2,...,max_degree])
    scores = []     # ~~~ an object in which to record the results
    #
    # ~~~ For each possible degree that we're considering for polynomial regression
    for deg in possible_hyperparameters:
        #
        # ~~~ Do cross validation
        estimator = lambda x_train,y_train: univar_poly_fit( x_train, y_train, degree=deg )[0]  # ~~~ wrapper that fits a polynomial of degree `deg` to the data y_train ~ x_train
        current_scores = cross_val_score( estimator, x_train, y_train, cv=int(n_bins), scoring=mean_squared_error )
        scores.append(current_scores)
    #
    # ~~~ Take the best hyperparameter and train using the full data
    best_degree = possible_hyperparameters[ np.median(scores,axis=1).argmin() ] # ~~~ lowest median "generalization" error of trianing on only a subset of the data
    best_poly,_ = univar_poly_fit( x_train, y_train, degree=best_degree )
    if plot:
        points_with_curves(
                x = x_train,
                y = y_train,
                curves = (best_poly,) if (ground_truth is None) else (best_poly,ground_truth),
                title = f"Based on CV with {n_bins} Bins, Choose Degree {best_degree} Polynomial Regression",
                curve_colors = ("Blue",) if (ground_truth is None) else None,               # ~~~ `None` reverts to default settings of `points_with_curves`
                curve_labels = ("Fitted Polynomial",) if (ground_truth is None) else None,  # ~~~ `None` reverts to default settings of `points_with_curves`
                marker_color = "Blue" if (ground_truth is None) else None,      # ~~~ `None` reverts to default settings of `points_with_curves`
                curve_marks = ("-") if (ground_truth is None) else None,        # ~~~ `None` reverts to default settings of `points_with_curves`
                xlim = [-1,1] if plot_like_Foucart else None,       # ~~~ `None` reverts to default settings of `points_with_curves`
                ylim = [-1.3,5.3] if plot_like_Foucart else None,   # ~~~ `None` reverts to default settings of `points_with_curves`
                model_fit = (ground_truth is not None)
            )
    return scores

