
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

import sys
import numpy as np

#
# ~~~ Try to use scipy for efficiency, if it is available
try:
    import scipy as sp
except:
    pass

#
# ~~~ Do PCA assuming that each *row* of X_data is a data point
def row_PCA( X_data, k=None, center=False, verbose=True, alpha=0.05 ):
    #
    # ~~~ Center the data, if desired (this is also necessary for consistency with, e.g., sklearn's implementation)
    if center:
        X_data -= np.mean(X_data,axis=0)    # ~~~ shift each feature to have an empirical mean of zero
    #
    # ~~~ Set up the matrix on which to do SVD
    m,d = X_data.shape  # ~~~ under our assumption, m is the number of data points, and d is the number of features in each data point
    ell = min(m,d)      # ~~~ the smaller of the two
    X = X_data.T if d<m else X_data
    assert X.shape[0]==ell
    B = X @ X.T         # ~~~ the smaller of the two matrices X_data.T@X_data and X_data@X_data.T
    #
    # ~~~ Do SVD
    user_specified_k = (k is not None)
    scipy_is_available = "scipy" in sys.modules.keys()
    k = k if user_specified_k else ell
    if scipy_is_available and user_specified_k: 
        evals,evecs = sys.modules["scipy"].sparse.linalg.eigsh(B,k)
    else:
        evals,evecs = np.linalg.eigh(B)
    #
    # ~~~ Process the results of SVD
    evals = evals[::-1]     # ~~~ reverse the eigenvales to get them in *descending* order (rather than the ascending order in which numpy and scipy return them)
    evecs = evecs.T[::-1].T # ~~~ reverse the *columns* of the eigenvector matrix (each column is an eigenvector) to remain consistent with ording of the eigenvectors
    left_singular_vectors_of_X_data = evecs if ell==d else X_data.T @ evecs @ np.diag(1/evals)  # ~~~ essentially as described in remark 8.2 of the text
    singualr_values_of_X_data = np.nan_to_num(np.sqrt(evals),nan=0) # ~~~ the singular values of X_data are the square roots of the eigenvalues of B
    if verbose and not user_specified_k:
        percentage_of_variance_explained = np.cumsum(singualr_values_of_X_data)/np.sum(singualr_values_of_X_data)
        enough = np.where( percentage_of_variance_explained>1-alpha )[0].min()  # ~~~ first index at which percentage_of_variance_explained>1-alpha
        enough += 1 # ~~~ to account for indexing starting at zero
        print(f"More than {1-alpha} percent of the variance is explained by the first {enough} principal components.")
    components = left_singular_vectors_of_X_data[:,:k]  # ~~~ the first k columns of U in the SVD of the matrix X_data.T in which each *column* is one data point
    singular_values = singualr_values_of_X_data[:k]     # ~~~ the first k singular values of X_data
    return components, singular_values

#