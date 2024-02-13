
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

import numpy as np
from matplotlib import pyplot as plt
from quality_of_life.my_numpy_utils import augment

### ~~~
## ~~~ Dependencies
### ~~~

def Foucarts_training_data( mneg=100, mpos=50, tag="linearly separable", plot=True ):
    #
    # ~~~ For reproducibility
    np.random.seed(1)
    def the_part_where_we_make_the_data( mneg, mpos, linearly_separable ):
        #
        # ~~~ Make the data that we will label negatively
        slope = 3/2 if linearly_separable else 1
        bias  = 1/2 if linearly_separable else 1/4
        Xneg_x = np.random.uniform(0,1,mneg)                          # ~~~ random horizontal coordinates
        Xneg_y = slope*Xneg_x + bias + np.random.normal(0,0.2,mneg)   # ~~~ vertical coordinate is a linear transoform of the horizontal coordinate, plus noise
        Xneg = np.column_stack((Xneg_x,Xneg_y))
        #
        # ~~~ Make the data that we will label positively
        shift = 1 if linearly_separable else 0.6
        bias  = 1/2 if linearly_separable else 1/4
        Xpos_x = np.random.uniform(0,1,mpos) + shift  # ~~~ random horizontal coordinates
        Xpos_y = np.random.uniform(0,1,mpos)          # ~~~ vertical coordinate is a linear transoform of the horizontal coordinate, plus noise
        Xpos = np.column_stack((Xpos_x,Xpos_y))
        return Xneg, Xpos
    #
    # ~~~ Reproduce the fact that Simon sets the seed, generates a separable dataset, and then generates the non-separable datasets one-by-one *without* resetting the seed
    Xneg, Xpos = the_part_where_we_make_the_data( mneg, mpos, True )
    if tag != "linearly separable":
        Xneg, Xpos = the_part_where_we_make_the_data( mneg, mpos, False )
    if tag == "sinusoidal" or tag == "radial":
        Xneg_x = np.random.uniform(0,6,mneg)
        Xneg_y = np.sin(Xneg_x) + np.random.uniform(0,0.5,mneg)
        Xneg = np.column_stack( (Xneg_x, Xneg_y) )
        Xpos_x = np.random.uniform(0,6,mpos)
        Xpos_y = np.sin(Xpos_x) - np.random.uniform(0,0.5,mpos)
        Xpos = np.column_stack( (Xpos_x, Xpos_y) )
    if tag == "radial":
        rpos = 1.1*np.sqrt(np.random.uniform(0,1,mpos))
        thetapos = np.random.uniform(0,2*np.pi,mpos)
        rneg = 0.8 + np.sqrt(np.random.uniform(0,1,mneg))
        thetaneg = np.random.uniform(0,2*np.pi,mneg)
        Xneg = np.column_stack( (rneg*np.cos(thetaneg), rneg*np.sin(thetaneg)) )
        Xpos = np.column_stack( (rpos*np.cos(thetapos), rpos*np.sin(thetapos)) )
    #
    # ~~~ Stack the data without shuffling, and assign labels
    X = np.vstack(( Xneg, Xpos))
    y = np.concatenate(( -np.ones(mneg), np.ones(mpos) ))
    #
    # ~~~ Plot the results, if desired
    if plot:
        plt.scatter( *X.T, c=y )
        plt.grid()
        plt.tight_layout()
        plt.show()
    #
    # ~~~ Most importantly, return them
    return X,y


def preceptron_update_without_bias( X, y, w, random_update=False ):
    correctly_classified = (np.sign(X@w)==np.sign(y))  # ~~~ an array of Boolean values
    if correctly_classified.min()==True:               # ~~~ if all True
        raise StopIteration
    i = np.random.choice(np.where(~correctly_classified)[0]) if random_update else correctly_classified.argmin()
    x_i = X[i,:]                            # ~~~ the (randomly chosen if random_update, else the first) misclassified data point
    new_w = w + y[i]/np.inner(x_i,x_i)*x_i  # ~~~ the update rule
    return new_w, i 



compute_slope_and_intercept = lambda w,b: (-w[0]/w[1], -b/w[1])


class HalfSpaceClassifier:
    def __init__(self,w,b):
        assert isinstance(w,np.ndarray) and w.ndim==1
        self.w = w
        self.b = b
        self.n_features = len(w)
    def __call__(self,X):
        predictions = np.sign( X@self.w + self.b )
        assert np.abs(predictions).min()>0     # ~~~ impose the assumption that we never exactly touch the classification boundary
        return  predictions



def training_data_to_feasibility_parameters(X_train,y_train):
    assert set(y_train)=={-1,1}
    A = augment(X_train) * y_train[:, np.newaxis]
    b = np.ones((A.shape[0],1))
    return A,b
