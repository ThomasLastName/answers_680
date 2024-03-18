
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

### ~~~
## ~~~ Dependencies
### ~~~

import numpy as np


### ~~~
## ~~~ Possible answer to exercises 1 and 2
### ~~~

#
# ~~~ A routine that takes parameters and activation, assembles and returns the shallow NN that they define
def build_shallow_network(
        a,      # ~~~ vector of outer weights
        W,      # ~~~ matrix of inner weights
        b,      # ~~~ vector of inner biases
        c,      # ~~~ outer bias
        sigma   # ~~~ activation function
    ):
    #
    # ~~~ Set expectations
    if len(W.shape)==1:
        W = W.reshape(-1,1)
    H,d = W.shape
    assert a.shape==(H,)
    assert b.shape==(H,)
    #
    # ~~~ Define a forward passing model subject to those expectations
    def model(x):
        #
        # ~~~ Shape x appropriately
        x = np.array(x)
        only_one_input = len(x.shape)==0 or x.shape==(d,)
        if only_one_input:
            x = np.array([x])       # ~~~ e.g., convert 1.2 into np.array([1.2]) and [1.2,3.4] into np.array([[1.2,3.4]])
        if len(x.shape)==1 and d==1:
            x = x.reshape(-1,d)     # ~~~ e.g., convert [1.2,3.4] into np.array([[1.2],[3.4]]) in the d==1 case
        n, dim = x.shape
        assert dim==d               # ~~~ assert that every *row* of x is a datapoint
        #
        # ~~~ Compute the forward step
        output = c + sigma(b+x@W.T)@a
        #
        # ~~~ Format the output as desired
        if only_one_input:
            output = output.item()
        else:
            assert output.shape==(n,)
        #
        # ~~~ Return the formatted output
        return output
    #
    # ~~~ Return the model (itself simply a function) that we defined
    return model

#
# ~~~ Define a univariate shallow ReLU network with 10 hidden units
np.random.seed(680)
H = 10
W = np.random.normal(size=(H,1))
a = np.random.normal(size=(H,))
b = np.random.normal(size=(H,))
c = 0.4
ReLU = lambda x: np.maximum(x,0)
univar_model = build_shallow_network(a,W,b,c,ReLU)

#
# ~~~ Define a biivariate shallow ReLU network with 10 hidden units
np.random.seed(680)
input_dim = 2
H = 10
W = np.random.normal(size=(H,input_dim))
a = np.random.normal(size=(H,))
b = np.random.normal(size=(H,))
c = 0.4
ReLU = lambda x: np.maximum(x,0)
bivar_model = build_shallow_network(a,W,b,c,ReLU)
