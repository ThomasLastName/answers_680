
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

### ~~~
## ~~~ Dependencies
### ~~~

import torch
from quality_of_life.my_torch_utils import hot_1_encode_an_integer


### ~~~
## ~~~ Possible answers
### ~~~

def measure_accuracy(model,X_test,y_test):
    with torch.no_grad():
        acc = ( model(X_test).argmax(dim=1) == y_test ).float().mean().item()   # ~~~ n_correct==( model(X_test).argmax(dim=1) == y_test ).sum()
    return acc


def my_cross_entropy(predicted,targets):
    encode = hot_1_encode_an_integer(n_class=5)
    t = encode(targets)
    p = predicted.softmax(dim=1)
    return ( -t*p.log() ).sum(axis=1).mean()
