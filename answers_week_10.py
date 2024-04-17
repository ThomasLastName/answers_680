
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/answers_680

### ~~~
## ~~~ Dependencies
### ~~~

import torch
from quality_of_life.my_torch_utils import convert_Dataset_to_Tensors



def my_dataloader( Dataset, batch_size=1, shuffle=False ):
    all_X, all_y = convert_Dataset_to_Tensors(Dataset)  # ~~~ extract the actual matrices
    order = torch.randperm(len(all_X)) if shuffle else torch.arange(len(all_X))
    all_X, all_y = all_X[order], all_y[order]           # ~~~ rearrange the data in that order
    x_batches, y_batches = torch.split(all_X,batch_size), torch.split(all_y,batch_size)
    return tuple(zip( x_batches, y_batches ))           # ~~~ convert the pair of tupes into tuples of pairs (literally, take the transpose)







# #
# # ~~~ Train the network
# for _ in range(epochs):             # ~~~ cycle `epochs`-many times through the entire dataset
#     #
#     # ~~~ Process the entire dataset
#     while len(remaining_indices)>0: # ~~~ until we run out of data to process
#         #
#         # ~~~ Assemble the next batch of data
#         for _ in range(b):          # ~~~ process the data in batches of size b
#             X,y = [],[]
#             try:                    # ~~~ try to take the next one of our randomly arranged data points
#                 image,label = MNIST_train[remaining_indices[0]]
#                 del remaining_indices[0]
#                 X.append(image)     # ~~~ add this image to the current batch of images
#                 y.append(label)     # ~~~ add its label to the current batch of labels
#             except:                 # ~~~ except if we have already run out (this happens on the final iteration of each epoch)
#                 pass                # ~~~ in that case, do nothing
#         X = torch.stack(X)          # ~~~ convert the list of images to a tensor (with pytorch's expected shaping conventions)
#         y = torch.stack(y)          # ~~~ convert the list of labels to a tensor (with pytorch's expected shaping conventions)

#