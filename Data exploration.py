#!/usr/bin/env python
# coding: utf-8

# # Data exploration
# First let's import the data.

# In[192]:


# !pip install python-mnist
from mnist import MNIST
import types


mndata = MNIST('.')
mndata.gz = True

images, labels = mndata.load_training()
images = [np.array(i).reshape((28, 28)) for i in images]

images_t, labels_t = mndata.load_testing()
images_t = [np.array(i).reshape((28, 28)) for i in images_t]


# Now let's see how the images look like.

# In[ ]:


get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


# In[197]:


# display a small sample

fig = plt.figure(figsize=(200,10))
n = 24
for i in range(n):
    plt.subplot(1, n, 1 + i)
    plt.imshow(images[i])

plt.show()


# And the same for the test data:

# In[194]:


fig = plt.figure(figsize=(200,10))
start = 3000
for i in range(start, start+24):
    plt.subplot(1, 24, 1 + i - start)
    plt.imshow(images_t[i])

plt.show()


# Check the distribution of the data between classes.

# In[195]:


categorized = np.array([[x for x, y in zip(images, labels) if y == i] for i in range(10)])
for i in categorized:
    print(len(i))


# It's exactly equal.  
# Now let's generate a mean image for each category.

# In[196]:


str_labels = [
 'T-shirt/top',
 'Trouser',
 'Pullover',
 'Dress',
 'Coat',
 'Sandal',
 'Shirt',
 'Sneaker',
 'Bag',
 'Ankle boot'
]


means = np.mean(categorized, axis=1)
fig = plt.figure(figsize=(20,2))
for i in range(10):
    plt.subplot(1, 10, 1 + i)
    plt.imshow(means[i])
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(str_labels[i])


plt.grid(False)
plt.show()


# ### Now let's see if they provide a valid model

# In[154]:


n = len(images)
correct = 0
op = np.mean
for i in range(n):
    if labels[i] == np.argmin([op(i) for i in np.abs(np.subtract(means, images[i]))]):
        correct += 1
correct / n
# op=mean   gives accuracy 0.6125666666666667
# op=sum    gives accuracy 0.6125666666666667
# op=median gives accuracy 0.3148666666666667


# Not the best. 
# Let's try to look at the image classes manually and deduce something from that

# In[156]:


n = len(categorized[0])
n


# In[184]:


for i in range(10):
    plt.subplot(4, 5, 1 + 2 * i)
    plt.imshow(np.min(categorized[i], axis=0))
    plt.subplot(4, 5, 1 + 2 * i + 1)
    plt.imshow(np.max(categorized[i], axis=0))
plt.show()


# The average images gave us important information: all of the clothing has the same orientation. This suggests that, unlike in a real life scenario, this dataset will punish augmentation by anything that is not a symmetry on the general shape of that category. This means that nothing can be augmented using rotation, everything aside from shoes can be reflected along the vertical axis, and shear can be used for everything. It is worth mentioning that the shoe images distinctly differ from the others, so it might be worth training a model that decides between shoes and everything else and ensembling it with two specific models.

# ## First model
# I took the first model from the ["What is torch.nn really?"](https://pytorch.org/tutorials/beginner/nn_tutorial.html) tutorial. It was created for the MNIST dataset, so MNIST-fashion was easy to swap into it.

# In[201]:


# import data
import torch
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, TensorDataset
from contextlib import contextmanager

train = FashionMNIST('.', download=True, train=True)
n = len(train)
valid_n = n * 3 // 10
cuda = torch.device("cuda")
cpu = torch.device("cpu")
dev = cpu #cuda if torch.cuda.is_available() else cpu

@contextmanager
def switch_to_cpu(model):
    global dev
    _dev = dev
    dev = cpu
    model.to(cpu)
    yield
    dev = _dev
    model.to(dev)

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

def get_dl(xs, ys):
    return WrappedDataLoader(DataLoader(TensorDataset(xs.type(torch.float32), ys), batch_size=30), preprocess)

valid_dl = get_dl(train.train_data[:valid_n], train.train_labels[:valid_n])
train_dl = get_dl(train.train_data[valid_n:], train.train_labels[valid_n:])

torch.manual_seed(2137)
torch.cuda.manual_seed_all(213742069)
torch.cuda.manual_seed(213742069)


# In[202]:


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# The following is adapted from:
# https://pytorch.org/tutorials/beginner/nn_tutorial.html

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def get_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
        nn.AvgPool2d(4),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )
    

lr = 0.1

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def accuracy(model, valid_dl):
    reals = []
    preds = []
    for xb, yb in valid_dl:
        reals.append(yb)
        preds.append(model(xb).argmax(dim=1))
    
    return accuracy_score(torch.cat(reals, 0), torch.cat(preds, 0))

def fit(model, loss_func, opt, train_dl, valid_dl):
    last_loss = 2137.
    epoch = 1
    while True:
        model.train()
        for xb, yb in train_dl:
#             print(model(xb).size())
#             assert False
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad(), switch_to_cpu(model):
            # print diagnostics about current epoch
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            print("Epoch %s: val_loss=%s, val_acc=%s, train_acc=%s" % (epoch, val_loss, accuracy(model, valid_dl), accuracy(model, train_dl)))
        if last_loss - val_loss < 0.00001:
            print("delta loss was %s, stopping" % (last_loss - val_loss))
            break
        last_loss = val_loss
        epoch += 1

def initer(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight)

model = get_model()
model.to(dev)
model.apply(initer)
fit(model, F.cross_entropy, torch.optim.Adam(model.parameters()), train_dl, valid_dl)
print("DONE!")


# This achieved a 0.84 accuracy on Kaggle, so not the best. Let's try to create separate models as discussed earlier.
