#!/usr/bin/env python
# coding: utf-8

# # Data exploration
# First let's import the data.

# In[9]:


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

# In[10]:


get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


# display a small sample
plt.style.use('grayscale')
fig = plt.figure(figsize=(200,10))
n = 24
for i in range(n):
    plt.subplot(1, n, 1 + i)
    plt.imshow(1 - images[i])

plt.show()


# And the same for the test data:

# In[15]:


fig = plt.figure(figsize=(200,10))
start = 3000
for i in range(start, start+24):
    plt.subplot(1, 24, 1 + i - start)
    plt.imshow(1 - images_t[i])
plt.style.use('grayscale')


# Check the distribution of the data between classes.

# In[17]:


categorized = np.array([[x for x, y in zip(images, labels) if y == i] for i in range(10)])
for i in categorized:
    print(len(i))


# It's exactly equal.  
# Now let's generate a mean image for each category.

# In[18]:


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
    plt.imshow(1 - means[i])
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


# In[19]:


for i in range(10):
    plt.subplot(4, 5, 1 + 2 * i)
    plt.imshow(np.min(categorized[i], axis=0))
    plt.subplot(4, 5, 1 + 2 * i + 1)
    plt.imshow(np.max(categorized[i], axis=0))
plt.show()


# The average images gave us important information: all of the clothing has the same orientation. This suggests that, unlike in a real life scenario, this dataset will punish augmentation by anything that is not a symmetry on the general shape of that category. This means that nothing can be augmented using rotation, everything aside from shoes can be reflected along the vertical axis, and shear can be used for everything. It is worth mentioning that the shoe images distinctly differ from the others, so it might be worth training a model that decides between shoes and everything else and ensembling it with two specific models.

# ## First model
# I took the first model from the ["What is torch.nn really?"](https://pytorch.org/tutorials/beginner/nn_tutorial.html) tutorial. It was created for the MNIST dataset, so MNIST-fashion was easy to swap into it.

# In[108]:


import torch
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, TensorDataset
from contextlib import contextmanager
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

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

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

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


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def predict_batch(model, data_loader):
    reals = []
    preds = []
    for xb, yb in data_loader:
        reals.append(yb)
        preds.append(model(xb).argmax(dim=1))
        
    return (torch.cat(reals, 0), torch.cat(preds, 0))

def accuracy(model, data_loader):
    return accuracy_score(*predict_batch(model, data_loader))

def loss(model, loss_func, data_loader):
    losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in data_loader])
    return np.sum(np.multiply(losses, nums)) / np.sum(nums)

def fit(model, loss_func, opt, train_dl, *, valid_dl=None, max_epochs=20, threshold=1e-6, min_epochs=None, silent=False):
    validation = valid_dl is not None
    print("Fitting a model...")
    last_loss = float("inf")
    epoch = 1
    accuracies = []
    losses = []
    while True:
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        
        with torch.no_grad(), switch_to_cpu(model):
            # print diagnostics about current epoch
            if validation:
                val_loss = loss(model, loss_func, valid_dl)
                val_acc = accuracy(model, valid_dl)
            train_loss = loss(model, loss_func, train_dl)
            train_acc = accuracy(model, train_dl)
        if not silent:
            if validation:
                print("Epoch %s: val_loss=%s, train_loss=%s, val_acc=%s, train_acc=%s" % (epoch, val_loss, train_loss, val_acc, train_acc))
            else:
                print("Epoch %s: train_loss=%s, train_acc=%s" % (epoch, train_loss, train_acc))
        
        losses.append((val_loss, train_loss) if validation else train_loss)
        accuracies.append((val_acc, train_acc) if validation else train_acc)
        
        compare_loss = (val_loss if validation else train_loss)
        min_epochs_satisified = min_epochs is None or epoch >= min_epochs
        
        # stop execution
        if threshold and last_loss - compare_loss < threshold and min_epochs_satisified:
            print("delta loss %s was lower than threshold %s, stopping" % (last_loss - compare_loss, threshold))
            break
        
        if max_epochs is not None and epoch >= max_epochs and min_epochs_satisified:
            print("Max epochs reached, stopping")
            break
        last_loss = compare_loss
        epoch += 1
    return np.array([losses, accuracies])

def initer(layer):
    if type(layer) == nn.Conv2d:
        pass#nn.init.kaiming_normal_(layer.weight)

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

def get_dl(xs, ys, prep=preprocess, bs=30):
    return WrappedDataLoader(DataLoader(TensorDataset(xs.type(torch.float32), ys), batch_size=bs), prep)

def get_dls(xs, ys, prep=preprocess, bs=30):
    n = len(ys) * 3 // 10
    return {'train': get_dl(xs[:n], ys[:n], prep, bs), 'valid': get_dl(xs[n:], ys[n:], prep, bs)}
        
torch.manual_seed(213742069)
torch.cuda.manual_seed_all(213742069)
torch.cuda.manual_seed(213742069)
train = FashionMNIST('.', download=True, train=True)
        
model = get_model()
model.to(dev)
model.apply(initer)
dls = get_dls(train.train_data, train.train_labels)
results = fit(model, F.cross_entropy, torch.optim.Adam(model.parameters()), dls['train'], valid_dl=dls['valid'], threshold=None, min_epochs=50)
print("DONE!")


# This achieved a Kaggle score of less than 0.85, so not the best. Let's see some statistics.

# In[115]:


acc = []
for y, xs in enumerate(categorized):
    n = len(xs)
    ys = y * torch.ones(n)
    xs = next(iter(get_dl(torch.from_numpy(xs), ys, preprocess, n)))[0]
    acc.append(accuracy_score(ys, model(xs).argmax(dim=1)))

plt.figure(figsize=(16, 9))
plt.bar(range(10), acc)
plt.xlabel('classes')
plt.ylabel('accuracy')
plt.title('Class accuracies')
plt.show()


# In[123]:


preds = model(next(iter(get_dl(train.train_data, train.train_labels, preprocess, 10000000)))[0]).argmax(dim=1)
acc = accuracy_score(train.train_labels, preds)
plt.figure(figsize=(16, 9))
plt.pie([acc_full, 1 - acc])
plt.title("Full accuracy: %s" % acc) 
plt.show()


# In[116]:


for i, (title, ylimits) in enumerate(zip(['Loss', 'Accuracy'], [(0.3, 0.7), (0.75, 0.9)])):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16, 9))
    # results_ = np.array(results)[1]
    #for acc, label in zip(results_, ['binary', 'boot', 'non_boot']):
    plt.plot(results[i, :, 0], label='valid')
    plt.plot(results[i, :, 1], label='train')
    plt.ylim(*ylimits)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(title.lower())
    plt.legend()
    plt.show()


# We see that the loss on validation reaches the minimum almost immediately, even though the accuracy climbs back up from the downfall around epoch 20.  
# Furthermore, it is really bad at detecting shirts, with only 60% accuracy. 

# Next thing I tried was ensembling 3 separate models as discussed earlier.

# In[56]:


# Warning: takes at least 15 minutes to run
def get_model(n_classes=10):
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, n_classes, kernel_size=3, stride=2, padding=1),
        nn.AdaptiveMaxPool2d(1),
        Lambda(lambda x: x.view(x.size(0), -1)),
    )

boots = [5, 7, 9]
not_boots = [i for i in range(10) if i not in boots]

def prep_ens(x, y):
    for i in range(10):
        y[y == i] = int(i in boots)
    return preprocess(x, y)

def prep_indexer(indexer):
    def prep(x, y):
        for i, k in enumerate(indexer):
            y[y == k] = i
        return preprocess(x, y)
    return prep
    

# torch.manual_seed_all(213742069)
torch.manual_seed(213742069)

torch.cuda.manual_seed_all(213742069)
torch.cuda.manual_seed(213742069)


train = FashionMNIST('.', download=True, train=True)

data_loaders = [get_dls(train.train_data, train.train_labels, prep_ens)]

mask_y = torch.stack([train.train_labels == i for i in boots]).max(dim=0)[0]
mask_x = mask_y.view(-1, 1, 1).repeat(1, 28, 28)

data_loaders.append(get_dls(train.train_data.masked_select(mask_x).view(-1, 28, 28), train.train_labels.masked_select(mask_y), prep_indexer(boots)))
mask_x, mask_y = 1 - mask_x, 1 - mask_y
data_loaders.append(get_dls(train.train_data.masked_select(mask_x).view(-1, 28, 28), train.train_labels.masked_select(mask_y), prep_indexer(not_boots)))

models = [get_model(2), get_model(3), get_model(7)]
min_epochs = [5, 20, 20]
max_epochs = [7, 25, None]
results = []
for model, dl, min_epoch, max_epoch in zip(models, data_loaders, min_epochs, max_epochs):
    model.to(dev)
    model.apply(initer)
    train_dl, valid_dl = (dl, None) if dl.__class__ == WrappedDataLoader else dl
    result = fit(
        model,
        F.cross_entropy,
        torch.optim.Adam(model.parameters()), 
        dl['train'], 
        valid_dl=dl['valid'], 
        max_epochs=50,
        threshold=None
    )
    results.append(result)
print("DONE!")


# In[75]:


for i, (title, ylimits) in enumerate(zip(['Loss', 'Accuracy'], [(0., 0.9), (0.68, 1.01)])):
    plt.figure(figsize=(16, 9))
    for acc, label in zip(results[:, i], ['binary', 'boot', 'non_boot']):
        plt.plot(acc[:, 0], label=label + '_valid')
        plt.plot(acc[:, 1], label=label + '_train')
    plt.ylim(*ylimits)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(title.lower())
    plt.legend()
    plt.show()


# The plots don't appear like we fixed the issues mentioned earlier, let's see the prediction accuracy though.

# In[121]:


def predict(X):
    X1 = models[0](X)
    X1 = X1.argmax(dim=1)
    X2 = []
    for i, x in enumerate(X1):
        # x == 1 means binary model thinks it's a boot
        res = models[2 - x](X[i].view(1, 1, 28, 28))
        val = res.argmax(dim=1).item()
        X2.append(boots[val] if x == 1 else not_boots[val])
    
    return torch.tensor(X2), X1

accs_cat = []
accs_cat_binary = []
for i, cat in enumerate(categorized):
    n = len(cat)
    ys = i * torch.ones(n)
    trains = next(iter(get_dl(torch.from_numpy(cat), ys, prep_ens, n)))
    preds = predict(trains[0])
    accs_cat.append(accuracy_score(ys, preds[0]))
    accs_cat_binary.append(accuracy_score(trains[1], preds[1]))

plt.figure(figsize=(16, 9))
plt.bar(range(10), accs_cat_binary, label='binary')
plt.bar(range(10), accs_cat, label='full ensemble')
plt.title("Class accuracy comparison")
plt.xlabel('classes')
plt.ylabel('accuracy')
plt.legend(loc=3)
plt.show()


# In[124]:


trains = next(iter(get_dl(train.train_data, train.train_labels, prep_ens, 10000000)))
preds = predict(trains[0])
acc_full = accuracy_score(train.train_labels, preds[0])

plt.figure(figsize=(16, 9))
plt.pie([acc_full, 1 - acc_full])
plt.title("Full accuracy: %s" % acc_full) 
plt.show()


# The train set accuracy actually decreased by 2%, the kaggle score suffered a similar loss. But let's check if it improved any single class.

# In[135]:


with torch.no_grad():
    acc = np.ones(10)
    for y, xs in enumerate(categorized):
        n = len(xs)
        ys = y * torch.ones(n)
        xs = next(iter(get_dl(torch.from_numpy(xs), ys, preprocess, n)))[0]
        acc[y] = accuracy_score(ys, model(xs).argmax(dim=1))

    accs_cat = np.ones(10)
    for i, cat in enumerate(categorized):
        n = len(cat)
        ys = i * torch.ones(n)
        trains = next(iter(get_dl(torch.from_numpy(cat), ys, prep_ens, n)))
        preds = predict(trains[0])
        accs_cat[i] = accuracy_score(ys, preds[0])

plt.figure(figsize=(16, 9))
plt.bar(str_labels, accs_cat - acc)
plt.title("Differences between first two models")
plt.xlabel('classes')
plt.ylabel('accuracy')

plt.show()


# The new net was gained a lot when classifying sneakers, while losing the most on the most problematic class, shirts. While the loss of accuracy on ankle boots is somewhat worrying, this can still prove be a decent ensemble if the base nets are improved.  
# However, the time has come for data augmentation. Let's stick with the first model and benchmark different augmentations on that.

# In[ ]:


test = FashionMNIST('.', download=True, train=False)
test_dl = get_dl(test.test_data, torch.ones(test.test_data.size(0)), preprocess, 10000)
preds = predict(next(iter(test_dl))[0])[0]
# preds = preds.argmax(dim=1)
import pandas as pd
df = pd.DataFrame()
df['Class'] = preds
df.index.name = 'Id'
df.to_csv('submission.csv')
df

