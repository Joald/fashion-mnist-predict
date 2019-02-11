#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torchvision.datasets import FashionMNIST


# In[115]:


from torch.utils.data import DataLoader, TensorDataset
train = FashionMNIST('.', download=True, train=True)
dl = DataLoader(TensorDataset(train.train_data, train.train_labels), batch_size=int(len(train) * 0.3))
it = iter(dl)
valid = next(it)
valid_dl = DataLoader(TensorDataset(valid[0].type(torch.float32), valid[1]), batch_size=30)
Xs = []
ys = []
for i in it:
    Xs.append(i[0].type(torch.float32))
    ys.append(i[1])

train_X = torch.cat(Xs, 0)
    
train_dl = DataLoader(TensorDataset(train_X, torch.cat(ys, 0)), batch_size=30)
# valid[0].size(), valid[1].size()


# In[116]:


import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class FirstCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

lr = 0.1

def loss_batch(model, loss_func, xb, yb, opt=None):
#     print(model(xb).type(), yb.type())
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


model = FirstCNN()
fit(20, model, F.cross_entropy, torch.optim.SGD(model.parameters(), lr=0.1), train_dl, valid_dl)
print("DON!")


# In[64]:


import numpy as np
from matplotlib.pyplot import imshow
#imshow(np.asarray(train[7][0]))
# train[7]
from torch.utils.data import TensorDataset
# train_ds = TensorDataset(train.train_data, train.train_labels
# TensorDataset(train)[:30000]
[method_name for method_name in dir(train) if callable(getattr(train, method_name))]
# len(train)


# In[6]:





# In[37]:


import torch
from skorch import NeuralNetClassifier
net_regr = NeuralNetClassifier(
    module=FirstCNN,
    max_epochs=20,
    lr=0.1,
    criterion=torch.nn.NLLLoss,
#     criterion__weight=weight,
    optimizer=torch.optim.SGD,
    optimizer__momentum=0.9,
#     device='cuda',  # uncomment this to train with CUDA
)


# In[30]:


train.train_labels.unique()


# In[34]:


train.train_data = train.train_data.type('torch.FloatTensor')
net_regr.fit(train.train_data, train.train_labels)


# In[117]:


test.test_data = test.test_data.type('torch.FloatTensor')
preds = model(test.test_data)


# In[119]:


preds


# In[118]:


import pandas as pd
df = pd.DataFrame()
df['Class'] = preds
df.index.name = 'Id'
df.to_csv('submission1.csv')
df

