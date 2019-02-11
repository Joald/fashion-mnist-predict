#!/usr/bin/env python
# coding: utf-8

# # Data exploration
# ### First let's import the data

# In[26]:


# !pip install python-mnist
from mnist import MNIST
mndata = MNIST('.')
mndata.gz = True
images, labels = mndata.load_training()
images = [np.array(i).reshape((28, 28)) for i in images]


# ### Now let's see how the images look like

# In[46]:


# display a small sample
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(200,10))
for i in range(24):
    plt.subplot(1, 24, 1 + i)
    plt.imshow(images[i])

plt.show()


# ### Check the distribution of the data between classes

# In[61]:


categorized = np.array([[x for x, y in zip(images, labels) if y == i] for i in range(10)])
for i in categorized:
    print(len(i))


# Seems to be equal. 
# Now let's generate a mean image for each category

# In[105]:


labels = [
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
#     print(means[i])
    plt.subplot(1, 10, 1 + i)
    plt.imshow(means[i])
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(labels[i])


plt.grid(False)
plt.show()


# ### Now let's see if they provide a valid model

# In[ ]:


n = len(images)
correct = 0
for i in range(n):
    np.min()


# In[108]:


x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
x1, x2, np.subtract(x1, x2)

