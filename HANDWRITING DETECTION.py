#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from sklearn.datasets import load_digits


# In[12]:


datasets=load_digits()


# In[13]:


datasets.data


# In[14]:


datasets.target


# In[15]:


datasets.data.shape


# In[16]:


dataImageLength=len(datasets.images)
dataImageLength


# #VISUALIZATION

# In[17]:


n=10 #No. of samples out of sample total 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow (datasets.images[-4])
plt.show()
datasets.images[n]


# In[18]:


n=10 #No. of samples out of sample total 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[3])
plt.show()
datasets.images[n]


# In[19]:


n=10 #No. of samples out of sample total 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[6])


# In[20]:


n=10 #No. of samples out of sample total 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[-2])


# In[21]:


n=10 #No. of samples out of sample total 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[-3])


# In[22]:


n=10 #No. of samples out of sample total 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[+2])


# In[23]:


n=10 #No. of samples out of sample total 1797
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[-4])
plt.show()
datasets.images[n]


# #SEGREGATION VALUES INTO INPUT X OUTPUT Y.

# In[24]:


X=datasets.images.reshape((dataImageLength,-1))
X


# In[25]:


Y=datasets.target
Y


# #SPLITTING DATA

# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[28]:


X_test.shape


# In[29]:


X_train.shape


# In[ ]:





# ##TRAINING

# In[30]:


from sklearn import svm
model=svm.SVC(kernel='rbf')
model.fit(X_train,y_train)


# In[31]:


from sklearn import svm
model_linear=svm.SVC(kernel='linear')
model_linear.fit(X_train,y_train)


# #PREDICTING

# In[32]:


n=-2
result=model.predict(datasets.images[n].reshape((1,-1)))
plt.imshow(datasets.images[n], cmap=plt.cm.gray_r,interpolation='nearest')
result
"\n"
plt.axis('off')
plt.title('%i' %result)
plt.show()


# In[ ]:





# #PREDICTION

# In[33]:


y_pred=model.predict(X_test)


# In[34]:


(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[35]:


#Evaluate Model


# In[36]:


from sklearn.metrics import accuracy_score
('Accuracy of the model{0}%'.format(accuracy_score(y_test,y_pred)*100))


# In[ ]:




