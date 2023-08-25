#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[3]:


df = df.iloc[:,2:]


# In[4]:


df.sample(5)


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.corr()


# In[14]:


plt.hist(df['Age'])
plt.xlabel('Age')
plt.show()


# In[43]:


sns.pairplot(df)


# In[17]:


sns.boxplot(df['Age'])


# In[18]:


sns.boxplot(df['EstimatedSalary'])


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.drop('Purchased',axis=1),df['Purchased'],test_size=0.3,random_state=0)


x_train.shape,x_test.shape


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#fit the scaler to the train set ,it will learn the parameters
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[21]:


x_train_scaled


# In[22]:


x_test_scaled


# In[23]:


#convert to dataframe
x_train_scaled = pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled,columns=x_test.columns)


# In[24]:


np.round(x_train.describe(),1)


# In[25]:


np.round(x_train_scaled.describe(),1)


# # Effect of scaling

# In[26]:


fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(12,5))
ax1.scatter(x_train['Age'],x_train['EstimatedSalary'])
ax1.set_title('Before scaling')
ax2.scatter(x_train['Age'],x_train['EstimatedSalary'],color='red')
ax2.set_title('After scaling')
plt.show()


# In[27]:


fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(12,5))
#before scaling
ax1.set_title('before scaling')
sns.kdeplot(x_train['Age'],ax=ax1)
sns.kdeplot(x_train['EstimatedSalary'],ax=ax1)
#after scaling
ax2.set_title('after scaling')
sns.kdeplot(x_train_scaled['Age'],ax=ax2)
sns.kdeplot(x_train_scaled['EstimatedSalary'],ax=ax2)
plt.show()


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


lr = LogisticRegression()


# In[32]:


lr.fit(x_train_transformed,)


# In[33]:


y_pred = lr.predict(x_test_scaled)


# In[37]:


from sklearn.metrics import accuracy_score
print("Accuracy",accuracy_score(y_test,y_pred))


# In[ ]:




