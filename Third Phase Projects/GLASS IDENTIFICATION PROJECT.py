#!/usr/bin/env python
# coding: utf-8

# # Glass Identification Project

# In[44]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from collections import Counter
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')


# In[45]:


data=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset3/main/glass.csv')
data


# In[46]:


data.columns=['ID','RI', 'Na', 'Mg', 
                   'Al', 'Si', 'K', 'Ca', 'Ba',
                       'Fe', 'Type_of_glass']


# In[47]:


data


# In[48]:


data=data.drop(['ID'], axis=1)
data


# In[49]:


data.describe()


# In[50]:


data.info()


# In[51]:


data=data.dropna()


# In[52]:


data.isnull().sum()


# In[53]:


data


# In[54]:


y=data['Type_of_glass']
y.head()


# In[55]:


data=data.drop('Type_of_glass', axis=1)
data.head()


# In[57]:


from sklearn import preprocessing
x=data.values
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
data=pd.DataFrame(x_scaled)
data


# In[61]:


# splitting the data for test and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(data,y,test_size=0.30, random_state=42)
print((X_train.shape))


# In[62]:


get_ipython().system('pip install pydotplus')


# In[63]:


clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[64]:


from sklearn.metrics import mean_squared_error
import math
clf=DecisionTreeClassifier()
clf.fit(X_train, y_train)
ypred=clf.predict(X_test)
asc=accuracy_score(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
rmse=math.sqrt(mse)
print(asc,mse,rmse)


# In[65]:


data.fillna(data.mean(), inplace=True)
data


# In[71]:


from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
clf=Perceptron(tol=1e-3, random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# In[72]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[74]:


#using logistic regression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)


# In[75]:


clf.score(X_test, y_test)


# In[76]:


# using random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
clf=RandomForestClassifier(n_estimators=150, max_depth=3, random_state=0)
clf.fit(X_train, y_train)
ypred=clf.predict(X_test)
accuracy_score(y_test, y_pred)
asc=accuracy_score(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
rmse=sqrt(mse)
print(asc,mse,rmse)


# In[90]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from math import sqrt
clf=MultinomialNB()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
accuracy_score(y_test, pred)
asc=accuracy_score(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
rmse=sqrt(mse)
print(asc,mse,rmse)


# In[89]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
pred=knn.predict(X_test)
accuracy_score(y_test, pred)
print(classification_report(y_test, pred))
print(confusion_matrix(y_pred,pred))


# In[ ]:




