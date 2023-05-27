#!/usr/bin/env python
# coding: utf-8

# # Rainfall Weather Forecasting

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from collections import Counter
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset3/main/weatherAUS.csv')
df


# In[3]:


# check for the data types
df.info()


# In[4]:


# checking for null values
df.isnull().sum()


# In[5]:


# checking for null values
df.isnull().mean()


# In[6]:


# deleting columns which are of no use
df=df.drop(['Date'], axis=1)
df


# In[7]:


df.columns


# In[8]:


#filling null values for numerical data
for column in['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm']:
    df[column]=df[column].fillna(df[column].mean())


# In[9]:


# checking for null values
df.isnull().mean()


# In[10]:


#filling null values for categorical data
df1=df.fillna(method='pad')


# In[11]:


df1.isnull().sum()


# In[12]:


df1.isnull().mean()


# In[13]:


df1.head()


# In[14]:


df1.describe()


# In[15]:


# to check for duplicates
df1.duplicated().sum()


# In[16]:


# removing duplicates
df1.drop_duplicates(inplace=True)


# In[17]:


df1


# In[18]:


df1.duplicated()


# In[19]:


df1.duplicated().sum()


# In[20]:


df1.describe()


# In[21]:


# finding unique values
for column in df1.columns:
    print(df1[column].nunique())


# In[22]:


# to check the object type/categorical data
for column in df1.columns:
    if df1[column].dtype==object:
        print("{}:{}".format(column,df1[column].unique()))


# In[23]:


# to check the numerical type/continous data
for column in df1.columns:
    if df1[column].dtype==float:
        print("{}:{}".format(column,df1[column].unique()))


# In[24]:


df1.columns


# In[25]:


# to check if the data is balance or not

df1['RainTomorrow'].value_counts()


# In[26]:


# visualizing the data RainTomorrow
sns.countplot(x=df1['RainTomorrow'])


# In[27]:


# checking the numerical and categorical columns
numeric_col=list(df1.select_dtypes(include=np.number).columns)
categorical_col=list(df1.select_dtypes(include=object).columns)


# In[28]:


numeric_col


# In[29]:


categorical_col


# In[30]:


# visualizing numeric data
for col in numeric_col:
    sns.histplot(x=df1[col], palette='PuBu',color='red')
    plt.show()


# In[31]:


# visualizing categorical data
for col in categorical_col:
    sns.histplot(x=df1[col], palette='PuBu', color='orange')
    plt.show()


# In[32]:


# using barplot to check relation between Humidity and RainTomorrow
sns.barplot(data=df1, x='RainTomorrow', y='Humidity9am', color='blue')
print(plt.show())


# In[33]:


# using barplot to check relation between Humidity and RainTomorrow
sns.barplot(data=df1, x='RainTomorrow', y='Humidity3pm', color='blue')
print(plt.show())


# In[34]:


# using barplot to check relation between Rainfall and RainTomorrow
sns.barplot(data=df1, x='RainTomorrow', y='Rainfall', color='violet')
print(plt.show())


# In[35]:


# using barplot to check relation between Location and Rainfall
sns.barplot(data=df1, x='Location', y='Rainfall', color='green')
print(plt.show())


# In[36]:


# Rainfall based on Location 
df1.groupby('Location')['Rainfall'].value_counts()


# In[37]:


# Location based on sunshine 
df1.groupby('Location')['Sunshine'].value_counts()


# In[38]:


# Location based on Humidity9am
df1.groupby('Location')['Humidity9am'].value_counts()


# In[39]:


# Location based on Humidity3pm
df1.groupby('Location')['Humidity3pm'].value_counts()


# In[40]:


# Data Preprocessing for classification problem
X=df1.drop(['RainTomorrow'], axis=1)
y=df1['RainTomorrow']


# In[41]:


X


# In[42]:


y


# In[43]:


# converting categorical data into numerical data using Label Encoder
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)
y


# In[44]:


# converting categorical data into numerical data using get_dummies
X=pd.get_dummies(X,drop_first=True)
X


# In[45]:


# training and testing the data
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=1, test_size=0.3)


# In[46]:


#using scaler technique to fine tune features transformation technique
scaler=StandardScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test=pd.DataFrame(scaler.transform(X_test),columns=X_train.columns)


# In[47]:


X_train.head()


# In[48]:


X_test.head()


# In[49]:


# model building using Logistic Regression
lr=LogisticRegression()
lr.fit(X_train, y_train)
y_pred=lr.predict(X_test)
print("Accuracy on training data:{:,.3f}".format(lr.score(X_train,y_train)))
print("Accuracy on test data:{:,.3f}".format(lr.score(X_test,y_test)))


# In[50]:


y_pred


# In[51]:


# checking accuracy score on test data and classification report
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[52]:


metrics.confusion_matrix(y_test, y_pred)


# In[53]:


# ploting heatmap to show graphical representation
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True)


# In[54]:


# model building using Support Vector Machine
from sklearn import svm
from sklearn.svm import SVC


# In[55]:


sv=svm.SVC()
sv.fit(X_train, y_train)
y_pred=sv.predict(X_test)
print("Accuracy on training data:{:,.3f}".format(sv.score(X_train,y_train)))
print("Accuracy on test data:{:,.3f}".format(sv.score(X_test,y_test)))


# In[56]:


# checking accuracy score on test data and classification report
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[57]:


metrics.confusion_matrix(y_test, y_pred)


# In[58]:


# model building using Random Forest Classifier
rfc=RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)
print("Accuracy on training data:{:,.3f}".format(rfc.score(X_train,y_train)))
print("Accuracy on test data:{:,.3f}".format(rfc.score(X_test,y_test)))


# In[59]:


metrics.confusion_matrix(y_test, y_pred)


# In[60]:


# checking accuracy score on test data and classification report
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[61]:


# model building using decisiontree classifier
dtc=DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
dtc.fit(X_train, y_train)
y_pred=dtc.predict(X_test)
print("Accuracy on training data:{:,.3f}".format(dtc.score(X_train,y_train)))
print("Accuracy on test data:{:,.3f}".format(dtc.score(X_test,y_test)))


# In[62]:


metrics.confusion_matrix(y_test, y_pred)


# In[63]:


# checking accuracy score on test data and classification report
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[64]:


'''Conclusion: Comparing four models Suport Vector Machine(SVM) and Logistic Regression, Random Forest Classifier. Decision Tree Classifier.
Accuracy score on test data for Logistic Regression:0.839, Accuracy score on test data for Support Vector Machine: 0.845
Accuracy score on test data for Random Forest Classifier: 0.827. 
Accuracy score on test data for DecisionTreeClassifier: 0.860. So we can choose DecisionTreeClassifier for model building.'''

