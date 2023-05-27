#!/usr/bin/env python
# coding: utf-8

# # Loan Application Status Prediction

# In[2]:


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
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from collections import Counter
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv')
df


# In[4]:


# check for the data types
df.info()


# In[5]:


# checking for null values
df.isnull().sum()


# In[6]:


# checking for null values
df.isnull().mean()


# In[7]:


# deleting columns which are not of use
df=df.drop(['Loan_ID'], axis=1)
df


# In[8]:


df.columns


# In[9]:


df.isnull().mean()


# In[10]:


#filling null values for numerical data
for column in['LoanAmount','Loan_Amount_Term','Credit_History']:
    df[column]=df[column].fillna(df[column].mean())


# In[11]:


# again checking for null values
df.isnull().mean()


# In[12]:


#filling null values for categorical data
df1=df.fillna(method='pad')


# In[13]:


# again checking for null values
df1.isnull().mean()


# In[14]:


df1


# In[15]:


df1.describe()


# In[16]:


df1.shape


# In[17]:


df1.duplicated()


# In[18]:


# Checking for duplicate values
df1.duplicated().sum()


# In[19]:


# finding unique values
for column in df.columns:
    print(df1[column].nunique())


# In[20]:


# to check the object type/categorical data
for column in df1.columns:
    if df1[column].dtype==object:
        print("{}:{}".format(column,df1[column].unique()))


# In[21]:


# to check if the data is balance or not
df1['Loan_Status'].value_counts()


# In[22]:


# visualizing the data Loan_Status
sns.countplot(x=df1['Loan_Status'])


# In[23]:


# checking the numerical and categorical columns
numeric_col=list(df1.select_dtypes(include=np.number).columns)
categorical_col=list(df1.select_dtypes(include=object).columns).


# In[24]:


numeric_col


# In[25]:


categorical_col


# In[28]:


# visualizing numeric data
for col in numeric_col:
    sns.histplot(x=df1[col], palette='PuBu', color='red')
    plt.show()


# In[29]:


# visualizing categorical data
for col in categorical_col:
    sns.histplot(x=df1[col], palette='PuBu', color='green')
    plt.show()


# In[30]:


# using scatterplot to check relation between Loan Status and Gender
sns.scatterplot(data=df1, x='Loan_Status',y='Gender', color='purple')
plt.show()


# In[31]:


# Loan Status based on Gender
df.groupby('Gender')['Loan_Status'].value_counts()


# In[32]:


# Loan Status based on Education
df.groupby('Education')['Loan_Status'].value_counts()


# In[33]:


# Loan Status based on Self_Employed
df.groupby('Self_Employed')['Loan_Status'].value_counts()


# In[34]:


# Loan Status based on Self_Employed
df.groupby('Credit_History')['Loan_Status'].value_counts()


# In[35]:


# Loan Status based on Property_Area
df.groupby('Property_Area')['Loan_Status'].value_counts()


# In[38]:


# using histplot to check relation between Loan Status and Property Area
sns.histplot(data=df1, x='Loan_Status',y='Property_Area', color='purple')
plt.show()


# In[74]:


# to check the outliers for the given data set
df.plot(kind='box', figsize=(12,12), layout=(3,3), sharex=False, subplots=True, color='red')


# In[39]:


# Data Preprocessing
X=df1.drop(['Loan_Status'], axis=1)
y=df1['Loan_Status']


# In[40]:


X


# In[41]:


y


# In[42]:


# converting categorical data into numerical data using Label Encoder Loan_Status data
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)
y


# In[43]:


# converting categorical data into numerical data using get_dummies
X=pd.get_dummies(X,drop_first=True)
X


# In[44]:


# training and testing the data
X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=1, test_size=0.3)


# In[45]:


#using scaler technique to fine tune features transformation technique
scaler=StandardScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test=pd.DataFrame(scaler.fit_transform(X_test),columns=X_train.columns)


# In[46]:


X_train.head()


# In[47]:


X_test


# In[48]:


y_test.sum()


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


metrics.confusion_matrix(y_test, y_pred)


# In[52]:


# ploting heatmap to show graphical represntation
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True)


# In[53]:


# checking accuracy score on test data and classification report
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[54]:


# model building using Support Vector Machine
from sklearn import svm
from sklearn.svm import SVC


# In[55]:


sv=svm.SVC()
sv.fit(X_train, y_train)
y_pred=sv.predict(X_test)
print("Accuracy on training data:{:,.3f}".format(sv.score(X_train, y_train)))
print("Accuracy on training data:{:,.3f}".format(sv.score(X_test,y_test)))
                                                 


# In[56]:


metrics.confusion_matrix(y_test,y_pred)


# In[57]:


# ploting heatmap to show graphical represntation
sns.heatmap(metrics.confusion_matrix(y_test,y_pred), annot=True)


# In[58]:


# checking accuracy score on test data and classification report
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))


# In[65]:


# model building using Random Forest Classifier
rfc=RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)
print("Accuracy on training data:{:,.3f}".format(rfc.score(X_train,y_train)))
print("Accuracy on test data:{:,.3f}".format(rfc.score(X_test,y_test)))


# In[66]:


metrics.confusion_matrix(y_test, y_pred)


# In[67]:


# ploting heatmap to show graphical representation
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True)


# In[68]:


# checking accuracy score on test data and classification report
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))


# In[69]:


# model building using decisiontree classifier
dtc=DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
dtc.fit(X_train, y_train)
y_pred=dtc.predict(X_test)
print("Accuracy on training data:{:,.3f}".format(dtc.score(X_train,y_train)))
print("Accuracy on test data:{:,.3f}".format(dtc.score(X_test,y_test)))


# In[70]:


metrics.confusion_matrix(y_test, y_pred)


# In[71]:


# checking accuracy score on test data and classification report
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[72]:


'''Conclusion: Comparing four models Suport Vector Machine(SVM) and Logistic Regression, Random Forest Classifier and DecisionTreeClassifier.
Accuracy score on test data for Logistic Regression:0.795, Accuracy score on test data for Support Vector Machine: 0.789
Accuracy score on test data for Random Forest Classifier: 0.789. Accuracy score on test data for DecisionTreeClassifier 0.724.
So we can choose Logistic Regression for model building as it's accuracy score is more.'''


# In[ ]:




