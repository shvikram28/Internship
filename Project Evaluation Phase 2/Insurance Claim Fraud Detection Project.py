#!/usr/bin/env python
# coding: utf-8

# # Insurance Claim Fraud Detection

# In[2]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from imblearn.combine import SMOTEENN
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from collections import Counter
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/Automobile_insurance_fraud.csv')
df


# In[4]:


# check for the data types
df.info()


# In[6]:


df.columns


# In[7]:


df.isnull().mean()


# In[9]:


df.describe()


# In[10]:


# finding unique values
for column in df.columns:
    print(df[column].nunique())


# In[11]:


# Checking for duplicate values
df.duplicated().sum()


# In[12]:


# to check the object type/categorical data
for column in df.columns:
    if df[column].dtype==object:
        print("{}:{}".format(column,df[column].unique()))


# In[15]:


# checking the numerical and categorical columns
numeric_col=list(df.select_dtypes(include=np.number).columns)
categorical_col=list(df.select_dtypes(include=object).columns)


# In[18]:


numeric_col


# In[19]:


categorical_col


# In[17]:


# visualizing numeric data
for col in numeric_col:
    sns.histplot(x=df[col], palette='PuBu', color='red')
    plt.show()


# In[20]:


# visualizing categorical data
for col in categorical_col:
    sns.histplot(x=df[col], palette='PuBu', color='green')
    plt.show()


# In[23]:


# Data Preprocessing
X=df.drop(['fraud_reported'], axis=1)
y=df['fraud_reported']


# In[24]:


X


# In[25]:


y


# In[26]:


# converting categorical data into numerical data using Label Encoder Loan_Status data
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)
y


# In[27]:


# converting categorical data into numerical data using get_dummies
X=pd.get_dummies(X,drop_first=True)
X


# In[28]:


# training and testing the data
X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=1, test_size=0.3)


# In[29]:


#using scaler technique to fine tune features transformation technique
scaler=StandardScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test=pd.DataFrame(scaler.fit_transform(X_test),columns=X_train.columns)


# In[30]:


X_train.head()


# In[31]:


X_test


# In[32]:


y_test.sum()


# In[43]:


#filling null values for numerical data
for column in['months_as_customer',
 'age',
 'policy_number',
 'policy_deductable',
 'policy_annual_premium',
 'umbrella_limit',
 'insured_zip',
 'capital-gains',
 'capital-loss',
 'incident_hour_of_the_day',
 'number_of_vehicles_involved',
 'bodily_injuries',
 'witnesses',
 'total_claim_amount',
 'injury_claim',
 'property_claim',
 'vehicle_claim',
 'auto_year',
 '_c39']:X[column]=X[column].fillna(X[column].mean())


# In[44]:


#filling null values for categorical data
X=X.fillna(method='pad')


# In[ ]:




