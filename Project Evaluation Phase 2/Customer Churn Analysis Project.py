#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Analysis

# In[1]:


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


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/Telecom_customer_churn.csv')
df


# In[3]:


# to check for the data types
df.info()


# In[4]:


df.dtypes


# In[5]:


# since TotalCharges has continuous data but it is an object type. Let us handle this column.
df['TotalCharges'].unique()


# In[6]:


df['TotalCharges'].nunique()


# In[7]:


# to chek the blank spaces
df.loc[df['TotalCharges']==" "]


# In[8]:


# replacing wide spaces in rows with nan values
df['TotalCharges']=df['TotalCharges'].replace(" ",np.nan)


# In[9]:


# checking for null values
df.isnull().sum()


# In[10]:


# converting the column type from object to float
df['TotalCharges']=df['TotalCharges'].astype(float)


# In[11]:


#checking the data type of TotalCharges
df['TotalCharges'].dtype


# In[12]:


# handling the nan values of TotalCharges column(Target Variable)
np.mean(df['TotalCharges'])


# In[13]:


# checking the iloc value
df.iloc[488,:]


# In[14]:


# filling up the nan values using mean method
df['TotalCharges']=df['TotalCharges'].fillna(np.mean(df['TotalCharges']))


# In[15]:


df['TotalCharges'].isnull().sum()


# In[16]:


# checking the iloc value
df.iloc[488,:]


# In[17]:


# checking for null values
df.isnull().sum()


# In[18]:


'''since there are no null values we can proceed and in this data we have to predict total charges.'''


# In[19]:


df.dtypes


# In[20]:


# checking for duplicated values
df.duplicated().sum()


# In[21]:


# making DataFrame for the nominal Data
df_vizualization_nominal=df[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'Churn']].copy()


# In[22]:


df_vizualization_nominal.columns


# In[23]:


# vizualization of the data
ax=sns.countplot(x='gender', data=df_vizualization_nominal)
print(df_vizualization_nominal['gender'].value_counts())


# In[24]:


# vizualization of the data
ax=sns.countplot(x='SeniorCitizen', data=df_vizualization_nominal)
print(df_vizualization_nominal['SeniorCitizen'].value_counts())


# In[25]:


'''1142 senior citizens are using services'''


# In[26]:


# vizualization of the data
ax=sns.countplot(x='Partner', data=df_vizualization_nominal)
print(df_vizualization_nominal['Partner'].value_counts())


# In[27]:


# vizualization of the data
ax=sns.countplot(x='Dependents', data=df_vizualization_nominal)
print(df_vizualization_nominal['Dependents'].value_counts())


# In[28]:


# vizualization of the data
ax=sns.countplot(x='PhoneService', data=df_vizualization_nominal)
print(df_vizualization_nominal['PhoneService'].value_counts())


# In[29]:


# vizualization of the data
ax=sns.countplot(x='MultipleLines', data=df_vizualization_nominal)
print(df_vizualization_nominal['MultipleLines'].value_counts())


# In[30]:


# vizualization of the data
ax=sns.countplot(x='InternetService', data=df_vizualization_nominal)
print(df_vizualization_nominal['InternetService'].value_counts())


# In[31]:


# vizualization of the data
ax=sns.countplot(x='OnlineSecurity', data=df_vizualization_nominal)
print(df_vizualization_nominal['OnlineSecurity'].value_counts())


# In[32]:


# vizualization of the data
ax=sns.countplot(x='OnlineBackup', data=df_vizualization_nominal)
print(df_vizualization_nominal['OnlineBackup'].value_counts())


# In[33]:


# vizualization of the data
ax=sns.countplot(x='DeviceProtection', data=df_vizualization_nominal)
print(df_vizualization_nominal['DeviceProtection'].value_counts())


# In[34]:


# vizualization of the data
ax=sns.countplot(x='TechSupport', data=df_vizualization_nominal)
print(df_vizualization_nominal['TechSupport'].value_counts())


# In[35]:


# vizualization of the data
ax=sns.countplot(x='StreamingTV', data=df_vizualization_nominal)
print(df_vizualization_nominal['StreamingTV'].value_counts())


# In[36]:


# vizualization of the data
ax=sns.countplot(x='Contract', data=df_vizualization_nominal)
print(df_vizualization_nominal['Contract'].value_counts())


# In[37]:


# vizualization of the data
ax=sns.countplot(x='PaperlessBilling', data=df_vizualization_nominal)
print(df_vizualization_nominal['PaperlessBilling'].value_counts())


# In[38]:


# vizualization of the data
ax=sns.countplot(x='PaymentMethod', data=df_vizualization_nominal)
print(df_vizualization_nominal['PaymentMethod'].value_counts())


# In[39]:


# vizualization of the data
ax=sns.countplot(x='Churn', data=df_vizualization_nominal)
print(df_vizualization_nominal['Churn'].value_counts())


# In[40]:


# making dataframe of the ordinal data
df_vizualization_ordinal=df[["customerID","tenure"]].copy()
df_vizualization_ordinal.columns


# In[41]:


# vizualization of the data
sns.catplot(x='SeniorCitizen',y='tenure',data=df)


# In[42]:


# checking the distribution of the continous value of the float type columns
df_vizualization_continuous=df[['MonthlyCharges','TotalCharges']].copy()


# In[43]:


sns.distplot(df_vizualization_continuous['TotalCharges'], kde=True)


# In[44]:


sns.distplot(df_vizualization_continuous['MonthlyCharges'], kde=True)


# In[45]:


# since the data has lot of string values we will going to convert string data to numerical data using encoding techniques
from sklearn.preprocessing import OrdinalEncoder
enc=OrdinalEncoder()


# In[46]:


for i in df.columns:
    if df[i].dtypes=='object':
        df[i]=enc.fit_transform(df[i].values.reshape(-1,1))
    


# In[47]:


df


# In[48]:


df.describe()


# In[49]:


# visualization of describe data using heatmap
plt.figure(figsize=(22,7))
sns.heatmap(df.describe(), annot=True, linewidths=0.1, linecolor='black',fmt="0.2f")


# In[50]:


# checking the correlation
df.corr()['TotalCharges'].sort_values()


# In[51]:


# visualization of correlation data using heatmap
plt.figure(figsize=(22,7))
sns.heatmap(df.corr(), annot=True, linewidths=0.1, linecolor='black',fmt="0.2f")


# In[52]:


# using barplot to check correlation value
plt.figure(figsize=(24,7))
df.corr()['TotalCharges'].sort_values(ascending=False).drop(['TotalCharges']).plot(kind='bar',color='c')
plt.xlabel('feature', fontsize=14)
plt.title('correlation', fontsize=18)
plt.show()


# In[53]:


df.skew()


# In[54]:


# checking the outliers on int and float type of data columns
df['SeniorCitizen'].plot.box()


# In[55]:


df['TotalCharges'].plot.box()


# In[56]:


df['MonthlyCharges'].plot.box()


# In[57]:


df['tenure'].plot.box()


# In[58]:


# removing the outlier
from scipy.stats import zscore
import numpy as np
z=np.abs(zscore(df))
threshold=3
np.where(z>3)


# In[59]:


df_new_z=df[(z<3).all(axis=1)]
df_new_z


# In[60]:


df_new_z.shape


# In[61]:


# percentage loss of data
Data_loss=((7043-6361)/7043)*100
Data_loss


# In[62]:


# Separating the columns into features and target
features=df.drop('TotalCharges', axis=1)
target=df['TotalCharges']


# In[63]:


features


# In[64]:


target


# In[65]:


# scaling the data min-max scaler
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[66]:


for i in range(0,100):
    features_train, features_test, target_train, target_test=train_test_split(features,target,random_state=i, test_size=0.2)
    lr.fit(features_train,target_train)
    pred_train=lr.predict(features_train)
    pred_test=lr.predict(features_test)
    print(f"At random state (i), the training accuracy is: {r2_score(target_train, pred_train)}")
    print(f"At random state (i), the testing accuracy is: {r2_score(target_test, pred_test)}")
    print("\n")
    
    


# In[67]:


features_train, features_test, target_train, target_test=train_test_split(features,target,random_state=12, test_size=0.2)


# In[68]:


# train the model
lr.fit(features_train, target_train)


# In[69]:


pred_test=lr.predict(features_test)
print(r2_score(target_test, pred_test))


# In[70]:


# cross validation of the model:
Train_accuracy=r2_score(target_train, pred_train)
Test_accuracy=r2_score(target_test, pred_test)
from sklearn.model_selection import cross_val_score
for j in range(2,10):
    cv_score=cross_val_score(lr,features, target, cv=j)
    cv_mean=cv_score.mean()
    print(f"at cross fold(j) the cv score is {cv_mean} and accuracy score for training is {Train_accuracy} and accuracy for the testing is {Test_accuracy}")
    print("\n")
    


# In[71]:


''' since the number of folds don't have such impact on the accuracy and the cv_score. so cv=5 is selected.
Hnadled the problem of the overfitting and the underfitting by checking the training and testing score'''


# In[72]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(x=target_test, y=pred_test, color='r')
plt.plot(target_test, target_test, color='b')
plt.xlabel('Actual charges', fontsize=14)
plt.ylabel('Predicted charges', fontsize=14)
plt.title('Linear Regression', fontsize=18)
plt.savefig('lr.png')
plt.show()


# In[73]:


# Regularization
from sklearn.linear_model import Lasso
parameters={'alpha':[.0001, .001, .01, .1, 1, 10], 'random_state':list(range(0,10))}
ls=Lasso()
clf=GridSearchCV(ls,parameters)
clf.fit(features_train, target_train)
print(clf.best_params_)


# In[74]:


# final model training
ls=Lasso(alpha=1,random_state=0)
ls.fit(features_train, target_train)
ls_score_training=ls.score(features_train, target_train)
pred_ls=ls.predict(features_test)
ls_score_training*100


# In[75]:


pred_ls=ls.predict(features_test)
lss=r2_score(target_test, pred_ls)
lss*100


# In[76]:


cv_score=cross_val_score(ls,features, target, cv=5)
cv_mean=cv_score.mean()
cv_mean*100


# In[77]:


# ensemble technique
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

parameters={'criterion':['friedman_mse','mae'],
           'max_features':['auto','sqrt','log2']}
rf=RandomForestRegressor()
clf=GridSearchCV(rf,parameters)
clf.fit(features_train,target_train)
print(clf.best_params_)


# In[78]:


# ensemble technique
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(criterion="friedman_mse", max_features="auto")
rf.fit(features_train, target_train)
rf.score(features_train, target_train)
pred_decision=rf.predict(features_test)
rfs=r2_score(target_test, pred_decision)
print('R2 Score:', rfs*100)
rfscore=cross_val_score(rf, features, target, cv=5)
rfc=rfscore.mean()
print('Cross Val Score:', rfc*100)


# In[79]:


''' Conclusion: We are getting model accuracy and cross validation both as 99.8% which shows our model is performing well'''


# In[ ]:




