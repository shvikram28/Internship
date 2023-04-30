#!/usr/bin/env python
# coding: utf-8

# # BANK MARKETING-TERM DEPOSIT PROJECT

# In[1]:


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


# In[2]:


#importing train data
train=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset5/main/termdeposit_train.csv')
train


# In[3]:


#importing test data
test=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset5/main/termdeposit_test.csv')
test


# In[4]:


test.isnull().sum()


# In[5]:


train.isnull().sum()


# In[6]:


train.describe()


# In[7]:


test.describe()


# In[8]:


test.info()


# In[9]:


train.info()


# In[10]:


display(train.shape, test.shape)


# In[11]:


# merging the train and test data
Deposit=pd.concat([train, test])
Deposit.shape


# In[12]:


Deposit


# In[64]:


# categorical features and its unique values
for col in Deposit.select_dtypes(include='object').columns:
    print(col)
    print(Deposit[col].unique())


# In[67]:


# find missing values
features_na=[features for features in Deposit.columns if Deposit[features].isnull().sum()>0]
for feature in features_na:
    print(feature, np.round(Deposit[feature].isnull().mean(), 4), '% missing values')
else:
    print("no missing values found")


# In[13]:


# find features with one value
for column in Deposit.columns:
    print(column,Deposit[column].nunique ())


# In[14]:


# explore the categorical features
categorical_features=[feature for feature in Deposit.columns if((Deposit[feature].dtypes=='O') &(feature not in['subscribed']))]
categorical_features


# In[15]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(Deposit[feature].unique())))


# In[16]:


# Categorical feature distribution
plt.figure(figsize=(15,80), facecolor='white')
plotnumber=1
for categorical_feature in categorical_features:
    ax=plt.subplot(12,3, plotnumber)
    sns.countplot(y=categorical_feature, data=Deposit)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()
    


# In[17]:


# relationship between categorical features and labels
# find out the relationship between categorical variable and dependent variable
for categorical_feature in categorical_features:
    sns.catplot(x='subscribed', col=categorical_feature, kind='count', data=Deposit)
plt.show()


# In[18]:


# check target label split over categorical features and find the count
for categorical_feature in categorical_features:
    print(Deposit.groupby(['subscribed', categorical_feature]).size())


# In[19]:


# explore the numerical features 
numerical_features=[feature for feature in Deposit.columns if((Deposit[feature].dtypes !='O') &(feature not in['subscribed']))]
print('number of numerical variables:', len(numerical_features))


# In[68]:


# visualize the numerical variables
Deposit[numerical_features]


# In[69]:


# find discrete numerical features
discrete_features=[feature for feature in numerical_features if len(Deposit[feature].unique())<25]
print("Discrete Variables Count: ()".format(len(discrete_features)))


# In[70]:


# find continous numerical features
continous_features=[feature for feature in numerical_features if feature not in discrete_features+['subscribed']]
print("continous feature Count: ()".format(len(continous_features)))


# In[71]:


# distribution of continous numerical features
plt.figure(figsize=(20,60), facecolor='white')
plotnumber=1
for continous_feature in continous_features:
    ax=plt.subplot(12,3,plotnumber)
    sns.distplot(Deposit[continous_features])
    plt.xlabel(continous_features)
    plotnumber+=1
plt.show()


# In[72]:


# boxplot to show target  distribution with respect numerical features
plt.figure(figsize=(20,60), facecolor='white')
plotnumber=1
for feature in continous_features:
    ax=plt.subplot(12,3,plotnumber)
    sns.boxplot(x="subscribed", y=Deposit[feature], data=Deposit)
    plt.xlabel(feature)
    plotnumber+=1
plt.show()


# In[25]:


# outliers in numerical features
plt.figure(figsize=(20,60), facecolor='orange')
plotnumber=1
for numerical_feature in numerical_features:
    ax=plt.subplot(12,3, plotnumber)
    sns.boxplot(Deposit[numerical_feature])
    plt.xlabel(numerical_feature)
    plotnumber+=1
plt.show()


# In[26]:


# explore the Correlation between numerical features
# checking for correlation
cor_mat=Deposit.corr()
fig=plt.figure(figsize=(15,7))
sns.heatmap(cor_mat,annot=True)



# In[27]:


# checking whether the data set is balanced or not based on target values in classification
# total patient count based on cardio_results
sns.countplot(x='subscribed', data=Deposit)
plt.show()


# In[28]:


Deposit['subscribed'].groupby(Deposit['subscribed']).count()


# In[29]:


# lets create a new dataframe using copy command
Deposit2=Deposit.copy()
Deposit2


# In[30]:


print(Deposit2.shape)
Deposit2.head()


# In[31]:


Deposit2.groupby(['subscribed','default']).size()


# In[32]:


Deposit2.drop(['default'], axis=1, inplace=True)


# In[33]:


Deposit2.groupby(['subscribed','pdays']).size()


# In[34]:


# drop pdays as it has -1 value for around 40%
Deposit2.drop(['pdays'], axis=1, inplace=True)


# In[35]:


# remove outliers in feature 'age'
Deposit2.groupby('age', sort=True)['age'].count()


# In[37]:


# remove outliers in feature balance
Deposit2.groupby(['subscribed','balance'], sort=True)['balance'].count()
# these ouliers should not be removed as balance gos high, client show interest on deposit


# In[38]:


# remove outlier in feature duration
Deposit2.groupby(['subscribed','duration'], sort=True)['duration'].count()


# In[39]:


# remove outliers in feature campaign
Deposit2.groupby(['subscribed', 'campaign'], sort=True)['campaign'].count()


# In[40]:


Deposit3=Deposit2[Deposit2['campaign']<33]


# In[41]:


Deposit3.groupby(['subscribed','campaign'], sort=True)['campaign'].count()


# In[42]:


# remove outlier in feature previous
Deposit3.groupby(['subscribed','previous'], sort=True)['previous'].count()


# In[74]:


Deposit4=Deposit3[Deposit3['previous']<31]


# In[75]:


Deposit4.groupby(['subscribed','previous'], sort=True)['previous'].count()


# In[76]:


cat_columns=['job', 'marital','education','contact','month','poutcome']
for col in cat_columns:
    Deposit4=pd.concat([Deposit4.drop(col, axis=1), pd.get_dummies(Deposit4[col],prefix=col, prefix_sep='_',drop_first=True, dummy_na=False)])
    


# In[77]:


Deposit4.head()


# In[78]:


bool_columns=['housing', 'loan', 'subscribed']
for col in bool_columns:
    Deposit4[col+'_new']=Deposit4[col].apply(lambda x :1 if x == 'yes' else 0)
    Deposit4.drop(col, axis=1, inplace=True)


# In[79]:


Deposit4.head()


# In[61]:


Deposit4.columns


# In[80]:


# split data into training and test set
X=Deposit4.drop(['subscribed_new'], axis=1)
y=Deposit4['subscribed_new']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=0)


# In[81]:


len(X_train)


# In[82]:


len(X_test)


# In[83]:


# dropping columns as it has nan values
Deposit4.drop(['job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'month_may', 'month_nov', 'month_oct', 'month_sep','poutcome_other','poutcome_success','poutcome_unknown'], axis=1, inplace=True)


# In[84]:


Deposit4


# In[85]:


# model Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.model_selection import cross_val_score
model_score=cross_val_score(estimator=RandomForestClassifier(), X=X_train, y=y_train, cv=5)
print(model_score)
print(model_score.mean())

