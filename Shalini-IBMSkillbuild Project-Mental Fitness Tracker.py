#!/usr/bin/env python
# coding: utf-8

# #### IMPORTING REQUIRED LIBRARIES

# In[1]:


# Import the required libraries and load the dataset.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# #### READING DATA

# In[2]:


# Load the mental health dataset
d1 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')
d2 = pd.read_csv('prevalence-by-mental-and-substance-use-disorder.csv')


# In[3]:


d1.head(10)


# In[4]:


d2.head(10)


# #### MERGING TWO DATASETS

# In[5]:


#Merging the Datasets for optimal results
data = pd.merge(d1,d2)
data


# #### DATA CLEANING

# In[6]:


#Cleaning the Dataset
data.isnull().sum()


# In[7]:


#Since code column is not required, we can drop it.
data.drop('Code',axis=1,inplace=True)


# In[8]:


data.isnull().sum()


# In[9]:


data


# In[10]:


#size = row*column
data.size


# In[11]:


#shape = tuple of array dimension (row, column)
data.shape


# In[12]:


#set axis
data.set_axis(['Country','Year','DALY','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol'], axis='columns', inplace=True)


# In[13]:


data


# #### EXPLORATORY ANALYSIS

# In[14]:


plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True,cmap='Greens')
plt.plot()


# In[15]:


#Pairplot
sns.pairplot(data,corner=True)
plt.show()


# In[16]:


mean = data['DALY'].mean()
mean


# In[17]:


pip install plotly


# In[18]:


import plotly.express as px
fig = px.pie(data, values='DALY', names='Year')
fig.show()


# #### YEARWISE VARIATIONS IN MENTAL FITNESS OF DIFFERENT COUNTRIES

# In[19]:


fig=px.bar(data,x='Year',y='DALY',color='Country',template='ggplot2')
fig.show()


# #### ENCODING THE CATEGORICAL VALUES

# In[20]:


data.info()


# In[21]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':
        data[i]=l.fit_transform(data[i])

data.head()


# #### SPLITTING THE DATASET INTO TRAINING AND TESTING SET

# In[22]:


# Split the dataset into features (X) and target variable (y)
X = data.drop('DALY',axis=1)
y = data['DALY']


# In[23]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### MODEL TRAINING

# ##### 1.Decision Tree Regression

# In[24]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(X_train,y_train)

#predicting the value for training set
ytrain_pred=dtr.predict(X_train)

# model evaluation for training set
from sklearn.metrics import r2_score,mean_squared_error
print("The model performance for training set:\n1)Mean Square Error={}\n2)RMSE = {}\n3)R2_Score = {}".format(mean_squared_error(y_train,ytrain_pred),np.sqrt(mean_squared_error(y_train, ytrain_pred)),r2_score(y_train,ytrain_pred)))


#predicting the value for testing set
ytest_pred=dtr.predict(X_test)

# model evaluation for testing set
print("The model performance for testing set:\n1)Mean Square Error={}\n2)RMSE = {}\n3)R2_Score = {}".format(mean_squared_error(y_test,ytest_pred),np.sqrt(mean_squared_error(y_test, ytest_pred)),r2_score(y_test,ytest_pred)))


# ##### 2.Random Forest Regression

# In[25]:


from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_estimators=10,random_state=42)
rfr.fit(X_train,y_train)

#predicting the value for training set
ytrain_pred=rfr.predict(X_train)

# model evaluation for training set
print("The model performance for training set:\n1)Mean Square Error={}\n2)RMSE = {}\n3)R2_Score = {}".format(mean_squared_error(y_train,ytrain_pred),np.sqrt(mean_squared_error(y_train, ytrain_pred)),r2_score(y_train,ytrain_pred)))


#predicting the value for testing set
ytest_pred=rfr.predict(X_test)

# model evaluation for testing set
print("The model performance for testing set:\n1)Mean Square Error={}\n2)RMSE = {}\n3)R2_Score = {}".format(mean_squared_error(y_test,ytest_pred),np.sqrt(mean_squared_error(y_test, ytest_pred)),r2_score(y_test,ytest_pred)))


# ###### Random Forest Regression works well on both train and test sets with r2 score of 0.99.
# ###### As well as Decision Tree Regression also works well on both train and test set with r2 score of 0.98.

# In[26]:


# Real-Time Tracking (Example: Predict mental fitness label for a new data point)
features = ['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol']
new_data = pd.DataFrame([[0,2023,0.22,0.34,0.40,0.34,0.65,0.52,0.22]], columns=features)
prediction = rfr.predict(new_data)
print('Predicted Mental Fitness Label:', prediction)


# In[ ]:




