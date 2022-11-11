#!/usr/bin/env python
# coding: utf-8

# In[23]:


#importing the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[24]:


#reading the datasets
companies=pd.read_csv("1000_Companies.csv")
x=companies.iloc[:, :-1].values                
y=companies.iloc[:,4].values


# In[25]:


#displaying the data
companies.head()


# In[26]:


#display in visualise manner
sns.heatmap(companies.corr())


# In[27]:


#Here we are coverting normal string data into integer to get the proper result
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])
# transform = make_column_transformer((OneHotEncoder(), x[:, 3]), remainder = 'passthrough')

onehotencoder = OneHotEncoder()
enc_data = onehotencoder.fit_transform(x).toarray()


# In[10]:


companies.


# In[31]:


#avoid dummy values
x=x[:,1:]


# In[52]:


#training the model using sklearn 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2  , random_state = 0)


# In[53]:


#Here we are training using LinearRegression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[54]:


#to see the ouput after converting string into int
y_pred=regressor.predict(x_test)

y_pred


# In[55]:


#to check the linearRegression coefficient
print(regressor.coef_)


# In[56]:


#to check the linearRegression Intercept
print(regressor.intercept_)


# In[57]:


#we can see the output
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:




