#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Linear Regression
# #### Linear regression is a type of supervised machine learning algorithm that computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation to observed data. 
# When there is only one independent feature, it is known as Simple Linear Regression, and when there are more than one feature, it is known as Multiple Linear Regression.

# In[2]:


df = pd.read_csv("student_scores.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.value_counts()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[11]:


plt.scatter(hours,scores)
plt.show()


# In[21]:


hours.size
x = df['Hours']
y = df['Scores']


# In[26]:


learning_rate = 0.01  #step size for gradient descent
m=len(hours) # number of iterations


# In[28]:


# initaialize co-efficients / parameters
w = 0 #slope
b = 0 # intercept


# In[27]:


# gradient descent algorthim
for i in range(m):
    y_pred = w*x + b # predictions
    error = y_pred - y # error

    cost_function = (1/2*m)*np.sum(error ** 2) # cost function(rmse)
# gradient of the cost function
    w_gradient = (1/m) * np.sum(error * x)
    b_gradient = (1/m) * np.sum(error)
# update co-efficients
    w -= learning_rate * w_gradient
    b -= learning_rate * b_gradient
print(f"final slope: {w}")
print(f"final intercept: {b}")
# predicton function
def predict(x):
    return w*x + b
# visualization
y_pred = predict(x)
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Fitted line')
plt.title('hours of study vs scores students got')
plt.xlabel('hours of study')
plt.ylabel('scores')
plt.legend()
plt.show()

