#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import string as str


# In[2]:


# load data
data = pd.read_csv('combined_samples_gt.csv')

# peek data
data.head()


# In[3]:


# drop nan
data = data.dropna()


# In[4]:


# initialize X and y
X = data['Tweet']
y = data['Ground Truth']


# In[5]:


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)


# In[6]:


# create logistic regression model
model = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])


# In[7]:


# fit data
model.fit(X_train,y_train)


# In[8]:


# insert model predictions
data['Predictions'] = ''
for i in (data.index - 1):
    data.iloc[i, 10] = model.predict([data.iloc[i, 1]])

data.head()


# In[9]:


# match column
data['Match'] = 'True'

for i in (data.index - 1):
    if (data['Predictions'].iloc[i] != data['Ground Truth'].iloc[i]):
        data['Match'].iloc[i] = False
        
data.head()


# In[10]:


# count correct and incorrect
result = data['Match'].value_counts()

# find correct count
correct = result.iloc[0]

# find incorrect count
incorrect = result.iloc[1]

# find total
total = correct + incorrect

# find accuracy
accuracy = ((correct / total) * 100).round(2)

# print accuracy
print("Model accuracy is: {}%".format(accuracy))


# In[11]:


# export dataframe
# data.to_csv('LR Model Trained Data.csv', index = False)


# In[12]:


# Get the original dataset and run the model prediction on the Tweets (non-samples)
original_df = pd.read_csv("original_dataset.csv", encoding = "ISO-8859-1")

original_df = original_df.dropna()
original_df['Predictions'] = ''
original_df.head(30)


# In[13]:


# insert model predictions
for i, row in original_df.iterrows():
    original_df.at[i, 'Predictions'] = model.predict([row['Tweet']])

original_df.head(30)


# In[14]:


original_df.to_csv('LR_full_predictions.csv', index = False)

