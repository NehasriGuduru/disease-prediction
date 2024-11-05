#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


# Load dataset
data = pd.read_csv('medical_data.csv')

data.head(10)


# In[5]:


# Preprocessing
X = data.drop('target', axis=1)  
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[6]:


# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[7]:


# Predictions
y_pred = model.predict(X_test)


# In[8]:


# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
# Count plot for a categorical feature (e.g., symptoms)
plt.figure(figsize=(12, 6))
sns.countplot(y='chills', data=data)  # Replace 'symptom' with your symptom column name
plt.title('Count of Symptoms')
plt.xlabel('Count')
plt.ylabel('Symptom')
plt.show()


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select relevant features for the correlation matrix
selected_features = ['chills', 'itching', 'toxic_look_(typhos)', 'polyuria']

# Calculate correlation matrix
correlation_matrix = data[selected_features].corr()

# Generate a heatmap for correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Multiple Features')
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

# Select relevant features
selected_features = ['increased_appetite', 'chills', 'itching', 'mucoid_sputum']  

# Data for radar chart
data_values = data[selected_features].mean().values

# Number of variables/features
num_vars = len(selected_features)

# Angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is a circle, so we need to "complete the loop" by appending the start to the end.
data_values = np.concatenate((data_values, [data_values[0]]))
angles += angles[:1]

# Radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, data_values, color='blue', alpha=0.25)
ax.plot(angles, data_values, color='blue', linewidth=2)

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(selected_features)

plt.title('Radar Chart for Multiple Features')
plt.show()


# In[ ]:




