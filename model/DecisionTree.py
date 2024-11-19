#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# In[2]:


df = pd.read_csv('River.csv')

df.info()
df.head()


# In[3]:


df_filtered = df[['pH', 'Nitrate', 'Color', 'Turbidity', 'Odor', 'Chlorine', 'Total Dissolved Solids', 'Water Temperature', 'Target']]

df_filtered.isnull().sum()


# In[4]:


df_filtered = df_filtered.dropna()


# In[5]:


categorical_cols = ['Color', 'Odor']

le = LabelEncoder()

for col in categorical_cols:
    df_filtered[col] = le.fit_transform(df_filtered[col])

for col in categorical_cols:
    print(f"\n{col} Encoding Mapping:")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)


# EDA

# In[6]:


df_filtered.describe()


# In[7]:


df_filtered.hist(figsize=(10, 10), bins=20)
plt.show()


# In[8]:


plt.figure(figsize=(10, 8))
sns.heatmap(df_filtered.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[9]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_cols = ['pH', 'Nitrate', 'Turbidity', 'Chlorine', 
                  'Total Dissolved Solids', 'Water Temperature']

df_filtered[numerical_cols] = scaler.fit_transform(df_filtered[numerical_cols])


# In[10]:


X = df_filtered.drop('Target', axis=1)
y = df_filtered['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Set Size: {X_train.shape}")
print(f"Testing Set Size: {X_test.shape}")


# Decision Tree Classifier

# In[11]:


model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)


# In[12]:


y_pred = model.predict(X_test)

#accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[13]:


# Feature importance visualization
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='y', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# SVM
# Save model in DecisionTree.py
import joblib
joblib.dump(model, 'decision_tree_model.pkl')

# In[ ]:




