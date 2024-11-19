#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset from CSV
file_path = 'River.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Filter the dataset to keep only rows where 'Source' is 'River'
filtered_data = data[data['Source'] == 'River']

# Save the filtered data to a new CSV file
filtered_file_path = 'Filtered_River.csv'
filtered_data.to_csv(filtered_file_path, index=False)

print(f"Filtered data saved to {filtered_file_path}")


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC  # SVM Algorithm
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv('River.csv')

# Drop the Index column if it exists
data.drop(columns=['Index'], inplace=True, errors='ignore')

# Define feature columns and target variable
features = ['pH', 'Nitrate', 'Turbidity', 'Odor', 'Water Temperature']
target = 'Target'

# Separate features and target variable
X = data[features]
y = data[target]

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, features)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the SVM model with preprocessing pipeline
svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', probability=True))  # SVM with linear kernel
])

# Train the model
svm_pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = svm_pipeline.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred))

