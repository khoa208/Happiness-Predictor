import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle

# Load the preprocessed data
data = pd.read_excel('final_cleaned_happiness_data.xlsx')

# Define Target and Features
# Assuming 'Happiness Score' is the target variable
X = data.drop(columns=['Happiness Score'])  # Features
y = data['Happiness Score']                 # Target

# Convert Categorical Variables to Numeric Using One-Hot Encoding
# Check for any non-numeric columns and apply one-hot encoding
X = pd.get_dummies(X, drop_first=True)  # Converts categorical columns to numeric

# Split Data into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection for Reduced Feature Set
# Using SelectKBest to select top features
selector = SelectKBest(score_func=f_regression, k=5)  # Choose 'k' based on your preference
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

# Build and Train Models
# Model with Full Feature Set
model_full = LinearRegression()
model_full.fit(X_train, y_train)

# Model with Reduced Feature Set
model_reduced = LinearRegression()
model_reduced.fit(X_train_reduced, y_train)

# Save Models as .model Files
# Saving the full feature model
with open('finalized_model_full.model', 'wb') as file:
    pickle.dump(model_full, file)

# Saving the reduced feature model
with open('finalized_model_reduced.model', 'wb') as file:
    pickle.dump(model_reduced, file)

# Load and Test Models (Optional)
# Loading the full feature model
loaded_model_full = pickle.load(open('finalized_model_full.model', 'rb'))
print("Score of full feature model:", loaded_model_full.score(X_test, y_test))

# Loading the reduced feature model
loaded_model_reduced = pickle.load(open('finalized_model_reduced.model', 'rb'))
print("Score of reduced feature model:", loaded_model_reduced.score(X_test_reduced, y_test))
