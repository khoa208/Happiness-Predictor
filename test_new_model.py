import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import numpy as np
import pickle

# Load the preprocessed data
data = pd.read_excel('final_cleaned_happiness_data.xlsx')

# Define Target and Features
X = data.drop(columns=['Happiness Score'])  # Features
y = data['Happiness Score']                 # Target

# Convert Categorical Variables to Numeric Using One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Split Data into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection for Reduced Feature Set
selector = SelectKBest(score_func=f_regression, k=5)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)

# Function to Evaluate Models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Fit the model
    model.fit(X_train, y_train)
    # Predict on test set
    y_pred = model.predict(X_test)
    # Calculate R-squared score
    score = r2_score(y_test, y_pred)
    print(f"R-squared score for {model_name}: {score}")
    # Cross-validation
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
    print(f"Cross-validation score for {model_name}: {cv_score}\n")
    return model

# Try Different Models
# Model 1: Random Forest with Full Feature Set
model_rf_full = RandomForestRegressor(random_state=42)
evaluate_model(model_rf_full, X_train, X_test, y_train, y_test, "Random Forest (Full Feature Set)")

# Model 2: Random Forest with Reduced Feature Set
model_rf_reduced = RandomForestRegressor(random_state=42)
evaluate_model(model_rf_reduced, X_train_reduced, X_test_reduced, y_train, y_test, "Random Forest (Reduced Feature Set)")

# Model 3: Gradient Boosting with Full Feature Set
model_gb_full = GradientBoostingRegressor(random_state=42)
evaluate_model(model_gb_full, X_train, X_test, y_train, y_test, "Gradient Boosting (Full Feature Set)")

# Model 4: Gradient Boosting with Reduced Feature Set
model_gb_reduced = GradientBoostingRegressor(random_state=42)
evaluate_model(model_gb_reduced, X_train_reduced, X_test_reduced, y_train, y_test, "Gradient Boosting (Reduced Feature Set)")

# Model 5: Support Vector Regressor with Full Feature Set
model_svr_full = SVR()
evaluate_model(model_svr_full, X_train, X_test, y_train, y_test, "Support Vector Regressor (Full Feature Set)")

# Save the best model
best_model = model_gb_full  # Replace this with the model that gives the best performance
with open('best_model.model', 'wb') as file:
    pickle.dump(best_model, file)
