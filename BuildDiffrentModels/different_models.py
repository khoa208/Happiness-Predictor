# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
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

# Initialize and Evaluate Different Models
# Model 1: Linear Regression
model_lr = LinearRegression()
evaluate_model(model_lr, X_train, X_test, y_train, y_test, "Linear Regression")

# Model 2: Random Forest Regressor
model_rf = RandomForestRegressor(random_state=42)
evaluate_model(model_rf, X_train, X_test, y_train, y_test, "Random Forest Regressor")

# Model 3: Gradient Boosting Regressor
model_gb = GradientBoostingRegressor(random_state=42)
evaluate_model(model_gb, X_train, X_test, y_train, y_test, "Gradient Boosting Regressor")

# Model 4: Support Vector Regressor
model_svr = SVR()
evaluate_model(model_svr, X_train, X_test, y_train, y_test, "Support Vector Regressor")

# Model 5: K-Nearest Neighbors Regressor
model_knn = KNeighborsRegressor(n_neighbors=5)
evaluate_model(model_knn, X_train, X_test, y_train, y_test, "K-Nearest Neighbors Regressor")

# Save All Models
models = {
    "Linear Regression": model_lr,
    "Random Forest Regressor": model_rf,
    "Gradient Boosting Regressor": model_gb,
    "Support Vector Regressor": model_svr,
    "K-Nearest Neighbors Regressor": model_knn
}

# Save each model as a .model file
for model_name, model in models.items():
    filename = f"{model_name.replace(' ', '_')}.model"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
        print(f"Saved {model_name} as {filename}")
