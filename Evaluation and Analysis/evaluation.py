import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load Preprocessed Data
data = pd.read_excel('../final_cleaned_happiness_data.xlsx')

# Define Target and Features
X = data.drop(columns=['Happiness Score'])  # Features
y = data['Happiness Score']                 # Target

# One-Hot Encoding for Categorical Variables
X = pd.get_dummies(X, drop_first=True)

# Split Data into Train/Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to Evaluate Metrics
def evaluate_metrics(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} Metrics:")
    print(f"R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    return {'Model': model_name, 'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# Load Pre-Trained Models
model_filenames = {
    "Linear Regression": "Linear_Regression.model",
    "Random Forest Regressor": "Random_Forest_Regressor.model",
    "Gradient Boosting Regressor": "Gradient_Boosting_Regressor.model",
    "Support Vector Regressor": "Support_Vector_Regressor.model",
    "K-Nearest Neighbors Regressor": "K-Nearest_Neighbors_Regressor.model"
}

results = []

for model_name, filename in model_filenames.items():
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        results.append(evaluate_metrics(y_test, y_pred, model_name))

# Save Metrics Results
results_df = pd.DataFrame(results)
results_df.to_csv('loaded_model_performance.csv', index=False)

# Visualizations
# Bar Chart for Metrics
metrics_to_plot = ['R2', 'MAE', 'RMSE']
results_melted = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 6))
sns.barplot(data=results_melted, x='Metric', y='Value', hue='Model')
plt.title("Pre-Built Model Performance Comparison")
plt.ylabel("Value")
plt.show()

# Scatter Plot for Predictions
for model_name, filename in model_filenames.items():
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        y_pred = model.predict(X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f"{model_name}: Predicted vs. Actual Happiness Scores")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.show()

# Print Summary Table
print("Summary of Model Performance:")
print(results_df)
