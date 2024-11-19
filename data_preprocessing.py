import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
data = pd.read_csv('another_happiness.csv')

# Data Preprocessing
def preprocess_data():
    # Drop unnecessary columns
    data_cleaned = data.drop(columns=['Country', 'Region'])
    
    # Handle missing values (if any)
    data_cleaned = data_cleaned.dropna()

    # Split features and target
    X = data_cleaned.drop(columns=['Happiness Score'])
    y = data_cleaned['Happiness Score']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for reuse in Flask app
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Build and Train the Model
def build_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model for reuse in Flask app
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

# Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

# Main Function
if __name__ == "__main__":
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Data Preprocessing Completed.")

    # Build and train the model
    model = build_model(X_train, y_train)
    print("Model Training Completed.")

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
