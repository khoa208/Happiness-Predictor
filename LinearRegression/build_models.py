import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

# Load the preprocessed data
data = pd.read_excel('processed_happiness_data.xlsx')

# Feature Engineering Enhancements
if 'GDP' in data.columns and 'Population' in data.columns:
    data['GDP_per_capita'] = data['GDP'] / (data['Population'] + 1)
    data['Log_GDP'] = np.log1p(data['GDP'])
    data['Log_Population'] = np.log1p(data['Population'])
else:
    print("GDP or Population column missing; skipping GDP_per_capita and log features.")
    data['GDP_per_capita'] = 0
    data['Log_GDP'] = 0
    data['Log_Population'] = 0

if 'Family' in data.columns and 'Health (Life Expectancy)' in data.columns:
    data['Family_Health'] = data['Family'] + data['Health (Life Expectancy)']
    data['Health_per_Freedom'] = data['Health (Life Expectancy)'] * data['Freedom']
else:
    print("Family or Health (Life Expectancy) column missing; skipping Family_Health.")
    data['Family_Health'] = 0
    data['Health_per_Freedom'] = 0

if 'Freedom' in data.columns and 'Economy (GDP per Capita)' in data.columns:
    data['Freedom_to_GDP'] = data['Freedom'] / (data['Economy (GDP per Capita)'] + 1)
else:
    print("Freedom or Economy (GDP per Capita) column missing; skipping Freedom_to_GDP.")
    data['Freedom_to_GDP'] = 0

if 'Generosity' in data.columns:
    data['Generosity_Log'] = np.log1p(data['Generosity'])
else:
    print("Generosity column missing; skipping Generosity_Log.")
    data['Generosity_Log'] = 0

if 'Trust (Government Corruption)' in data.columns and 'Generosity' in data.columns:
    data['Trust_Generosity_Ratio'] = data['Trust (Government Corruption)'] / (data['Generosity'] + 1)
else:
    print("Trust or Generosity column missing; skipping Trust_Generosity_Ratio.")
    data['Trust_Generosity_Ratio'] = 0

print("New features added:")
print(data.head())

# Handle missing values
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

if len(num_cols) > 0:
    num_imputer = SimpleImputer(strategy='median')
    data[num_cols] = num_imputer.fit_transform(data[num_cols])

if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# Define Target and Features
X = data.drop(columns=['Happiness Score'], errors='ignore')
y = data['Happiness Score']

X = pd.get_dummies(X, drop_first=True)

# Save feature names
with open('full_feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns, f)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Feature Selection with Lasso
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
selected_features = X.columns[lasso.coef_ != 0]
X_train_lasso = X_train_scaled[:, lasso.coef_ != 0]
X_test_lasso = X_test_scaled[:, lasso.coef_ != 0]

with open('selected_features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save PCA components
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Models
models = {
    "Linear Regression (Full)": LinearRegression(),
    "Linear Regression (Lasso)": LinearRegression(),
    "Linear Regression (PCA)": LinearRegression(),
    "Random Forest (Full)": RandomForestRegressor(random_state=42),
    "Random Forest (Lasso)": RandomForestRegressor(random_state=42),
    "Random Forest (PCA)": RandomForestRegressor(random_state=42),
    "Gradient Boosting (Full)": GradientBoostingRegressor(random_state=42),
    "Gradient Boosting (Lasso)": GradientBoostingRegressor(random_state=42),
    "Gradient Boosting (PCA)": GradientBoostingRegressor(random_state=42),
}

# Training and Evaluation
results = {}
for name, model in models.items():
    if "Full" in name:
        model.fit(X_train_scaled, y_train)
        r2 = model.score(X_test_scaled, y_test)
    elif "Lasso" in name:
        model.fit(X_train_lasso, y_train)
        r2 = model.score(X_test_lasso, y_test)
    elif "PCA" in name:
        model.fit(X_train_pca, y_train)
        r2 = model.score(X_test_pca, y_test)
    results[name] = r2
    print(f"{name} R^2 Score: {r2}")

# Hyperparameter Tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)
print("Best Random Forest Parameters:", grid_search.best_params_)
print("Best Random Forest R^2 Score:", grid_search.best_score_)
