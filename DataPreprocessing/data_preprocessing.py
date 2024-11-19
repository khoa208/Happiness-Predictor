from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
happiness_data = pd.read_excel('happiness_data.xlsx', engine='openpyxl')

# Copy the dataset for processing
data = happiness_data.copy()

# --- Data Exploration and Visualization ---
# Select only numerical columns for the correlation matrix
num_data = data.select_dtypes(include=[np.number])

# Plotting correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = num_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# --- Experimenting with Attribute Combinations ---
data['GDP per Population'] = data['GDP'] / data['Population']
data['Family + Health'] = data['Family'] + data['Health (Life Expectancy)']

# --- Handling Missing Values ---
num_imputer = SimpleImputer(strategy='median')
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[num_cols] = num_imputer.fit_transform(data[num_cols])

# Identify categorical columns, excluding 'Country' and 'Region'
cat_cols = data.select_dtypes(include=['object']).columns.difference(['Country', 'Region'])

# Check if there are any categorical columns to impute
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
else:
    print("No categorical columns to impute.")

# --- Handling Outliers ---
z_scores = np.abs((data[num_cols] - data[num_cols].mean()) / data[num_cols].std())
for col in num_cols:
    outliers = z_scores[col] > 3
    data.loc[outliers, col] = data[col].median()

# --- Handling Text and Categorical Attributes ---
# Apply one-hot encoding to categorical attributes excluding "Country" and "Region"
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_cat_cols = one_hot_encoder.fit_transform(data[cat_cols])
encoded_cat_df = pd.DataFrame(encoded_cat_cols, columns=one_hot_encoder.get_feature_names_out(cat_cols))

# Remove the original categorical columns except for "Country" and "Region"
data = data.drop(columns=cat_cols)

# Concatenate one-hot encoded columns with the original data, keeping "Country" and "Region" only once
data = pd.concat([data, encoded_cat_df], axis=1)

# --- Feature Scaling ---
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Save the final cleaned data without duplicate "Country" and "Region" columns
data.to_excel('final_cleaned_happiness_data.xlsx', index=False)
