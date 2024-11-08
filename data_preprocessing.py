from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (adjust the file path as needed)
happiness_data = pd.read_excel('happiness_data.xlsx', engine='openpyxl')

# Copying the dataset for processing
data = happiness_data.copy()

# --- Data Exploration and Visualization ---
# Select only numerical columns for the correlation matrix
num_data = data.select_dtypes(include=[np.number])  # Includes only numerical columns

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
# Imputing missing values with median for numerical features
num_imputer = SimpleImputer(strategy='median')
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[num_cols] = num_imputer.fit_transform(data[num_cols])

# Imputing missing values with most frequent for categorical features
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# --- Handling Outliers ---
# Using Z-score to identify outliers (values > 3 standard deviations from the mean)
z_scores = np.abs((data[num_cols] - data[num_cols].mean()) / data[num_cols].std())

# Replacing outliers with median values
for col in num_cols:
    outliers = z_scores[col] > 3
    data.loc[outliers, col] = data[col].median()

# --- Handling Text and Categorical Attributes ---
# Applying one-hot encoding to categorical attributes
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_cat_cols = one_hot_encoder.fit_transform(data[cat_cols])
encoded_cat_df = pd.DataFrame(encoded_cat_cols, columns=one_hot_encoder.get_feature_names_out(cat_cols))
# Concatenate the one-hot encoded columns with the original data (excluding original categorical columns)
data_with_features = pd.concat([data.drop(cat_cols, axis=1), encoded_cat_df], axis=1)

# --- Feature Scaling ---
scaler = StandardScaler()
data_with_features[num_cols] = scaler.fit_transform(data_with_features[num_cols])

# --- Saving the data with original features retained ---
data_with_features.to_csv('final_cleaned_happiness_data_with_features.csv', index=False)

# --- Dimensionality Reduction (PCA) on the data with original features retained ---
pca = PCA(n_components=0.95)  # Retain 95% of variance
data_pca = pca.fit_transform(data_with_features)

# Convert the PCA-transformed data to a DataFrame for further use
data_pca_df = pd.DataFrame(data_pca)

# --- Saving the PCA-transformed data ---
data_pca_df.to_csv('final_cleaned_happiness_data_pca.csv', index=False)

# Verify saved files: 
# - "final_cleaned_happiness_data_with_features.csv" for original features
# - "final_cleaned_happiness_data_pca.csv" for PCA-transformed features
