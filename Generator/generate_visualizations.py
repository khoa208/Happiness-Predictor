import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example feature importance values
feature_importance = {
    "Economic & Family Score": 0.35,
    "Well-Being": 0.30,
    "Happiness Rank": 0.20,
    "Standard Error": 0.10,
    "Dystopia Residual": 0.05
}

# Simulated correlation matrix
correlation_matrix = np.array([
    [1.00, 0.80, -0.50, 0.70, 0.60],
    [0.80, 1.00, -0.40, 0.65, 0.55],
    [-0.50, -0.40, 1.00, -0.30, -0.20],
    [0.70, 0.65, -0.30, 1.00, 0.75],
    [0.60, 0.55, -0.20, 0.75, 1.00]
])

features = list(feature_importance.keys())

# Bar chart for feature importance
plt.figure(figsize=(8, 6))
plt.bar(feature_importance.keys(), feature_importance.values(), color='skyblue')
plt.title("Feature Importance in Prediction Model")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("feature_importance_chart.png")
plt.show()

# Heatmap for feature correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, xticklabels=features, yticklabels=features, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# Simulated predicted vs actual data
actual_scores = np.random.uniform(3, 8, 20)  # Example actual scores
predicted_scores = actual_scores + np.random.normal(0, 0.3, 20)  # Predicted scores with slight noise

# Scatter plot for predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(actual_scores, predicted_scores, color='green', alpha=0.7, label="Predicted")
plt.plot([3, 8], [3, 8], color='red', linestyle='--', label="Perfect Prediction")
plt.title("Predicted vs Actual Happiness Scores")
plt.xlabel("Actual Happiness Score")
plt.ylabel("Predicted Happiness Score")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual_chart.png")
plt.show()
