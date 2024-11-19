import matplotlib.pyplot as plt
import pandas as pd

# Simulating dataset features for visualizations
data = {
    "Feature": [
        "Economy (GDP per Capita)", "Family", "Health (Life Expectancy)", 
        "Freedom", "Trust (Government Corruption)", "Generosity", "Dystopia Residual"
    ],
    "Contribution": [30, 25, 20, 10, 5, 5, 5]  # Example contribution percentages
}

# Converting to DataFrame
df = pd.DataFrame(data)

# Pie Chart: Contribution of Features to Happiness Score
plt.figure(figsize=(8, 8))
plt.pie(
    df["Contribution"], 
    labels=df["Feature"], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Paired.colors
)
plt.title("Feature Contributions to Happiness Score")
plt.tight_layout()

# Save and show the chart
plt.savefig("feature_contribution_pie_chart.png")
plt.show()
