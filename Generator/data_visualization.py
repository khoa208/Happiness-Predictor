import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('another_happiness.csv')

# Example visualization: Happiness Score distribution
def plot_happiness_distribution():
    plt.figure(figsize=(10, 6))
    plt.hist(data['Happiness Score'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Happiness Scores')
    plt.xlabel('Happiness Score')
    plt.ylabel('Frequency')
    plt.show()

# Example: Scatter plot of GDP vs Happiness Score
def plot_gdp_vs_happiness():
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Economy (GDP per Capita)'], data['Happiness Score'], color='green')
    plt.title('GDP vs Happiness Score')
    plt.xlabel('Economy (GDP per Capita)')
    plt.ylabel('Happiness Score')
    plt.show()

# Call the visualization functions
if __name__ == "__main__":
    plot_happiness_distribution()
    plot_gdp_vs_happiness()
