from flask import Flask, render_template
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__, template_folder='src')

# Define the relative file path to your dataset
file_path = os.path.join('dataset', 'studentsperformance.csv')

@app.route('/')
def index():
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop the 'StudentID' column if it exists
    if 'StudentID' in df.columns:
        df = df.drop(columns=['StudentID'])

    # Check for missing values in each column
    missing_values = df.isnull().sum()

    # Fill missing values with specific strategies
    df_cleaned_mean = df.fillna(df.mean())  # Fill with mean
    df_cleaned_median = df.fillna(df.median())  # Fill with median
    df_cleaned_zero = df.fillna(0)  # Fill with 0

    # Check again for missing values to ensure none remain
    cleaned_missing_values = df_cleaned_zero.isnull().sum()

    # Get a summary of descriptive statistics
    summary_stats = df.describe()

    # Calculate Mean, Median, Mode, Std Dev, Variance, Min, Max, Range, Percentiles
    mean_values = df.mean()
    median_values = df.median()
    mode_values = df.mode().iloc[0]
    std_dev = df.std()
    variance = df.var()
    min_values = df.min()
    max_values = df.max()
    range_values = max_values - min_values
    percentiles = df.quantile([0.25, 0.5, 0.75])

    # Plot histograms and save as images
    histogram_path = 'static/histograms.png'
    df.hist(bins=20, figsize=(15, 10))
    plt.suptitle('Histograms of Numerical Variables')
    plt.savefig(histogram_path)
    plt.close()

    # Plot box plots and save as images
    boxplot_path = 'static/boxplots.png'
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df)
    plt.title('Box Plots of Numerical Variables')
    plt.savefig(boxplot_path)
    plt.close()

    # Generate and plot the correlation matrix heatmap and save as images
    corr_matrix = df.corr()
    heatmap_path = 'static/correlation_matrix.png'
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.savefig(heatmap_path)
    plt.close()

    # Prepare data to send to the template
    stats_data = {
        'missing_values': missing_values.to_dict(),
        'cleaned_missing_values': cleaned_missing_values.to_dict(),
        'mean': mean_values.to_dict(),
        'median': median_values.to_dict(),
        'mode': mode_values.to_dict(),
        'std_dev': std_dev.to_dict(),
        'variance': variance.to_dict(),
        'min': min_values.to_dict(),
        'max': max_values.to_dict(),
        'range': range_values.to_dict(),
        'percentiles': percentiles.to_dict(),
        'summary_stats': summary_stats.to_dict(),
        'histogram_path': histogram_path,
        'boxplot_path': boxplot_path,
        'heatmap_path': heatmap_path,
    }

    return render_template('index.html', stats=stats_data)

if __name__ == '__main__':
    app.run(debug=True)
