import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the relative file path to your dataset
file_path = os.path.join('dataset', 'studentsperformance.csv')

# Load the dataset
@st.cache  # Cache the data loading for performance
def load_data():
    df = pd.read_csv(file_path)

    # Drop the 'StudentID' column if it exists
    if 'StudentID' in df.columns:
        df = df.drop(columns=['StudentID'])
    return df

df = load_data()

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

# Streamlit UI
st.title('Students Performance Analysis')

# Display missing values
st.subheader('Missing Values')
st.write(missing_values)

# Display cleaned missing values
st.subheader('Cleaned Missing Values')
st.write(cleaned_missing_values)

# Display summary statistics
st.subheader('Summary Statistics')
st.write(summary_stats)

# Display mean, median, mode, std dev, variance, min, max, range, percentiles
st.subheader('Descriptive Statistics')
st.write('Mean:', mean_values)
st.write('Median:', median_values)
st.write('Mode:', mode_values)
st.write('Standard Deviation:', std_dev)
st.write('Variance:', variance)
st.write('Minimum Values:', min_values)
st.write('Maximum Values:', max_values)
st.write('Range:', range_values)
st.write('Percentiles:', percentiles)

# Plot histograms
st.subheader('Histograms of Numerical Variables')
fig, ax = plt.subplots(figsize=(15, 10))
df.hist(bins=20, ax=ax)
plt.suptitle('Histograms of Numerical Variables')
st.pyplot(fig)

# Plot box plots
st.subheader('Box Plots of Numerical Variables')
fig, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(data=df, ax=ax)
plt.title('Box Plots of Numerical Variables')
st.pyplot(fig)

# Generate and plot the correlation matrix heatmap
st.subheader('Correlation Matrix Heatmap')
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
plt.title('Correlation Matrix Heatmap')
st.pyplot(fig)
