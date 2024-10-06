import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

# Define the Google Drive file ID and the output file name
file_id = '1rjhNT-Q9ENl-WAoJFD8vrI0d-bMwCyL7'
output_file = 'studentsperformance.csv'

# Download the dataset from Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

# Load the dataset
@st.cache_data  # Cache the data loading for performance
def load_data():
    df = pd.read_csv(output_file)

    # Drop the 'StudentID' column if it exists
    if 'StudentID' in df.columns:
        df = df.drop(columns=['StudentID'])
    return df

df = load_data()

import streamlit as st

# Title Section
st.title("Student Performance Dataset: An Exploration of Key Factors")
st.image("dataset-cover.jpg", use_column_width=True)
# Introduction Section
st.subheader("Introduction")
st.markdown("""
This dataset provides an in-depth look at various factors affecting the academic performance of students. 
The data spans across multiple demographic and behavioral aspects such as **age**, **gender**, **ethnicity**, **parental involvement**, and **extracurricular activities**, 
all of which are analyzed in relation to **students' Grade Point Average (GPA)** and **overall performance**.
""")

# Interactive Navigation
section = st.selectbox(
    "Choose a section to explore:",
    ["Overview", "Dataset Source", "Purpose of Exploration", "Data Fields Overview"]
)

# Display content based on the selected section
if section == "Overview":
    st.markdown("### Overview")
    st.markdown("""
                
    The dataset originally contained **15 fields**, but we dropped the **StudentID** field as it serves as a unique identifier for students but doesn't contribute to the analysis of their academic performance. 

    We focus on the remaining **14 fields**, which are divided into the following categories:
    - **Demographic Details**: Age, Gender, Ethnicity
    - **Study Habits**: Weekly study time, absences, tutoring
    - **Parental Involvement**: Levels of parental support
    - **Extracurricular Activities**: Participation in sports, music, and volunteering
    - **Academic Performance**: GPA and grade classification

    Our goal is to analyze how these features influence student performance and determine what factors correlate with better or worse academic outcomes.
    """)

elif section == "Dataset Source":
    st.markdown("### Dataset Source")
    st.markdown("""
    This dataset is sourced from Kaggle and created by Rabie ElKharoua. It aims to explore the impact of various factors—such as demographics, study habits, and 
    parental involvement—on student academic performance. The dataset is valuable for educational research and predictive modeling, offering insights that can 
    inform strategies to enhance student success and targeted support.
                
    **Dataset Link**: [Student Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset )
    """)

elif section == "Purpose of Exploration":
    st.markdown("### Purpose of Exploration")
    st.markdown("""
    The primary aim of this exploration is to analyze the factors influencing student academic performance based on the dataset. Specifically, we seek to answer the following questions:

    - What is the age distribution of students, and how does it relate to their performance?
    - How do gender and ethnicity distributions appear in the context of academic achievement?
    - What levels of parental education exist, and how do they affect student outcomes?
    - How many hours do students typically study weekly, and what is the relationship between study time and GPA?
    - What are the patterns of absenteeism, and how does this correlate with academic performance?
    - How prevalent is tutoring among students, and does it impact their grades?
    - What is the level of parental support, and how does it influence academic success?
    - How do extracurricular activities (sports, music, volunteering) correlate with academic performance?
    - What insights can we draw from the relationships between GPA and other variables in the dataset?
    By addressing these questions, we aim to uncover actionable insights into the determinants of academic performance, guiding strategies for improvement in educational settings.
    """)

elif section == "Data Fields Overview":
    st.markdown("### Data Fields Overview")
    st.markdown("""
    This dataset now consists of **14 key fields**, grouped into five main categories, which are described below.

    #### Demographic Details:
    - **Age**: Students' age ranges from 15 to 18 years.
    - **Gender**: 0 for Male, 1 for Female.
    - **Ethnicity**: 
        - 0: Caucasian  
        - 1: African American  
        - 2: Asian  
        - 3: Other

    #### Study Habits:
    - **StudyTimeWeekly**: Weekly study time in hours, ranging from 0 to 20.
    - **Absences**: Number of absences during the school year, ranging from 0 to 30.
    - **Tutoring**: 0 indicates No, 1 indicates Yes for receiving tutoring.

    #### Parental Involvement:
    - **ParentalSupport**: 
        - 0: None  
        - 1: Low  
        - 2: Moderate  
        - 3: High  
        - 4: Very High

    #### Extracurricular Activities:
    - **Extracurricular**: 0 indicates No, 1 indicates Yes for participating in extracurricular activities.
    - **Sports**: 0 indicates No, 1 indicates Yes for participating in sports.
    - **Music**: 0 indicates No, 1 indicates Yes for participating in music activities.
    - **Volunteering**: 0 indicates No, 1 indicates Yes for participating in volunteering activities.

    #### Academic Performance:
    - **GPA**: Grade Point Average on a scale from 2.0 to 4.0.
    - **GradeClass**: The target variable, which classifies students into letter grades based on GPA:
        - 0: 'A' (GPA >= 3.5)  
        - 1: 'B' (3.0 <= GPA < 3.5)  
        - 2: 'C' (2.5 <= GPA < 3.0)  
        - 3: 'D' (2.0 <= GPA < 2.5)  
        - 4: 'F' (GPA < 2.0)
    """)

# Horizontal Divider
st.markdown("---")




# Check for missing values in each column
# missing_values = df.isnull().sum()

# # Fill missing values with specific strategies
# df_cleaned_mean = df.fillna(df.mean())  # Fill with mean
# df_cleaned_median = df.fillna(df.median())  # Fill with median
# df_cleaned_zero = df.fillna(0)  # Fill with 0

# # Check again for missing values to ensure none remain
# cleaned_missing_values = df_cleaned_zero.isnull().sum()

# # Get a summary of descriptive statistics
# summary_stats = df.describe()

# # Calculate Mean, Median, Mode, Std Dev, Variance, Min, Max, Range, Percentiles
# mean_values = df.mean()
# median_values = df.median()
# mode_values = df.mode().iloc[0]
# std_dev = df.std()
# variance = df.var()
# min_values = df.min()
# max_values = df.max()
# range_values = max_values - min_values
# percentiles = df.quantile([0.25, 0.5, 0.75])

# # Streamlit UI
# st.title('Students Performance Analysis')

# # Display missing values
# st.subheader('Missing Values')
# st.write(missing_values)

# # Display cleaned missing values
# st.subheader('Cleaned Missing Values')
# st.write(cleaned_missing_values)

# # Display summary statistics
# st.subheader('Summary Statistics')
# st.write(summary_stats)

# # Display mean, median, mode, std dev, variance, min, max, range, percentiles
# st.subheader('Descriptive Statistics')
# st.write('Mean:', mean_values)
# st.write('Median:', median_values)
# st.write('Mode:', mode_values)
# st.write('Standard Deviation:', std_dev)
# st.write('Variance:', variance)
# st.write('Minimum Values:', min_values)
# st.write('Maximum Values:', max_values)
# st.write('Range:', range_values)
# st.write('Percentiles:', percentiles)

# # Plot histograms
# st.subheader('Histograms of Numerical Variables')
# fig, ax = plt.subplots(figsize=(15, 10))
# df.hist(bins=20, ax=ax)
# plt.suptitle('Histograms of Numerical Variables')
# st.pyplot(fig)

# # Plot box plots
# st.subheader('Box Plots of Numerical Variables')
# fig, ax = plt.subplots(figsize=(15, 10))
# sns.boxplot(data=df, ax=ax)
# plt.title('Box Plots of Numerical Variables')
# st.pyplot(fig)

# # Generate and plot the correlation matrix heatmap
# st.subheader('Correlation Matrix Heatmap')
# corr_matrix = df.corr()
# fig, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
# plt.title('Correlation Matrix Heatmap')
# st.pyplot(fig)