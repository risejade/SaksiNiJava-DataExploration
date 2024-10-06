import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import altair as alt

st.title("Student Performance Dataset: An Exploration of Key Factors")
st.image("dataset-cover.jpg", use_column_width=True)

st.subheader("Introduction")
st.markdown("""
This dataset provides an in-depth look at various factors affecting the academic performance of students. 
The data spans across multiple demographic and behavioral aspects such as **age**, **gender**, **ethnicity**, **parental involvement**, and **extracurricular activities**, 
all of which are analyzed in relation to **students' Grade Point Average (GPA)** and **overall performance**.
""")

with st.expander("Overview"):
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

with st.expander("Dataset Source"):
    st.markdown("""
    This dataset is sourced from Kaggle and created by Rabie ElKharoua. It aims to explore the impact of various factors—such as demographics, study habits, and 
    parental involvement—on student academic performance. The dataset is valuable for educational research and predictive modeling, offering insights that can 
    inform strategies to enhance student success and targeted support.

    **Dataset Link**: [Student Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset )
    """)

with st.expander("Purpose of Exploration"):
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

with st.expander("Data Fields Overview"):
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

st.markdown("---")

st.subheader("Exploratory Data Analysis (EDA)")

# Countplot of each categorical variable

file_id = '1rjhNT-Q9ENl-WAoJFD8vrI0d-bMwCyL7'
output_file = 'studentsperformance.csv'

gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

@st.cache_data  
def load_data():
    df = pd.read_csv(output_file)

    if 'StudentID' in df.columns:
        df = df.drop(columns=['StudentID'])
    return df

df = load_data()


df.drop(['GPA'], axis=1, inplace=True)

numerical_columns = [col for col in df.columns if df[col].nunique() > 5]

categorical_columns = df.columns.difference(numerical_columns).difference(['GradeClass']).to_list()

custom_labels = {
    'Ethnicity': ['Caucasian', 'African American', 'Asian', 'Other'],
    'Age': [15, 16, 17, 18],
    'ParentalEducation': ['None', 'High School', 'Some College', 'Bachelor\'s', 'Higher'],
    'Tutoring': ['No', 'Yes'],
    'ParentalSupport': ['No', 'Low', 'Moderate', 'High', 'Very High'],
    'Extracurricular': ['No', 'Yes'],
    'Sports': ['No', 'Yes'],
    'Music': ['No', 'Yes'],
    'Volunteering': ['No', 'Yes'],
    'Gender': ['Male', 'Female']
}

selected_variable = st.selectbox('Select a categorical variable to display a countplot', categorical_columns)

st.subheader(f'Countplot of {selected_variable}')

counts = df[selected_variable].value_counts().sort_index()

counts_df = pd.DataFrame(counts).reset_index()
counts_df.columns = [selected_variable, 'Count']

counts_df[selected_variable] = counts_df[selected_variable].replace(dict(enumerate(custom_labels[selected_variable])))

colors = sns.color_palette("Set2", n_colors=len(counts_df))

st.bar_chart(data=counts_df.set_index(selected_variable)['Count'], use_container_width=True)



# How does the impact of absences on GPA vary by age?

file_id = '1rjhNT-Q9ENl-WAoJFD8vrI0d-bMwCyL7'
output_file = 'studentsperformance.csv'

gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

@st.cache_data  
def load_data():
    df = pd.read_csv(output_file)

    if 'StudentID' in df.columns:
        df = df.drop(columns=['StudentID'])
    return df

df = load_data()


df.Age = df.Age.astype('int64')
df['Absences'] = pd.to_numeric(df['Absences'], errors='coerce')

age_list = df.Age.unique()
age_selection = st.multiselect('Select Age(s) for analysis', age_list, [15, 16])

df_selection = df[df['Age'].isin(age_selection)]

absences_selection = st.slider('Select number of absences', 0.0, 30.0, (0.0, 30.0), step=0.1)
df_selection = df_selection[df_selection['Absences'].between(absences_selection[0], absences_selection[1])]

if df_selection.empty:
    st.write("No data available for the selected criteria.")
else:
    reshaped_df = df_selection.pivot_table(index='Absences', columns='Age', values='GPA', aggfunc='mean', fill_value=0)
    reshaped_df = reshaped_df.sort_index(ascending=True)

    df_editor = st.data_editor(
        reshaped_df, 
        height=212, 
        use_container_width=True,
        column_config={col: st.column_config.NumberColumn(col) for col in reshaped_df.columns},
        num_rows="dynamic"
    )

    df_chart = df_editor.reset_index().melt(id_vars='Absences', var_name='Age', value_name='GPA')

    gpa_chart = alt.Chart(df_chart).mark_line().encode(
        x=alt.X('Absences:Q', title='Number of Absences'),
        y=alt.Y('GPA:Q', title='Average GPA'),
        color='Age:N',
        tooltip=['Age', 'Absences', 'GPA']
    ).properties(
        title='Average GPA by Absences and Age',
        height=300
    )

    st.altair_chart(gpa_chart, use_container_width=True)

    