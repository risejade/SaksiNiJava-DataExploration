import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import plotly.express as px
import altair as alt

st.title("Student Performance Dataset: An Exploration of Key Factors 📈🎓")

st.markdown("---")

st.image("dataset-cover.jpg", use_column_width=True)

st.subheader("Introduction")
st.markdown("""
This dataset provides an in-depth look at various factors affecting the academic performance of students. 
The data spans across multiple demographic and behavioral aspects such as **age**, **gender**, **ethnicity**, **parental involvement**, and **extracurricular activities**, 
all of which are analyzed in relation to **students' Grade Point Average (GPA)** and **overall performance**.
""")

with st.expander("📜 Overview"):
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

with st.expander("🌐 Dataset Source"):
    st.markdown("""
    This dataset is sourced from Kaggle and created by Rabie ElKharoua. It aims to explore the impact of various factors—such as demographics, study habits, and 
    parental involvement—on student academic performance. The dataset is valuable for educational research and predictive modeling, offering insights that can 
    inform strategies to enhance student success and targeted support.

    **Dataset Link**: [Student Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset )
    """)

with st.expander("🖋️ Purpose of Exploration"):
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

with st.expander("👁️ Data Fields Overview"):
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

with st.expander("🔑 Key Statistics"):
    
    # Add the descriptive statistics text
    st.markdown("""
        The dataset contains 2,392 records and 15 fields; it includes meter- and time-related measurements
        and categorical data. The age of the students is between the ages of fifteen and eighteen years, with a mean
        age of about sixteen years. Regarding the gender of the students, 51% are males (coded as 0) and 49% are females (coded as 1).
        There are four ethnic classifications, most of them being whites. Parental education levels are also shown to be distinct, 
        where most learners are from homes with parents who have completed high school. Weekly study time averages around 9.77 hours, 
        with students studying between 0 to 20 hours, having an average of about 14.54 hours. 
        The average number of absences is 54, with a maximum of 29. About 30% of students have tutors, and parental involvement varies 
        from none to extremely high, with an average mark. Extracurricular activity rates are roughly at 38%, while the average GPA 
        is approximately 2.98, ranging from 0.11 to 4.00. The target variable is Grade Class, which has a distribution of grades, 
        with most students receiving ‘C’ and ‘B’ grades. These important statistics help the user understand and analyze student performance, 
        as well as various aspects that may be affecting it.
    """)

    # Define the data for statistics
    data_stats = {
        "Metric": [
            "Total Records", 
            "Mean Age", 
            "Gender (Male)", 
            "Gender (Female)", 
            "Most Common Ethnicity", 
            "Parental Education Level", 
            "Average Study Hours", 
            "Average Absences", 
            "Students with Tutors", 
            "Average GPA", 
            "Grade Class Distribution"
        ],
        "Value": [
            2392, 
            16.47, 
            "51%", 
            "49%", 
            "Whites", 
            "High School", 
            "9.77", 
            "14.54", 
            "30%", 
            "2.98", 
            "Mostly C and B grades"
        ]
    }

    # Create a DataFrame
    df_stats = pd.DataFrame(data_stats)

    # Display statistics table
    st.subheader("Key Descriptive Statistics")
    st.table(df_stats)

    # Optional: Visualize key statistics
    st.subheader("Key Statistics Visualization")
    st.bar_chart(df_stats.set_index('Metric')['Value'].apply(pd.to_numeric, errors='coerce').dropna())

st.markdown("---")

file_id = '1rjhNT-Q9ENl-WAoJFD8vrI0d-bMwCyL7'
output_file = 'studentsperformance.csv'

with st.spinner('Loading data...'):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

@st.cache_data  
def load_data():
    df = pd.read_csv(output_file)

    if 'StudentID' in df.columns:
        df = df.drop(columns=['StudentID'])
    return df

with st.spinner('Loading data into the application...'):
    df = load_data()
    
st.subheader("Descriptive Statistics")

st.markdown("""
        The age statistics of the students in the dataset reveal that the average age is approximately 16.47 years, 
        indicating that most students are around this age. The median age stands at 16.00 years, 
        suggesting that half of the students are younger than 16 and half are older, reinforcing the centrality of age within this dataset. 
        Additionally, the mode age is 15 years, highlighting that this age group is particularly common among the students. 
        The standard deviation of 1.12 years indicates a relatively small spread in ages, with most students falling within one to two years of the average age. 
        These statistics provide valuable insights into the age distribution of the student population, which can be essential for further analysis of academic 
        performance and related factors.
        """)


mean_age = df['Age'].mean()
median_age = df['Age'].median()
mode_age = df['Age'].mode()[0] 
std_dev_age = df['Age'].std()


stats_data = {
    "Statistic": ["Mean", "Median", "Mode", "Standard Deviation"],
    "Age": [mean_age, median_age, mode_age, std_dev_age]
}

stats_df = pd.DataFrame(stats_data)


st.table(stats_df)

st.markdown("---")


st.subheader("Student Performance Dataset: Age and Gender Pie Charts")

chart_type = st.selectbox('Select a chart to display:', ['Age', 'Gender'])

if chart_type == 'Age':
    if not df['Age'].isnull().any():
        age_counts = df['Age'].value_counts().reset_index()
        age_counts.columns = ['Age', 'Count']
        
        fig_age = px.pie(age_counts, names='Age', values='Count', title='Age Distribution', 
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_age)
    else:
        st.write("No valid age data available.")

elif chart_type == 'Gender':
    df['Gender'] = df['Gender'].replace({0: 'Male', 1: 'Female'})
    if not df['Gender'].isnull().any():
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig_gender = px.pie(gender_counts, names='Gender', values='Count', title='Gender Distribution', 
                            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_gender)
    else:
        st.write("No valid gender data available.")

st.markdown("---")

st.subheader("Exploratory Data Analysis (EDA)")

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

    st.markdown("---")