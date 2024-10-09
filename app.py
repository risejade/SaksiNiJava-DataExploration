import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import plotly.express as px
import altair as alt

st.set_page_config(layout="centered")

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

st.subheader("Exploratory Data Analysis (EDA)")

# Countplot of each categorical variable

with st.expander("Countplot - categorical variable"):
    # Countplot of each categorical variable
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

with st.expander("Pie Chart - Age and Gender"):
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

with st.expander("Count Plot - Categorical Variable"):
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


with st.expander("Line Chart - Average GPA by Absences and Age"):
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

        # GPA Distribution by Demographic Features Section (box plot)
with st.expander("Box plot - GPA Distribution by Demographics"):
    st.subheader("GPA Distribution by Demographics")
    selected_feature = st.selectbox("Select a demographic feature to analyze GPA distribution", ['Age', 'Gender', 'Ethnicity'])
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=selected_feature, y='GPA', data=df)
    st.pyplot(plt.gcf())
    st.markdown(f"The boxplot shows how GPA varies based on **{selected_feature}**. This helps to see the distribution of GPA among different groups.")

# Study Time vs GPA Scatter Plot Section
with st.expander("Scatter plot - Study time vs GPA"):
    st.subheader("Study Time vs GPA")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='StudyTimeWeekly', y='GPA', hue='GradeClass', data=df, palette='tab10')
    plt.title("Study Time vs. GPA")
    st.pyplot(plt.gcf())
    st.markdown("This scatter plot shows the relationship between weekly study time and GPA. Points are colored based on students' grade classifications.")

# Bar Chart of Grade Class Distribution Section
with st.expander("Bar Graph - Grade Class Distribution"):
    st.subheader("Grade Class Distribution")
    grade_class_group = df.groupby('GradeClass').size().reset_index(name='Counts')
    grade_class_group['GradeClass'] = grade_class_group['GradeClass'].replace({
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'
    })
    st.bar_chart(data=grade_class_group.set_index('GradeClass')['Counts'])
    st.markdown("The bar chart above shows the distribution of students across different grade classes.")

# GPA by Parental Support Section
with st.expander("Box Plot - GPA by Parental Support"):
    st.subheader("GPA by Parental Support")
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='ParentalSupport', y='GPA', data=df, palette='muted')
    st.pyplot(plt.gcf())
    st.markdown("This boxplot illustrates how **parental support** levels correlate with GPA.")

# Correlation Heatmap Section
with st.expander("Correlation Matrix Heatmap"):
    corr = df.corr()
    plt.figure(figsize=(10, 6))
    
    # Create the heatmap with formatted annotations
    sns.heatmap(
        corr, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        fmt=".2f",  # Format the numbers to two decimal places
        annot_kws={"size": 10},  # Adjust the font size of the annotations
        linewidths=0.5,  # Add lines between cells for better separation
        linecolor='gray'  # Color of the lines
    )
    
    plt.title("Correlation Matrix Heatmap", fontsize=14)  # Title for the heatmap
    st.pyplot(plt.gcf())
    st.markdown("The heatmap above shows correlations between numeric variables such as study time, absences, parental support, and GPA.")

variable_options = [
    "Age",
    "Gender",
    "Ethnicity",
    "Parental Education",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "Parental Support",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GPA",
    "GradeClass"
]

with st.expander("Histogram"):
    selected_variable = st.selectbox("Select variable to display histogram:", variable_options)

    # Define a function to plot the histogram based on the selected variable
    def plot_histogram(variable):
        plt.figure(figsize=(10, 6))
    
        if variable == "Age":
            sns.histplot(df['Age'], bins=10, kde=True)
            plt.title("Age Distribution of Students")
        elif variable == "Gender":
            sns.histplot(df['Gender'], bins=2, kde=False)
            plt.title("Gender Distribution of Students")
        elif variable == "Ethnicity":
            sns.histplot(df['Ethnicity'], bins=4, kde=False)
            plt.title("Ethnicity Distribution of Students")
        elif variable == "Parental Education":
            sns.histplot(df['ParentalSupport'], bins=5, kde=False)
            plt.title("Parental Education Distribution")
        elif variable == "StudyTimeWeekly":
            sns.histplot(df['StudyTimeWeekly'], bins=10, kde=True)
            plt.title("Weekly Study Time Distribution")
        elif variable == "Absences":
            sns.histplot(df['Absences'], bins=10, kde=True)
            plt.title("Absences Distribution")
        elif variable == "Tutoring":
            sns.histplot(df['Tutoring'], bins=2, kde=False)
            plt.title("Tutoring Distribution")
        elif variable == "Parental Support":
            sns.histplot(df['ParentalSupport'], bins=5, kde=False)
            plt.title("Parental Support Distribution")
        elif variable == "Extracurricular":
            sns.histplot(df['Extracurricular'], bins=2, kde=False)
            plt.title("Extracurricular Activity Distribution")
        elif variable == "Sports":
            sns.histplot(df['Sports'], bins=2, kde=False)
            plt.title("Sports Participation Distribution")
        elif variable == "Music":
            sns.histplot(df['Music'], bins=2, kde=False)
            plt.title("Music Participation Distribution")
        elif variable == "Volunteering":
            sns.histplot(df['Volunteering'], bins=2, kde=False)
            plt.title("Volunteering Participation Distribution")
        elif variable == "GPA":
            sns.histplot(df['GPA'], bins=10, kde=True)
            plt.title("GPA Distribution")
        elif variable == "GradeClass":
            sns.histplot(df['GradeClass'], bins=5, kde=False)
            plt.title("Grade Class Distribution")

        plt.xlabel(variable)
        plt.ylabel("Frequency")
        st.pyplot(plt)

    # Call the function to plot the histogram for the selected variable
    plot_histogram(selected_variable)

    st.markdown("---")

# Boxplots 
def create_boxplots(data):
    variables = [
        'Age', 
        'Gender', 
        'Ethnicity', 
        'Parental Education', 
        'StudyTimeWeekly', 
        'Absences', 
        'Tutoring', 
        'Parental Support', 
        'Extracurricular', 
        'Sports', 
        'Music', 
        'Volunteering', 
        'GPA', 
        'GradeClass'
    ]
    
with st.expander("Box Plot"):
    # Add a selectbox for choosing the variable to visualize
    st.subheader("Boxplots of Student Variables")
    selected_variable = st.selectbox("Choose a variable to display its boxplot:", 
                                    options=[
                                        'Age', 
                                        'Gender', 
                                        'Ethnicity', 
                                        'Parental Education', 
                                        'StudyTimeWeekly', 
                                        'Absences', 
                                        'Tutoring', 
                                        'Parental Support', 
                                        'Extracurricular', 
                                        'Sports', 
                                        'Music', 
                                        'Volunteering', 
                                        'GPA', 
                                        'GradeClass'
                                    ])

    # Function to create boxplot
    def create_boxplot(data, variable):
        plt.figure(figsize=(10, 6))
        if variable in ['Gender', 'Ethnicity', 'Tutoring', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GradeClass']:
            sns.boxplot(x=variable, y='GPA', data=data)
            plt.title(f'Boxplot of GPA by {variable}')
            plt.xlabel(variable)
            plt.ylabel('GPA')
        else:
            sns.boxplot(y=variable, data=data)
            plt.title(f'Boxplot of {variable}')
            plt.ylabel(variable)
        
        plt.grid(True)
        st.pyplot(plt)

    # Create and display the boxplot for the selected variable
    create_boxplot(df, selected_variable)


st.subheader("Conclusion")
st.image("conclu.jpg", use_column_width=True)


conclusion_text = """
    This report highlights factors that influence student academic performance. Most students are 16 
    years old and are evenly split between genders with diverse ethnicities. Parental education levels vary but 
    fewer have higher degrees. It also shows that students tend to study around 5 to 15 hours per week, but 
    most did not have tutoring. Absenteeism is also very common with an average of 15 days (about 2 weeks) 
    absent. In terms of extracurricular activities, there aren’t a lot of students involved. Based on the GPAs, 
    students perform moderately, but the majority are in advanced classes suggesting that many are in advanced 
    classes despite their average grades. 

    The data shows strong correlations between certain factors and academic performance. There’s a 
    negative correlation of absences and GPA suggesting that students tend to perform poorly when there are 
    more absences. On the other hand, positive correlation exists between GPA and study, and between GPA 
    and parental support, which suggests that students who study longer and receive support from their parents 
    will tend to perform well academically. However, demographics and being in an extracurricular activity shows 
    little to no correlation with GPA. 

    In conclusion, attendance, study habits, and family background are the most influential factors in 
    determining academic performance. The students who regularly attended school, spent more hours on 
    studying, and came from supportive parents were likely to have better grades. Meanwhile, extracurricular 
    activities, gender, and race did not seem to have influence over GPA. To enhance academic outcomes, 
    students could focus on attending classes more and making schools a better place by encouraging better 
    study habits and more involvement of parents.
    """


st.markdown(conclusion_text)