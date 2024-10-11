import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import plotly.express as px
import altair as alt



st.set_page_config(
    page_title="Student Performance Dataset : Data Exploration", 
    page_icon="🎓", 
    layout="centered" 
)



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
    - How many hours do students typically study weekly, and what is the relationship between study time and GPA?
    - How prevalent is tutoring among students, and does it impact their grades?
    - What is the level of parental support, and how does it influence academic success?
    - How does participation in extracurricular activities affect GPA and performance, considering age?

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
    
st.subheader("📋 Descriptive Statistics")

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

with st.expander("Pie Chart - Age and Gender", expanded=True):
    st.subheader("Age and Gender Pie Charts")

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


variable_options = [
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


# Assuming df and variable_options are already defined
with st.expander("Histogram", expanded=True):
    selected_variable = st.selectbox("Select variable to display histogram:", variable_options)

    # Define a function to plot the histogram based on the selected variable
    def display_histogram(variable):
        if variable in ["Age", "StudyTimeWeekly", "Absences", "GPA"]:
            # For continuous variables
            fig = px.histogram(df, x=variable, nbins=10, title=f"{variable} Distribution")
            st.plotly_chart(fig)
        elif variable in ["Gender", "Ethnicity", "Parental Education", "Tutoring", 
                          "Parental Support", "Extracurricular", "Sports", 
                          "Music", "Volunteering", "GradeClass"]:
            # For categorical variables
            fig = px.histogram(df, x=variable, title=f"{variable} Distribution", histnorm='percent')
            st.plotly_chart(fig)

    # Call the function to plot the histogram for the selected variable
    display_histogram(selected_variable)

# with st.expander("Count Plot - Categorical Variable", expanded=True):
#     df.drop(['GPA'], axis=1, inplace=True)

#     numerical_columns = [col for col in df.columns if df[col].nunique() > 5]

#     categorical_columns = df.columns.difference(numerical_columns).difference(['GradeClass']).to_list()

#     custom_labels = {
#         'Ethnicity': ['Caucasian', 'African American', 'Asian', 'Other'],
#         'Age': [15, 16, 17, 18],
#         'ParentalEducation': ['None', 'High School', 'Some College', 'Bachelor\'s', 'Higher'],
#         'Tutoring': ['No', 'Yes'],
#         'ParentalSupport': ['No', 'Low', 'Moderate', 'High', 'Very High'],
#         'Extracurricular': ['No', 'Yes'],
#         'Sports': ['No', 'Yes'],
#         'Music': ['No', 'Yes'],
#         'Volunteering': ['No', 'Yes'],
#         'Gender': ['Male', 'Female']
#     }

#     selected_variable = st.selectbox('Select a categorical variable to display a countplot', categorical_columns)

#     st.subheader(f'Countplot of {selected_variable}')

#     counts = df[selected_variable].value_counts().sort_index()

#     counts_df = pd.DataFrame(counts).reset_index()
#     counts_df.columns = [selected_variable, 'Count']

#     counts_df[selected_variable] = counts_df[selected_variable].replace(dict(enumerate(custom_labels[selected_variable])))

#     colors = sns.color_palette("Set2", n_colors=len(counts_df))

#     st.bar_chart(data=counts_df.set_index(selected_variable)['Count'], use_container_width=True)


with st.expander("What is the age distribution of students, and how does it relate to their performance?", expanded=True):
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
        
with st.expander("How do gender and ethnicity distributions appear in the context of academic achievement?", expanded=True):
    
    st.subheader("👨 Gender and Ethnicity Distributions in Context of Academic Achievement")
    
    st.markdown("""
    **Grade Class: Classification of students' grades based on GPA:**
    - 0: 'A' (GPA >= 3.5)
    - 1: 'B' (3.0 <= GPA < 3.5)
    - 2: 'C' (2.5 <= GPA < 3.0)
    - 3: 'D' (2.0 <= GPA < 2.5)
    - 4: 'F' (GPA < 2.0)
    """)

    # Map the numerical values of Gender and Ethnicity for clearer labels
    df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'})
    df['Ethnicity'] = df['Ethnicity'].map({
        0: 'Caucasian',
        1: 'African American',
        2: 'Asian',
        3: 'Other'
    })
    df['GradeClass'] = df['GradeClass'].map({
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'F'
    })

    # Create selectors for gender and ethnicity
    selected_gender = st.selectbox("Select Gender:", options=df['Gender'].unique())
    selected_ethnicity = st.selectbox("Select Ethnicity:", options=df['Ethnicity'].unique())

    # Filter the DataFrame based on selected gender and ethnicity
    filtered_df = df[(df['Gender'] == selected_gender) & (df['Ethnicity'] == selected_ethnicity)]

    # Grouping data by gender and ethnicity and counting grade classes
    gender_ethnicity_performance = filtered_df.groupby(['Gender', 'Ethnicity', 'GradeClass']).size().reset_index(name='Count')

    # Check if there are any records for the selected filters
    if not gender_ethnicity_performance.empty:
        # Plotting the data using Plotly
        fig = px.bar(
            gender_ethnicity_performance,
            x='GradeClass',
            y='Count',
            color='GradeClass',
            title=f'Performance of {selected_gender} Students of {selected_ethnicity} Ethnicity',
            labels={'GradeClass': 'Grade Class', 'Count': 'Number of Students'},
            category_orders={'GradeClass': ['A', 'B', 'C', 'D', 'F']}
        )

        # Update layout for better readability
        fig.update_traces(texttemplate='%{y}', textposition='inside')
        fig.update_layout(xaxis_title='Grade Class', yaxis_title='Number of Students')

        # Show the plot
        st.plotly_chart(fig)



# Study Time vs GPA Scatter Plot Section
with st.expander("How many hours do students typically study weekly, and what is the relationship between study time and GPA?", expanded=True):
    st.subheader("⏳ Study Time vs GPA")
    
    # Create a Plotly scatter plot
    fig = px.scatter(df, x='StudyTimeWeekly', y='GPA', color='GradeClass', 
                     color_continuous_scale=px.colors.sequential.Viridis,
                     labels={'StudyTimeWeekly': 'Weekly Study Time', 'GPA': 'GPA'},
                     title="Study Time vs. GPA")

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title="Weekly Study Time",
        yaxis_title="GPA",
        legend_title="Grade Class",
        title_x=0.5
    )
    
    # Center the title
    fig.update_layout(
        title=dict(
            text=f"Study Time vs. GPA",
            x=0.5,          # Center the title
            xanchor='center',  # Anchor the title at the center
            font=dict(size=20)  # Optional: Set font size for better visibility
        )
    )

    # Display the Plotly chart
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""The scatterplot titled "Study Time vs. GPA" illustrates the relationship between the amount of time students spend studying weekly (on the x-axis) 
                and their corresponding GPA (on the y-axis). Each point on the scatterplot represents an individual student's GPA based on their weekly study time. 
                The colors of the points correspond to different grade classes, as indicated in the legend: purple (0.0), red (1.0), orange (2.0), green (3.0), and blue (4.0).
                The scatterplot reveals a general trend where higher study time appears to correlate with higher GPA. 
                While there is considerable variability in GPA at all levels of study time, the clustering of colors suggests that students who dedicate more time to studying tend to 
                achieve better grades, particularly those in higher grade classes. This visualization helps in understanding the potential impact of study habits on academic performance, 
                indicating that increased study time may be beneficial for improving GPA.""")

    
with st.expander("How prevalent is tutoring among students, and does it impact their grades?", expanded=True):
    
    st.subheader("🥇 Impact on Tutoring of Students")
    
    # Map the numerical values for clarity
    df['Tutoring'] = df['Tutoring'].map({0: 'No', 1: 'Yes'})

    # Create a histogram for GPA based on tutoring status
    fig = px.histogram(
        df,
        x='GPA',
        color='Tutoring',
        barmode='overlay',  # Overlay bars to see the distribution
        title='Distribution of GPA by Tutoring Status',
        labels={'GPA': 'Grade Point Average', 'Tutoring': 'Tutoring Status'},
        opacity=0.75,
         color_discrete_map={'Yes': 'green', 'No': 'orange'},
        marginal='box'  # Adding box plot on top for summary statistics
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Grade Point Average (GPA)',
        yaxis_title='Number of Students',
        xaxis=dict(
            tickmode='linear',  # Linear tick mode for better visibility
            dtick=0.5  # Set tick intervals for GPA
        ),
        title_x=0.2  # Center the title
    )

    # Show the plot
    st.plotly_chart(fig)
    
with st.expander("What is the level of parental support, and how does it influence academic success?", expanded=True):

    st.subheader("👪 Parental Support vs Academic Success")

    # Mapping ParentalSupport numerical values for clearer labels
    df['ParentalSupport'] = df['ParentalSupport'].map({
        0: 'None',
        1: 'Low',
        2: 'Moderate',
        3: 'High',
        4: 'Very High'
    })

    # Creating a boxplot using Plotly
    fig = px.box(
        df,
        x='ParentalSupport',
        y='GPA',
        color='ParentalSupport',
        title='Boxplot: Parental Support Levels vs GPA',
        labels={'ParentalSupport': 'Parental Support Level', 'GPA': 'Grade Point Average (GPA)'},
        category_orders={"ParentalSupport": ['None', 'Low', 'Moderate', 'High', 'Very High']},
        color_discrete_map={
            'None': 'gray',
            'Low': 'red',
            'Moderate': 'orange',
            'High': 'green',
            'Very High': 'blue'
        }
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Parental Support Level",
        yaxis_title="GPA",
        boxmode='group'  # Group boxes by Parental Support Level
    )

    # Display the Plotly chart
    st.plotly_chart(fig)

    # Explanation markdown
    st.markdown("""
        This boxplot visualizes the relationship between **Parental Support** and **GPA**. Each box represents the distribution of GPA values for students with different levels of parental support. The middle line in each box represents the median GPA for that group, while the boxes capture the interquartile range (IQR), and the whiskers show the full range of data.
        
        From the plot, you can observe how higher levels of parental support might correlate with higher GPA outcomes, indicating a potential positive influence on academic success.
    """)
    
with st.expander("How does participation in extracurricular activities affect GPA and performance, considering age?", expanded=True):

    st.subheader("🏀 Extracurricular Activities, Age, and Academic Success")

    # Allow user to select one or multiple ages
    selected_ages = st.multiselect(
        "Select age(s) to filter:",
        options=df['Age'].unique(),
        default=df['Age'].unique()  # Default selects all ages
    )

    # Filter the dataframe based on the selected ages
    filtered_df = df[df['Age'].isin(selected_ages)]

    # Map the Extracurricular and GradeClass values for clearer labels
    filtered_df['Extracurricular'] = filtered_df['Extracurricular'].map({0: 'No', 1: 'Yes'})
    filtered_df['GradeClass'] = filtered_df['GradeClass'].map({
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'F'
    })

    # Creating a histogram for GPA based on extracurricular activities
    fig = px.histogram(
        filtered_df,
        x='GPA',
        color='Extracurricular',
        nbins=20,  # Number of bins
        title='Histogram: GPA Distribution by Extracurricular Participation (Filtered by Age)',
        labels={'Extracurricular': 'Participation in Extracurricular Activities', 'GPA': 'Grade Point Average (GPA)'},
        color_discrete_map={'Yes': 'green', 'No': 'orange'},
        barmode='overlay'  # Overlay bars for better comparison
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="GPA",
        yaxis_title="Number of Students",
        bargap=0.1,  # Small gap between bars
        barmode='overlay'  # Bars for each category will overlap
    )

    # Display the GPA histogram chart
    st.plotly_chart(fig)


with st.expander("Correlation Matrix Heatmap", expanded=True):
    
    st.subheader("Correlation Matrix Heatmap")

    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr = numeric_df.corr()

    # Create a Plotly heatmap with a valid colorscale
    fig = px.imshow(
        corr,
        text_auto=".2f",  # Format numbers to 2 decimal places
        color_continuous_scale='Viridis',  # You can change this to other valid colorscales
        aspect="auto",
        labels=dict(x="Variables", y="Variables", color="Correlation Coefficient"),
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Variables",
    )
    
    # Display the Plotly chart
    st.plotly_chart(fig, use_container_width=True)

    # Markdown explanation
    st.markdown("""The heatmap you provided is a correlation matrix, 
                which visually represents the relationships between 
                different variables in a dataset related to predicting 
                student grades (GPA). Each cell in the matrix contains 
                a correlation coefficient, ranging from -1 to 1, which 
                indicates the strength and direction of the relationship 
                between two variables. A correlation of 1 (shown in yellow) 
                indicates a perfect positive relationship, meaning as one 
                variable increases, the other also increases. A correlation 
                of -1 (shown in dark purple) signifies a perfect negative 
                relationship, meaning that as one variable increases, the 
                other decreases. A correlation near 0 (greenish-blue) 
                implies no significant relationship between the two variables.""")

    # Adding a bullet list of key findings
    st.write("Key Findings from the Correlation Matrix:")
    findings = [
        "📌 There is a strong negative correlation (-0.92) between Absences and GPA, indicating that more absences are associated with a significantly lower GPA.",
        "📌 There is a positive but weak correlation (0.18) between StudyTimeWeekly and GPA, suggesting that more study time is somewhat linked to a higher GPA.",
        "📌 A small positive correlation (0.09) exists between Sports and Extracurricular activities, indicating that students involved in sports are slightly more likely to participate in other extracurriculars.",
        "📌 Several variables, including Age, Parental Education, Music, and Volunteering, show near-zero correlations with GPA, suggesting minimal impact on academic performance."
    ]
    st.write("- " + "\n- ".join(findings))

# Other code continues here...

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

