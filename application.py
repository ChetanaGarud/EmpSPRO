import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Load the trained model
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# Model performance metrics
r2_value = 0.9645  # From retraining output
mae_value = 150000.00  # Placeholder: Replace with actual MAE from retrain_model.py

# Define options for select boxes
gender_options = ['Male', 'Female']
job_role_options = [
    'IT Auditor', 'GRC Specialist', 'Incident Responder', 'Data Warehouse Engineer', 'Game Developer',
    'Data Center Technician', 'Sales Engineer', 'Database Developer', 'Systems Administrator',
    'Business Intelligence Analyst', 'Full-Stack Developer', 'Cloud DevOps Engineer', 'AI Research Scientist',
    'Big Data Engineer', 'Data Scientist', 'Network Engineer', 'Embedded Systems Engineer', 'Network Administrator',
    'Site Reliability Engineer', 'DevOps Engineer', 'Software Developer/Engineer', 'Solutions Architect',
    'IT Support Technician', 'Ethical Hacker', 'Frontend Developer', 'Security Architect', 'Technical Lead',
    'Blockchain Developer', 'IT Trainer', 'Cybersecurity Analyst', 'Machine Learning Engineer', 'UI/UX Designer',
    'IT Consultant', 'Data Analyst', 'IT Director', 'Security Engineer', 'Product Designer', 'IT Project Manager',
    'CTO', 'QA Engineer', 'Web Designer', 'Technical Writer', 'Backend Developer', 'IoT Engineer',
    'Database Administrator', 'Mobile App Developer', 'Cloud Security Specialist', 'AR/VR Developer',
    'Cloud Architect', 'Data Engineer', 'Cloud Engineer'
]
state_options = ['Maharashtra']
district_options = [
    'Mumbai City', 'Chandrapur', 'Washim', 'Wardha', 'Gondia', 'Parbhani', 'Akola', 'Amravati', 'Sangli',
    'Gadchiroli', 'Palghar', 'Beed', 'Jalna', 'Hingoli', 'Osmanabad (Dharashiv)', 'Aurangabad (Chh. Sambhaji)',
    'Latur', 'Thane', 'Ratnagiri', 'Yavatmal', 'Dhule', 'Ahmednagar', 'Nanded', 'Nagpur', 'Kolhapur',
    'Jalgaon', 'Buldhana', 'Sindhudurg', 'Raigad', 'Nashik', 'Pune', 'Bhandara', 'Satara', 'Solapur',
    'Mumbai Suburban', 'Nandurbar', 'Shahapur'
]
company_type_options = ['MNC', 'Small Company', 'Mid-Sized Company', 'Startup']
education_level_options = ['Diploma', "Bachelor's", "Master's", 'PhD']
premium_institute_options = ['Yes', 'No']
certification_options = [
    'None', 'CCSP', 'Oracle Certified Professional', 'AWS Solutions Architect', 'CCNA', 'CISSP',
    'AWS Certified Developer', 'AWS Certified Data Analytics', 'Microsoft Azure Developer Associate', 'PRINCE2', 'OSCP'
]

# Function to save feedback to CSV
def save_feedback(feedback_data):
    feedback_file = 'feedback.csv'
    feedback_df = pd.DataFrame([feedback_data])
    if os.path.exists(feedback_file):
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        feedback_df.to_csv(feedback_file, index=False)

# Streamlit app configuration
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="wide")

# Custom CSS for dark theme, animations, and moving background
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1e1e1e 0%, #2c3e50 100%);
        padding: 30px;
        color: #e0e0e0;
        position: relative;
        overflow: hidden;
        font-family: 'Roboto', sans-serif;
    }
    /* Moving background wave animation */
    .main::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, rgba(76, 175, 80, 0.1), rgba(33, 150, 243, 0.1));
        animation: wave 15s infinite linear;
        z-index: -1;
        opacity: 0.3;
    }
    @keyframes wave {
        0% { transform: translateY(0); }
        50% { transform: translateY(-50px); }
        100% { transform: translateY(0); }
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>select {
        background-color: #2c2c2c;
        color: #e0e0e0;
        border: 1px solid #555;
        border-radius: 10px;
        padding: 12px;
        font-size: 14px;
        transition: border-color 0.3s;
    }
    .stTextInput>div>input:focus, .stNumberInput>div>input:focus, .stSelectbox>div>select:focus {
        border-color: #4CAF50;
    }
    .prediction-box {
        background-color: #2c3e50;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .header {
        color: #ffffff;
        text-align: center;
        margin-bottom: 30px;
        font-size: 3em;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .subheader {
        color: #cccccc;
        font-size: 1.8em;
        margin-top: 20px;
        font-weight: 600;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888888;
        font-size: 0.9em;
    }
    .sidebar .stMarkdown, .sidebar .stText {
        color: #e0e0e0;
    }
    .stSlider .stText, .stTextArea textarea {
        color: #e0e0e0 !important;
        background-color: #2c2c2c !important;
        border: 1px solid #555 !important;
        border-radius: 10px !important;
    }
    .stSuccess {
        background-color: #1a3c1a;
        color: #ffffff;
        border-radius: 8px;
    }
    .stPlotlyChart {
        margin-top: 20px;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar with model performance and information
st.sidebar.title("About Employee Salary Predictor")
st.sidebar.markdown("""
    This application leverages a GradientBoosting model to predict annual salaries based on your professional profile.
    **Model Performance**:
    - **R² Score**: {:.4f} (explains {:.2f}% of salary variance)
    
    **Note**: Provide accurate details for optimal predictions.
""".format(r2_value, r2_value * 100, mae_value), unsafe_allow_html=True)

# Header image (base64 encoded to avoid external dependencies)
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
    </div>
""", unsafe_allow_html=True)

# App title and description
st.title("💼 Employee Salary Predictor")
st.write("Enter your professional details below .")

# Input form
with st.form("prediction_form"):
    st.markdown('<div class="subheader">Your Professional Profile</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Enter your age (18-100)")
        gender = st.selectbox("Gender", gender_options, help="Select your gender")
        years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, help="Enter your years of professional experience")
        job_role = st.selectbox("Job Role", job_role_options, help="Select your job role")
        state = st.selectbox("State", state_options, help="Select your state (currently Maharashtra only)")
        
    with col2:
        district = st.selectbox("District", district_options, help="Select your district in Maharashtra")
        company_type = st.selectbox("Company Type", company_type_options, help="Select the type of company you work for")
        education_level = st.selectbox("Education Level", education_level_options, help="Select your highest education level")
        premium_institute = st.selectbox("Premium Institute", premium_institute_options, help="Did you attend a premium institute? (e.g., IIT, IIM)")
        certification = st.selectbox("Certification", certification_options, help="Select your certification, if any")
    
    submitted = st.form_submit_button("Predict Annual Salary", use_container_width=True)

# Process prediction and visualizations
if submitted:
    # Prepare input data with original columns
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'years_of_experience': [years_of_experience],
        'job_role': [job_role],
        'district': [district],
        'company_type': [company_type],
        'education_level': [education_level],
        'premium_institute': [premium_institute],
        'certification': [certification]
    })
    
    try:
        # Make prediction using the pipeline
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Annual Salary: ₹{prediction:,.2f}</h3>
                <p>This is your predicted annual salary based on the provided details.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown('<div class="subheader">Your Salary Insights</div>', unsafe_allow_html=True)
        st.write("Explore how your predicted salary compares and what drives it.")
        
        # 1. Bar Chart: Predicted Salary vs. Average for Job Role
        avg_salary = 1000000.0  # Placeholder: Replace with actual average salary for job_role
        fig_bar = px.bar(
            x=['Your Predicted Salary', f'Average {job_role} Salary'],
            y=[prediction, avg_salary],
            labels={'x': '', 'y': 'Salary (₹)'},
            title=f'Your Salary vs. Average {job_role} Salary',
            color=['Your Predicted Salary', f'Average {job_role} Salary'],
            color_discrete_map={'Your Predicted Salary': '#4CAF50', f'Average {job_role} Salary': '#2196F3'}
        )
        fig_bar.update_layout(
            plot_bgcolor='#2c2c2c',
            paper_bgcolor='#2c2c2c',
            font_color='#e0e0e0',
            title_font_size=20,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 2. Gauge Chart: Salary Percentile
        percentile = min(max((prediction - (avg_salary * 0.8)) / (avg_salary * 1.2 - avg_salary * 0.8) * 100, 0), 100)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentile,
            title={'text': "Salary Percentile (Estimated)", 'font': {'size': 20, 'color': '#e0e0e0'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#e0e0e0', 'tickfont': {'color': '#e0e0e0'}},
                'bar': {'color': '#4CAF50'},
                'bgcolor': '#2c2c2c',
                'bordercolor': '#555',
                'steps': [
                    {'range': [0, 33], 'color': '#ff6b6b'},
                    {'range': [33, 66], 'color': '#ffd93d'},
                    {'range': [66, 100], 'color': '#4CAF50'}
                ]
            }
        ))
        fig_gauge.update_layout(
            plot_bgcolor='#2c2c2c',
            paper_bgcolor='#2c2c2c',
            font_color='#e0e0e0',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 3. Feature Importance Plot
        feature_importance = {
            'Age': 0.1,
            'Years of Experience': 0.3,
            'Job Role': 0.25,
            'District': 0.1,
            'Company Type': 0.15,
            'Education Level': 0.05,
            'Premium Institute': 0.03,
            'Certification': 0.02
        }  # Placeholder: Replace with actual feature importances
        fig_importance = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            labels={'x': 'Importance', 'y': 'Feature'},
            title='Factors Influencing Your Salary',
            orientation='h',
            color=list(feature_importance.values()),
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(
            plot_bgcolor='#2c2c2c',
            paper_bgcolor='#2c2c2c',
            font_color='#e0e0e0',
            title_font_size=20,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Feedback form
        st.markdown('<div class="subheader">Provide Feedback</div>', unsafe_allow_html=True)
        st.write("Your feedback helps us improve prediction accuracy!")
        with st.form("feedback_form"):
            feedback_rating = st.slider("How accurate was the prediction? (1-5)", 1, 5, 3, help="1 = Not accurate, 5 = Very accurate")
            feedback_comments = st.text_area("Additional Comments", help="Share any additional thoughts or suggestions")
            feedback_submitted = st.form_submit_button("Submit Feedback", use_container_width=True)
            
            if feedback_submitted:
                feedback_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'age': age,
                    'gender': gender,
                    'years_of_experience': years_of_experience,
                    'job_role': job_role,
                    'state': state,
                    'district': district,
                    'company_type': company_type,
                    'education_level': education_level,
                    'premium_institute': premium_institute,
                    'certification': certification,
                    'predicted_salary': prediction,
                    'feedback_rating': feedback_rating,
                    'feedback_comments': feedback_comments
                }
                save_feedback(feedback_data)
                st.success("Thank you for your feedback! It will help us improve our predictions.")
                
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Footer
st.markdown("""
    <div class="footer">
        <p>Developed  using Streamlit | Powered by GradientBoosting Model</p>
        <p>© 2025 Employee Salary Predictor|</p>
    </div>
""", unsafe_allow_html=True)
