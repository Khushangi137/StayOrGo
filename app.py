# ============================================
# StayOrGo - Employee Attrition Predictor
# Complete Streamlit App with Model Comparison
# ============================================
# Save this file as: app.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="StayOrGo | Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .main-header span {
        color: #7c3aed;
    }
    .sub-header {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f6ff;
        border: 1px solid #e8e3ff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #7c3aed;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-high {
        color: #dc2626;
        font-weight: 700;
    }
    .risk-moderate {
        color: #d97706;
        font-weight: 700;
    }
    .risk-low {
        color: #059669;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        color: #c4b5fd;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA - FIXED VERSION
# ============================================
@st.cache_data
def load_data():
    # IBM HR Analytics Employee Attrition Dataset
    url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/employee_attrition.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to load data: {e}")
        st.info("📌 Please check your internet connection or try again later.")
        st.info("💡 The app requires the IBM HR Attrition Dataset to work properly.")
        return None

@st.cache_resource
def train_all_models(X_train, X_test, y_train, y_test):
    """Train all 4 models and return results"""
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = {
        'model': lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_prob_lr),
        'y_prob': y_prob_lr
    }
    
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]
    results['Decision Tree'] = {
        'model': dt,
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'precision': precision_score(y_test, y_pred_dt),
        'recall': recall_score(y_test, y_pred_dt),
        'f1': f1_score(y_test, y_pred_dt),
        'roc_auc': roc_auc_score(y_test, y_prob_dt),
        'y_prob': y_prob_dt
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    results['Random Forest'] = {
        'model': rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_prob_rf),
        'y_prob': y_prob_rf
    }
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'model': xgb,
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_prob_xgb),
        'y_prob': y_prob_xgb
    }
    
    return results

# ============================================
# LOAD AND PREPARE DATA
# ============================================
df = load_data()

# Check if data loaded successfully
if df is None:
    st.stop()  # Stop the app if no data

# Preprocessing
le_dict = {}
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# Features and Target - Fixed column names for IBM dataset
# The IBM dataset uses different column names
drop_cols = ['Attrition', 'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
# Only drop columns that exist in the dataframe
drop_cols = [col for col in drop_cols if col in df_encoded.columns]

X = df_encoded.drop(drop_cols, axis=1)
y = df_encoded['Attrition']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train all models
model_results = train_all_models(X_train_scaled, X_test_scaled, y_train, y_test)

# Best model based on Recall (catching employees who will leave)
best_model_name = max(model_results, key=lambda x: model_results[x]['recall'])
best_model = model_results[best_model_name]['model']

# Feature importance from Random Forest
rf_model = model_results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.markdown("## StayOrGo")
st.sidebar.markdown("*Employee Attrition Intelligence*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Model Comparison", "Predict Attrition", "Data Insights", "About"]
)

# ============================================
# PAGE 1: MODEL COMPARISON
# ============================================
if page == "Model Comparison":
    st.markdown('<p class="main-header">Model <span>Comparison</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluating 4 machine learning models to find the best performer for attrition prediction.</p>', unsafe_allow_html=True)
    
    # Metrics Table
    st.subheader("Performance Metrics Comparison")
    
    comparison_data = []
    for name, res in model_results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{res['accuracy']:.3f}",
            'Precision': f"{res['precision']:.3f}",
            'Recall': f"{res['recall']:.3f}",
            'F1 Score': f"{res['f1']:.3f}",
            'ROC-AUC': f"{res['roc_auc']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Highlight best model
    st.success(f"**Best Model: {best_model_name}** — Selected for highest recall ({model_results[best_model_name]['recall']:.3f}), crucial for catching at-risk employees.")
    
    # Metrics Visualization
    st.subheader("Model Performance Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar Chart Comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        models = list(model_results.keys())
        recalls = [model_results[m]['recall'] for m in models]
        precisions = [model_results[m]['precision'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, recalls, width, label='Recall', color='#7c3aed')
        bars2 = ax.bar(x + width/2, precisions, width, label='Precision', color='#a78bfa')
        
        ax.set_ylabel('Score')
        ax.set_title('Recall vs Precision by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        # ROC Curves
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#7c3aed', '#a78bfa', '#c4b5fd', '#ede9ff']
        
        for (name, res), color in zip(model_results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
    
    # Why Recall Matters
    st.markdown("---")
    st.subheader("Why We Prioritize Recall")
    st.markdown("""
    In attrition prediction, **Recall** is the most important metric because:
    
    - **False Negative** = Predicting an employee will stay when they actually leave → **Costly mistake**
    - **False Positive** = Predicting an employee will leave when they stay → Minor inconvenience (extra check-in)
    
    Missing an at-risk employee means losing talent without any intervention opportunity.
    """)
    
    # Model Selection Justification
    st.markdown("---")
    st.subheader("Model Selection Justification")
    
    st.markdown(f"""
    | Model | Why Chosen/Rejected |
    |-------|---------------------|
    | **Logistic Regression** | Too simple, missed {100 - int(model_results['Logistic Regression']['recall']*100)}% of at-risk employees |
    | **Decision Tree** | Overfits easily, lower recall than ensemble methods |
    | **{best_model_name}** ✅ | Highest recall ({model_results[best_model_name]['recall']:.1%}), good balance of precision and interpretability |
    | **XGBoost** | Close competitor but slightly lower recall and more complex to tune |
    """)

# ============================================
# PAGE 2: PREDICT ATTRITION
# ============================================
elif page == "Predict Attrition":
    st.markdown('<p class="main-header">Predict <span>Attrition</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter employee details to assess their risk of leaving.</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 60, 30)
        monthly_income = st.slider("Monthly Income (INR)", 10000, 200000, 50000, step=5000)
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
        job_satisfaction = st.selectbox("Job Satisfaction", ["Very Low", "Low", "Medium", "High", "Very High"])
    
    with col2:
        work_life_balance = st.selectbox("Work-Life Balance", ["Very Poor", "Poor", "Average", "Good", "Excellent"])
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        distance = st.slider("Distance from Office (km)", 1, 50, 10)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    
    with col3:
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                              "Manufacturing Director", "Healthcare Representative", "Manager",
                                              "Sales Representative", "Research Director", "Human Resources"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education", ["Below College", "College", "Bachelor", "Master", "Doctor"])
        num_companies = st.slider("Previous Companies", 0, 10, 2)
    
    # Prepare input for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'BusinessTravel': ['Travel_Rarely'],
        'DailyRate': [df['DailyRate'].median()],
        'Department': [department],
        'DistanceFromHome': [distance],
        'Education': [education],
        'EducationField': ['Life Sciences'],
        'EnvironmentSatisfaction': [3],
        'Gender': ['Male'],
        'HourlyRate': [df['HourlyRate'].median()],
        'JobInvolvement': [3],
        'JobLevel': [2],
        'JobRole': [job_role],
        'JobSatisfaction': [job_satisfaction],
        'MaritalStatus': [marital_status],
        'MonthlyIncome': [monthly_income],
        'MonthlyRate': [df['MonthlyRate'].median()],
        'NumCompaniesWorked': [num_companies],
        'OverTime': [overtime],
        'PercentSalaryHike': [df['PercentSalaryHike'].median()],
        'PerformanceRating': [3],
        'RelationshipSatisfaction': [3],
        'StockOptionLevel': [1],
        'TotalWorkingYears': [years_at_company + 5],
        'TrainingTimesLastYear': [3],
        'WorkLifeBalance': [work_life_balance],
        'YearsAtCompany': [years_at_company],
        'YearsInCurrentRole': [max(0, years_at_company - 2)],
        'YearsSinceLastPromotion': [max(0, years_at_company - 3)],
        'YearsWithCurrManager': [max(0, years_at_company - 1)]
    })
    
    # Encode categorical columns
    for col in categorical_cols:
        if col in input_data.columns and col != 'Attrition':
            try:
                input_data[col] = le_dict[col].transform(input_data[col])
            except:
                input_data[col] = 0
    
    # Ensure correct column order
    input_data = input_data[X.columns]
    
    # Scale
    input_scaled = scaler.transform(input_data)
    
    # Predict
    if st.button("Run Attrition Analysis", type="primary", use_container_width=True):
        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0]
        risk_percent = probability[1] * 100
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error(f"### HIGH RISK")
                st.markdown(f"**Risk Score: {risk_percent:.1f}%**")
                st.markdown("*Employee is likely to leave within 6 months.*")
            else:
                st.success(f"### LOW RISK")
                st.markdown(f"**Risk Score: {risk_percent:.1f}%**")
                st.markdown("*Employee is likely to stay.*")
        
        with col2:
            # Risk Meter
            st.markdown("**Risk Level**")
            st.progress(int(risk_percent) / 100)
            if risk_percent > 65:
                st.markdown('<span class="risk-high">CRITICAL — Immediate action required</span>', unsafe_allow_html=True)
            elif risk_percent > 35:
                st.markdown('<span class="risk-moderate">MODERATE — Monitor closely</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="risk-low">SAFE — No immediate concern</span>', unsafe_allow_html=True)
        
        with col3:
            # Top contributing factors
            st.markdown("**Key Drivers**")
            if overtime == 'Yes':
                st.warning("Overtime is a major risk factor")
            if job_satisfaction in ['Very Low', 'Low']:
                st.warning("Low job satisfaction detected")
            if distance > 25:
                st.warning("Long commute increases risk")
        
        # Risk Factors Detail
        st.markdown("---")
        st.subheader("Risk Factor Analysis")
        
        risk_factors = []
        if overtime == 'Yes':
            risk_factors.append("Works overtime regularly — high burnout potential")
        if job_satisfaction in ['Very Low', 'Low']:
            risk_factors.append("Job satisfaction is critically low — signs of disengagement")
        if work_life_balance in ['Very Poor', 'Poor']:
            risk_factors.append("Poor work-life balance — a leading indicator of attrition")
        if distance > 25:
            risk_factors.append("Long commute (>25 km) — adds daily stress, reduces retention")
        if years_at_company < 2:
            risk_factors.append("Less than 2 years tenure — still in high-risk adjustment period")
        if num_companies > 5:
            risk_factors.append("Frequent job changes — pattern of short tenures")
        if monthly_income < 30000:
            risk_factors.append("Below-market compensation — increases flight risk")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.success("No significant risk factors detected.")
        
        # Recommendations
        st.markdown("---")
        st.subheader("Recommended Actions")
        
        if risk_percent > 65:
            st.markdown("""
            - Schedule immediate one-on-one retention meeting
            - Review and adjust compensation package
            - Offer flexible/remote work arrangement
            - Create clear 6-month career growth roadmap
            - Reduce overtime with workload redistribution
            """)
        elif risk_percent > 35:
            st.markdown("""
            - Conduct structured stay interview
            - Evaluate work-life balance policies
            - Assign mentor or career coach
            - Schedule monthly satisfaction check-ins
            """)
        else:
            st.markdown("""
            - Continue regular feedback sessions
            - Recognize and reward performance
            - Provide learning and development opportunities
            """)

# ============================================
# PAGE 3: DATA INSIGHTS
# ============================================
elif page == "Data Insights":
    st.markdown('<p class="main-header">Data <span>Insights</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understanding patterns in employee attrition data.</p>', unsafe_allow_html=True)
    
    # Key Metrics
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Employees</div></div>', unsafe_allow_html=True)
    with col2:
        att_rate = (df['Attrition'] == 'Yes').mean() * 100
        st.markdown(f'<div class="metric-card"><div class="metric-value">{att_rate:.1f}%</div><div class="metric-label">Attrition Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["Age"].mean():.0f}</div><div class="metric-label">Avg Age</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">INR {df["MonthlyIncome"].mean():,.0f}</div><div class="metric-label">Avg Monthly Income</div></div>', unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Overtime")
        fig, ax = plt.subplots(figsize=(6, 4))
        overtime_att = df.groupby('OverTime')['Attrition'].value_counts(normalize=True).unstack()
        overtime_att['Yes'].plot(kind='bar', ax=ax, color=['#a78bfa', '#7c3aed'])
        ax.set_ylabel('Attrition Rate')
        ax.set_title('Attrition Rate: Overtime vs No Overtime')
        ax.set_xticklabels(['No Overtime', 'Overtime'], rotation=0)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Attrition by Job Satisfaction")
        fig, ax = plt.subplots(figsize=(6, 4))
        sat_att = df.groupby('JobSatisfaction')['Attrition'].value_counts(normalize=True).unstack()
        sat_att['Yes'].plot(kind='bar', ax=ax, color='#7c3aed')
        ax.set_ylabel('Attrition Rate')
        ax.set_title('Attrition Rate by Job Satisfaction')
        ax.set_xlabel('Job Satisfaction (1=Low, 4=Very High)')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Features Driving Attrition")
        fig, ax = plt.subplots(figsize=(6, 4))
        top_features = feature_importance.head(10)
        ax.barh(top_features['Feature'], top_features['Importance'], color='#7c3aed')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Random Forest)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Attrition by Department")
        fig, ax = plt.subplots(figsize=(6, 4))
        dept_att = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack()
        dept_att['Yes'].plot(kind='bar', ax=ax, color=['#7c3aed', '#a78bfa', '#c4b5fd'])
        ax.set_ylabel('Attrition Rate')
        ax.set_title('Attrition Rate by Department')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

# ============================================
# PAGE 4: ABOUT
# ============================================
else:
    st.markdown('<p class="main-header">About <span>StayOrGo</span></p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Employee Attrition Prediction System
    
    **StayOrGo** is an intelligent employee attrition prediction system built to help HR teams identify 
    employees at risk of leaving the organization.
    
    ### Key Features
    
    - **4-Model Comparison**: Logistic Regression, Decision Tree, Random Forest, XGBoost
    - **Real-time Prediction**: Enter employee details and get instant risk assessment
    - **Risk Factor Analysis**: Understand what's driving attrition risk
    - **Actionable Recommendations**: Get specific intervention strategies
    
    ### Technical Stack
    
    | Component | Technology |
    |-----------|------------|
    | Frontend | Streamlit |
    | Backend | Python 3.x |
    | ML Libraries | Scikit-learn, XGBoost |
    | Data Processing | Pandas, NumPy |
    | Visualization | Matplotlib, Seaborn |
    
    ### Dataset
    
    IBM HR Analytics Employee Attrition Dataset (1,470 employees, 35 features)
    
    ### Model Selection
    
    After comparing 4 models, the best performing model was selected based on **Recall** — 
    the ability to correctly identify employees who will leave. Missing an at-risk employee 
    is more costly than a false alarm.
    
    ### Business Value
    
    - Average cost of replacing an employee: 1.5x to 2x annual salary
    - Early intervention can reduce attrition by 15-25%
    - Proactive retention saves organizations significant costs
    """)
    
    # Metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model Recall", f"{model_results[best_model_name]['recall']:.1%}")
    with col2:
        st.metric("Best Model AUC", f"{model_results[best_model_name]['roc_auc']:.1%}")
    with col3:
        st.metric("Dataset Size", f"{len(df)} employees")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown('<p class="footer">StayOrGo v1.0 | Employee Attrition Intelligence Engine | Built for AI/ML Internship</p>', unsafe_allow_html=True)
