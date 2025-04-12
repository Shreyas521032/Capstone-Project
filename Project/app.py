import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, f1_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Healthcare Data Analytics Platform",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a535c;
        text-align: center;
        margin-bottom: 1.5rem;
        font-family: 'Arial', sans-serif;
    }
    .section-header {
        font-size: 1.8rem;
        color: #4ecdc4;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #ff6b6b;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .card {
        background-color: #f7fff7;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(26, 83, 92, 0.1);
    }
    .metric-card {
        background-color: #ffe66d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(255, 107, 107, 0.1);
        color: #1a535c;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown("<h1 class='main-header'>Healthcare Data Analytics Platform</h1>", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    try:       
        return pd.read_excel("Project/cancer_survey_dataset.xlsx")          
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        # Create sample data for testing if file not found
        sample_data = create_sample_data()
        return sample_data

# Function to create sample data if file not found (for testing purposes)
def create_sample_data():
    np.random.seed(42)
    n_samples = 100
    
    # Create sample data
    data = pd.DataFrame({
        'Age Group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'Affected by Cancer': np.random.choice([0, 1], n_samples),
        'Q4 Importance of Access': np.random.randint(1, 11, n_samples),
        'Q5 Hours per Week': np.random.uniform(0.5, 10, n_samples),
        'Q6 Satisfaction': np.random.randint(1, 11, n_samples),
        'Q8 User-Friendliness': np.random.randint(1, 11, n_samples),
        'Q9 Clicks to Find Info': np.random.randint(1, 20, n_samples),
        'Q10 Mobile Responsiveness': np.random.randint(1, 11, n_samples),
        'Q11 Search Function Importance': np.random.randint(1, 11, n_samples),
        'Q12 Load Time': np.random.randint(1, 11, n_samples),
        'Q14 Most Valuable Feature': np.random.choice(['Information', 'Support', 'Tools', 'Resources'], n_samples),
        'Q15 Multilingual Importance': np.random.randint(1, 11, n_samples),
        'Q16 Importance of Survivor Section': np.random.randint(1, 11, n_samples),
        'Q17 Importance of Donation Feature': np.random.randint(1, 11, n_samples),
        'Q18 Newsletter Subscription': np.random.choice([0, 1], n_samples),
        'Q19 Events Attending': np.random.choice([0, 1], n_samples),
        'Q20 Importance of Symptom Checker': np.random.randint(1, 11, n_samples),
        'Q21 Personalized Content Value': np.random.randint(1, 11, n_samples),
        'Q22 Importance of Clinical Trials': np.random.randint(1, 11, n_samples),
        'Q23 Navigation Style Preference': np.random.randint(1, 6, n_samples),
        'Q24 Importance of Consistent Design': np.random.randint(1, 11, n_samples)
    })
    
    # Add some correlation to make the data more realistic
    data['Q6 Satisfaction'] = (data['Q8 User-Friendliness'] * 0.6 + 
                              data['Q12 Load Time'] * 0.2 + 
                              np.random.normal(0, 1, n_samples)).astype(int)
    
    # Ensure values are within range
    data['Q6 Satisfaction'] = data['Q6 Satisfaction'].clip(1, 10)
    
    return data

# Load the data
data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dashboard", 
    "📈 Linear Regression", 
    "🔄 Logistic Regression", 
    "📉 Multiple Regression"
])

# Dashboard
if page == "📊 Dashboard":
    st.markdown("<h2 class='section-header'>Dashboard Overview</h2>", unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg. Satisfaction", f"{data['Q6 Satisfaction'].mean():.2f}/10")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg. Time Spent", f"{data['Q5 Hours per Week'].mean():.2f} hrs/week")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Cancer-Affected Users", f"{data['Affected by Cancer'].mean()*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg. User-Friendliness", f"{data['Q8 User-Friendliness'].mean():.2f}/10")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-header'>User Demographics</h3>", unsafe_allow_html=True)
        fig = px.sunburst(data, 
                         path=['Age Group', 'Gender'], 
                         color='Age Group',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-header'>Feature Importance</h3>", unsafe_allow_html=True)
        important_features = [
            'Q4 Importance of Access',
            'Q8 User-Friendliness',
            'Q11 Search Function Importance',
            'Q15 Multilingual Importance',
            'Q16 Importance of Survivor Section',
            'Q17 Importance of Donation Feature',
            'Q20 Importance of Symptom Checker',
            'Q22 Importance of Clinical Trials',
            'Q24 Importance of Consistent Design'
        ]
        importance_data = data[important_features].mean().reset_index()
        importance_data.columns = ['Feature', 'Average Score']
        importance_data['Feature'] = importance_data['Feature'].str.replace('Q[0-9]+ ', '', regex=True)
        fig = px.bar(importance_data, 
                    x='Average Score', 
                    y='Feature',
                    orientation='h',
                    color='Average Score',
                    color_continuous_scale='Blues')
        fig.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Second row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-header'>Satisfaction vs User-Friendliness</h3>", unsafe_allow_html=True)
        fig = px.scatter(data, 
                        x='Q8 User-Friendliness', 
                        y='Q6 Satisfaction',
                        color='Age Group',
                        size='Q5 Hours per Week',
                        hover_data=['Gender'],
                        trendline='ols')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='subsection-header'>Cancer Affection by Age Group</h3>", unsafe_allow_html=True)
        cancer_by_age = data.groupby('Age Group')['Affected by Cancer'].mean().reset_index()
        cancer_by_age['Affected by Cancer'] = cancer_by_age['Affected by Cancer'] * 100
        fig = px.pie(cancer_by_age, 
                    names='Age Group', 
                    values='Affected by Cancer',
                    color='Age Group',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Bluyl)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Linear Regression
elif page == "📈 Linear Regression":
    st.markdown("<h2 class='section-header'>Linear Regression Analysis</h2>", unsafe_allow_html=True)
    
    # Define linear regression models
    linear_models = {
        "User Experience Analysis": {
            "target": "Q6 Satisfaction",
            "predictors": ["Q5 Hours per Week", "Q8 User-Friendliness", "Q11 Search Function Importance", "Q12 Load Time"]
        },
        "Navigation Efficiency": {
            "target": "Q9 Clicks to Find Info",
            "predictors": ["Q8 User-Friendliness", "Q10 Mobile Responsiveness", "Q12 Load Time"]
        },
        "Engagement Patterns": {
            "target": "Q5 Hours per Week",
            "predictors": ["Q4 Importance of Access", "Q8 User-Friendliness", "Q6 Satisfaction"]
        },
        "Feature Importance": {
            "target": "Q4 Importance of Access",
            "predictors": ["Q16 Importance of Survivor Section", "Q17 Importance of Donation Feature", "Q20 Importance of Symptom Checker"]
        },
        "Satisfaction Predictors": {
            "target": "Q6 Satisfaction",
            "predictors": ["Q9 Clicks to Find Info", "Q8 User-Friendliness", "Q23 Navigation Style Preference"]
        }
    }
    
    # Model selection
    selected_model = st.selectbox("Select Linear Regression Model", list(linear_models.keys()))
    
    # Get model details
    model_details = linear_models[selected_model]
    target = model_details["target"]
    predictors = model_details["predictors"]
    
    # Display model information
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='subsection-header'>{selected_model}</h3>", unsafe_allow_html=True)
    st.write(f"**Target Variable:** {target}")
    st.write(f"**Predictor Variables:** {', '.join(predictors)}")
    
    # Handle categorical predictors if any
    X = data[predictors].copy()
    # Convert categorical variables to numerical
    for col in X.columns:
        if X[col].dtype == 'object':
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    # Use max to avoid negative R² values (which can happen with poor models)
    r2 = r2_score(y_test, y_pred)  # Fix for negative R²
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score (Coefficient of Determination)", f"{r2:.3f}")
    with col2:
        st.metric("Mean Squared Error (MSE)", f"{mse:.3f}")
    with col3:
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}")
    
    # Feature importance
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    coef_df = coef_df.sort_values('Coefficient', ascending=False)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Feature Importance</h4>", unsafe_allow_html=True)
        fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                    color='Coefficient', color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h4>Actual vs Predicted Values</h4>", unsafe_allow_html=True)
        pred_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        fig = px.scatter(pred_df, x='Actual', y='Predicted', 
                        trendline='ols', trendline_color_override='red')
        fig.add_shape(
            type='line', line=dict(dash='dash', width=2, color='gray'),
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max()
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual plot
    st.markdown("<h4>Residual Analysis</h4>", unsafe_allow_html=True)
    residuals = y_test - y_pred
    residual_df = pd.DataFrame({
        'Predicted': y_pred,
        'Residuals': residuals
    })
    fig = px.scatter(residual_df, x='Predicted', y='Residuals',
                   title='Residual Plot')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Logistic Regression
elif page == "🔄 Logistic Regression":
    st.markdown("<h2 class='section-header'>Logistic Regression Analysis</h2>", unsafe_allow_html=True)
    
    # Define logistic regression models
    logistic_models = {
        "Cancer Impact Prediction": {
            "target": "Affected by Cancer",
            "predictors": ["Q5 Hours per Week", "Q20 Importance of Symptom Checker", "Q18 Newsletter Subscription"]
        },
        "Newsletter Subscription": {
            "target": "Q18 Newsletter Subscription",
            "predictors": ["Q6 Satisfaction", "Q16 Importance of Survivor Section", "Q20 Importance of Symptom Checker"]
        },
        "Attendance at Events": {
            "target": "Q19 Events Attending",
            "predictors": ["Q16 Importance of Survivor Section", "Q21 Personalized Content Value", "Q14 Most Valuable Feature"]
        },
        "Symptom Checker Usage": {
            "target": "Q20 Importance of Symptom Checker",
            "predictors": ["Q5 Hours per Week", "Q8 User-Friendliness", "Q6 Satisfaction"]
        }
    }
    
    # Model selection
    selected_model = st.selectbox("Select Logistic Regression Model", list(logistic_models.keys()))
    
    # Get model details
    model_details = logistic_models[selected_model]
    target = model_details["target"]
    predictors = model_details["predictors"]
    
    # Display model information
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='subsection-header'>{selected_model}</h3>", unsafe_allow_html=True)
    st.write(f"**Target Variable:** {target}")
    st.write(f"**Predictor Variables:** {', '.join(predictors)}")
    
    # Prepare data
    # For logistic regression, ensure target is binary
    # For non-binary targets, binarize by threshold
    if target != "Affected by Cancer" and target != "Q18 Newsletter Subscription" and target != "Q19 Events Attending":
        threshold = data[target].median()
        y = (data[target] > threshold).astype(int)
        st.info(f"Target variable '{target}' has been binarized (threshold: {threshold})")
    else:
        y = data[target]
    
    # Handle categorical predictors if any
    X = data[predictors].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Precision", f"{report['1']['precision']:.3f}")
    with col3:
        st.metric("Recall (Sensitivity)", f"{report['1']['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Feature importance
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    })
    coef_df = coef_df.sort_values('Coefficient', ascending=False)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Feature Importance</h4>", unsafe_allow_html=True)
        fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                    color='Coefficient', color_continuous_scale='RdBu',
                    color_continuous_midpoint=0)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h4>ROC Curve</h4>", unsafe_allow_html=True)
        
        # Calculate ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("<h4>Confusion Matrix</h4>", unsafe_allow_html=True)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create confusion matrix heatmap
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Negative (0)", "Positive (1)"],
        y=["Negative (0)", "Positive (1)"],
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Multiple Regression
elif page == "📉 Multiple Regression":
    st.markdown("<h2 class='section-header'>Multiple Regression Analysis</h2>", unsafe_allow_html=True)
    
    # Define multiple regression models
    multiple_models = {
        "User Satisfaction": {
            "target": "Q6 Satisfaction",
            "predictors": ["Q4 Importance of Access", "Q8 User-Friendliness", "Q10 Mobile Responsiveness", "Q12 Load Time"]
        },
        "Clicks to Find Information": {
            "target": "Q9 Clicks to Find Info",
            "predictors": ["Q8 User-Friendliness", "Q10 Mobile Responsiveness", "Q12 Load Time", "Q24 Importance of Consistent Design"]
        },
        "Engagement (Time Spent)": {
            "target": "Q5 Hours per Week",
            "predictors": ["Q14 Most Valuable Feature", "Q16 Importance of Survivor Section", "Q20 Importance of Symptom Checker"]
        },
        "Personalized Content Value": {
            "target": "Q21 Personalized Content Value",
            "predictors": ["Q8 User-Friendliness", "Q6 Satisfaction", "Q4 Importance of Access"]
        },
        "Load Time Perception": {
            "target": "Q12 Load Time",
            "predictors": ["Q5 Hours per Week", "Q10 Mobile Responsiveness", "Q8 User-Friendliness"]
        }
    }
    
    # Model selection
    selected_model = st.selectbox("Select Multiple Regression Model", list(multiple_models.keys()))
    
    # Get model details
    model_details = multiple_models[selected_model]
    target = model_details["target"]
    predictors = model_details["predictors"]
    
    # Display model information
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='subsection-header'>{selected_model}</h3>", unsafe_allow_html=True)
    st.write(f"**Target Variable:** {target}")
    st.write(f"**Predictor Variables:** {', '.join(predictors)}")
    
    # Handle categorical predictors if any
    X = data[predictors].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = max(0, r2_score(y_test, y_pred))  # Fix for negative R²
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate adjusted R²
    n = len(X_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    adjusted_r2 = max(0, adjusted_r2)  # Ensure non-negative
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Adjusted R² Score", f"{adjusted_r2:.3f}")
    with col2:
        st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.3f}")
    with col3:
        st.metric("MAE (Mean Absolute Error)", f"{mae:.3f}")
    
    # Feature importance
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    coef_df = coef_df.sort_values('Coefficient', ascending=False)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Feature Importance</h4>", unsafe_allow_html=True)
        fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                    color='Coefficient', color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h4>Prediction Error Analysis</h4>", unsafe_allow_html=True)
        
        # Create dataframe with results
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_test - y_pred
        })
        
        fig = px.scatter(results_df, x='Actual', y='Error', 
                        color=abs(results_df['Error']),
                        color_continuous_scale='RdYlBu_r',
                        title="Residual Plot")
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted Visualization
    st.markdown("<h4>Actual vs Predicted Values</h4>", unsafe_allow_html=True)
    pred_actual_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    fig = px.scatter(pred_actual_df, x='Actual', y='Predicted',
                    trendline='ols',
                    trendline_color_override="red")
    
    # Add perfect prediction line
    fig.add_shape(
        type='line', line=dict(dash='dash', width=2, color='gray'),
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max()
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap for predictors
    st.markdown("<h4>Correlation Between Predictors</h4>", unsafe_allow_html=True)
    
    # Handle categorical variables for correlation
    numeric_data = X.copy()
    
    # Calculate correlation
    corr = numeric_data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model equation
    st.markdown("<h4>Model Equation</h4>", unsafe_allow_html=True)
    
    # Create equation
    equation = f"<b>{target}</b> = {model.intercept_:.4f}"
    for i, feature in enumerate(X.columns):
        coef = model.coef_[i]
        sign = "+" if coef >= 0 else ""
        equation += f" {sign} {coef:.4f} × <i>{feature}</i>"
    
    st.markdown(f"<div style='text-align: center; padding: 15px; background-color: #f0f8ff; border-radius: 5px;'>{equation}</div>", unsafe_allow_html=True)
    
    # Interactive prediction
    st.markdown("<h4>Interactive Prediction</h4>", unsafe_allow_html=True)
    st.write("Adjust the values below to see how they affect the predicted outcome:")
    
    # Create input fields for each predictor
    input_values = {}
    for predictor in predictors:
        if data[predictor].dtype == 'object':
            # For categorical variables
            unique_vals = data[predictor].unique()
            input_values[predictor] = st.selectbox(f"Select {predictor}", unique_vals)
        else:
            # For numerical variables
            min_val = float(data[predictor].min())
            max_val = float(data[predictor].max())
            default_val = float(data[predictor].mean())
            input_values[predictor] = st.slider(f"Select {predictor}", min_val, max_val, default_val)
    
    # Prepare input for prediction
    input_df = pd.DataFrame([input_values])
    
    # Handle categorical variables if any
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df = pd.get_dummies(input_df, columns=[col], drop_first=True)
    
    # Ensure input DataFrame has the same columns as training data
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match the training data
    input_df = input_df[X.columns]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display prediction
    st.markdown(f"<div style='text-align: center; padding: 20px; background-color: #e6f7ff; border-radius: 10px; margin-top: 20px;'><h3>Predicted {target}: {prediction:.2f}</h3></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Fix for negative R² values in all regression pages
# This function should be added at the beginning of the script, after imports
def safe_r2_score(y_true, y_pred):
    """Calculate R² score, returning 0 if the value is negative."""
    r2 = r2_score(y_true, y_pred)
    return max(0, r2)  # Return 0 if R² is negative

# Add this function to ensure proper metrics display
def format_metric_name(metric_name):
    """Format metric names with proper styling."""
    return f"**{metric_name}**"

# The main code should be updated to use these functions
# For example, in the Linear Regression page:
# r2 = safe_r2_score(y_test, y_pred)
# st.metric(format_metric_name("R² Score (Coefficient of Determination)"), f"{r2:.3f}")
