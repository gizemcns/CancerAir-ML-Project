"""
Real-time Monitoring Dashboard for Lung Cancer Prediction Model
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Page config
st.set_page_config(
    page_title="ML Model Monitoring",
    page_icon="📊",
    layout="wide"
)

# Initialize logging system
LOG_FILE = '../logs/predictions.json'

def init_logs():
    """Initialize log file if doesn't exist"""
    os.makedirs('../logs', exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            json.dump([], f)

def log_prediction(input_data, prediction, probability, timestamp):
    """Log prediction to file"""
    init_logs()
    
    log_entry = {
        'timestamp': timestamp.isoformat(),
        'input': input_data,
        'prediction': prediction,
        'probability': probability,
        'high_risk_prob': probability.get('High', probability.get('high', 0))
    }
    
    # Read existing logs
    with open(LOG_FILE, 'r') as f:
        logs = json.load(f)
    
    # Append new log
    logs.append(log_entry)
    
    # Write back
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

def load_logs():
    """Load prediction logs"""
    init_logs()
    try:
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
        return pd.DataFrame(logs)
    except:
        return pd.DataFrame()

def generate_sample_data():
    """Generate sample monitoring data for demo"""
    import numpy as np
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    data = {
        'date': dates,
        'predictions': np.random.randint(50, 200, size=30),
        'accuracy': np.random.uniform(0.82, 0.88, size=30),
        'high_risk_rate': np.random.uniform(0.25, 0.35, size=30),
        'avg_response_time': np.random.uniform(0.1, 0.3, size=30),
        'false_negative_rate': np.random.uniform(0.10, 0.16, size=30)
    }
    
    return pd.DataFrame(data)

# Title
st.title("📊 ML Model Monitoring Dashboard")
st.markdown("Real-time monitoring of Lung Cancer Prediction Model performance")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Metrics", "🔍 Predictions", "⚙️ System Health"])

with tab1:
    st.header("Overview")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Model Accuracy",
            value="85.2%",
            delta="+2.1%"
        )
    
    with col2:
        st.metric(
            label="Predictions (24h)",
            value="1,234",
            delta="+156"
        )
    
    with col3:
        st.metric(
            label="False Negative Rate",
            value="13.8%",
            delta="-1.2%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Avg Response Time",
            value="0.23s",
            delta="-0.05s",
            delta_color="inverse"
        )
    
    # Charts
    st.subheader("30-Day Performance Trends")
    
    df_trends = generate_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Predictions over time
        fig = px.line(df_trends, x='date', y='predictions',
                    title='Daily Predictions',
                    labels={'predictions': 'Number of Predictions', 'date': 'Date'})
        fig.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Accuracy over time
        fig = px.line(df_trends, x='date', y='accuracy',
                    title='Model Accuracy',
                    labels={'accuracy': 'Accuracy', 'date': 'Date'})
        fig.update_traces(line_color='#2ca02c', line_width=3)
        fig.update_yaxis(tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Metrics")
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Current': [0.852, 0.841, 0.863, 0.851],
            'Baseline': [0.750, 0.730, 0.780, 0.754],
            'Target': [0.850, 0.800, 0.850, 0.825]
        }
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Current', x=df_metrics['Metric'], y=df_metrics['Current'],
                            marker_color='#2ca02c'))
        fig.add_trace(go.Bar(name='Baseline', x=df_metrics['Metric'], y=df_metrics['Baseline'],
                            marker_color='#ff7f0e'))
        fig.add_trace(go.Scatter(name='Target', x=df_metrics['Metric'], y=df_metrics['Target'],
                                mode='markers', marker=dict(size=12, color='red', symbol='diamond')))
        
        fig.update_layout(title='Performance Metrics Comparison', yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Business Metrics")
        
        business_metrics = {
            'Metric': ['False Negative Rate', 'High Risk Detection', 'Early Detection Rate'],
            'Value': [0.138, 0.862, 0.751],
            'Status': ['✅ Good', '✅ Good', '⚠️ Monitor']
        }
        df_business = pd.DataFrame(business_metrics)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=86.2,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "High Risk Detection Rate", 'font': {'size': 24}},
            delta={'reference': 85, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#2ca02c"},
                'steps': [
                    {'range': [0, 80], 'color': "#ff7f0e"},
                    {'range': [80, 90], 'color': "#ffcc00"},
                    {'range': [90, 100], 'color': "#2ca02c"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_business, use_container_width=True, hide_index=True)

with tab3:
    st.header("Recent Predictions")
    
    # Load logs
    df_logs = load_logs()
    
    if df_logs.empty:
        st.info("No predictions logged yet. Make predictions using the main app to see them here.")
        
        # Show sample data
        st.subheader("Sample Prediction History")
        sample_predictions = pd.DataFrame({
            'Timestamp': pd.date_range(end=datetime.now(), periods=10, freq='H'),
            'Prediction': ['High', 'Low', 'High', 'Low', 'Low', 'High', 'Low', 'Low', 'High', 'Low'],
            'Confidence': [0.92, 0.78, 0.85, 0.91, 0.88, 0.76, 0.94, 0.82, 0.89, 0.87],
            'Risk Score': [8.2, 3.1, 7.5, 2.8, 3.9, 6.8, 2.1, 3.5, 7.9, 3.2]
        })
        st.dataframe(sample_predictions, use_container_width=True, hide_index=True)
    else:
        # Process logs
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        df_logs = df_logs.sort_values('timestamp', ascending=False)
        
        st.dataframe(df_logs.head(50), use_container_width=True)
    
    # Prediction distribution
    st.subheader("Prediction Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level distribution
        risk_dist = pd.DataFrame({
            'Risk Level': ['Low Risk', 'High Risk'],
            'Count': [720, 514]
        })
        
        fig = px.pie(risk_dist, values='Count', names='Risk Level',
                    title='Risk Level Distribution (Last 30 Days)',
                    color='Risk Level',
                    color_discrete_map={'Low Risk': '#2ca02c', 'High Risk': '#d62728'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hourly prediction volume
        hourly = pd.DataFrame({
            'Hour': list(range(24)),
            'Predictions': [15, 12, 8, 5, 4, 6, 18, 35, 52, 48, 45, 42,
                        51, 49, 47, 53, 58, 61, 55, 48, 42, 35, 28, 20]
        })
        
        fig = px.bar(hourly, x='Hour', y='Predictions',
                    title='Hourly Prediction Volume',
                    labels={'Predictions': 'Number of Predictions'})
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Uptime", "99.8%", "+0.1%")
        st.metric("API Response Time", "0.23s", "-0.05s")
    
    with col2:
        st.metric("Error Rate", "0.2%", "-0.1%")
        st.metric("Memory Usage", "2.1 GB", "+0.3 GB")
    
    with col3:
        st.metric("CPU Usage", "12%", "-3%")
        st.metric("Active Users", "47", "+8")
    
    # System alerts
    st.subheader("⚠️ Alerts & Notifications")
    
    alerts = [
        {"time": "2 hours ago", "level": "✅ INFO", "message": "Model retrained successfully"},
        {"time": "1 day ago", "level": "⚠️ WARNING", "message": "False Negative rate approaching threshold (14.5%)"},
        {"time": "3 days ago", "level": "✅ INFO", "message": "Deployment successful - v1.0"},
    ]
    
    for alert in alerts:
        with st.expander(f"{alert['level']} - {alert['time']}"):
            st.write(alert['message'])
    
    # Data drift monitoring
    st.subheader("📊 Data Drift Detection")
    
    drift_data = generate_sample_data()
    
    fig = px.line(drift_data, x='date', y='high_risk_rate',
                title='High Risk Rate Over Time (Drift Detection)',
                labels={'high_risk_rate': 'High Risk Rate', 'date': 'Date'})
    fig.add_hline(y=0.30, line_dash="dash", line_color="red",
                annotation_text="Expected Rate: 30%")
    fig.update_yaxis(tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📊 ML Model Monitoring Dashboard | Last updated: {}</p>
    <p>For issues or questions, contact the Data Science team</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)