# utils.py
import joblib
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime

def save_model(model, filename):
    """Save trained model to file with directory creation if needed"""
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    joblib.dump(model, filename)

def load_model(filename):
    """Load a trained model from a file with error handling"""
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        st.error(f"Model file not found at {filename}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_theme():
    """Create a consistent theme for visualizations"""
    return {
        'color_primary': '#0078D4',
        'color_secondary': '#41A5EE',
        'color_accent': '#FFB900',
        'color_background': '#FFFFFF',
        'color_text': '#323130',
        'color_grid': '#EAEEF3',
    }

def create_metric_card(title, value, delta=None, delta_suffix="%", is_upward_good=True):
    """Create a PowerBI-style metric card"""
    delta_color = ""
    delta_arrow = ""
    if delta is not None:
        if delta > 0:
            delta_color = "green" if is_upward_good else "red"
            delta_arrow = "↑"
        elif delta < 0:
            delta_color = "red" if is_upward_good else "green"
            delta_arrow = "↓"
        
        delta_html = f"""<span style="color:{delta_color}">
            {delta_arrow} {abs(delta):.1f}{delta_suffix}
        </span>"""
    else:
        delta_html = ""
    
    html = f"""
    <div style="background-color:white; border-radius:5px; padding:15px; margin:5px; box-shadow:0px 2px 5px rgba(0,0,0,0.1);">
        <h3 style="margin:0; font-size:14px; color:#605E5C;">{title}</h3>
        <p style="font-size:28px; font-weight:bold; margin:5px 0; color:#0078D4;">{value}</p>
        {delta_html}
    </div>
    """
    
    return html

def generate_summary_report(model, X, y, problem_type):
    """Generate a comprehensive summary report of model performance"""
    # Create a dictionary to store report data
    report = {}
    
    # Model information
    report['model_type'] = type(model).__name__
    report['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Data information
    report['num_samples'] = len(X)
    report['num_features'] = X.shape[1]
    
    # Generate predictions
    y_pred = model.predict(X)
    
    # Performance metrics based on problem type
    if problem_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        report['accuracy'] = accuracy_score(y, y_pred)
        report['precision'] = precision_score(y, y_pred, average='weighted')
        report['recall'] = recall_score(y, y_pred, average='weighted')
        report['f1'] = f1_score(y, y_pred, average='weighted')
    else:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        report['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
        report['r2'] = r2_score(y, y_pred)
        report['mae'] = mean_absolute_error(y, y_pred)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        top_features = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        report['top_features'] = top_features
    
    return report

def export_to_excel(df, filename="export.xlsx"):
    """Export dataframe to Excel file and provide download link"""
    # Create Excel file
    excel_file = df.to_excel(filename, index=False, engine='openpyxl')
    
    # Read file for download
    with open(filename, "rb") as f:
        data = f.read()
    
    # Create download link
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'
    
    return href
