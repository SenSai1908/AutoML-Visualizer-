# app.py
import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import preprocess_features
from ml_models import train_optimized_model, determine_problem_type
from utils import save_model, load_model
import visualization as viz
import plotly.express as px
import plotly.graph_objects as go
import altair as alt

# Set page config for a wider layout
st.set_page_config(
    page_title="Advanced Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for PowerBI look and feel
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 0;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    div.stButton > button {
        background-color: #0078D4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .css-1d391kg, .css-12w0qpk {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #323130;
        font-weight: 600;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0078D4;
    }
    .metric-label {
        font-size: 14px;
        color: #605E5C;
    }
    .dashboard-title {
        padding: 10px;
        background-color: #0078D4;
        color: white;
        font-size: 24px;
        font-weight: bold;
        border-radius: 5px 5px 0 0;
    }
    /* PowerBI card styling */
    .powerbi-card {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    .powerbi-card-header {
        font-size: 16px;
        font-weight: 600;
        color: #323130;
        margin-bottom: 10px;
        border-bottom: 1px solid #EAEEF3;
        padding-bottom: 5px;
    }
    /* Grid container for dashboard layout */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        grid-gap: 10px;
    }
    .grid-item-full {
        grid-column: span 12;
    }
    .grid-item-half {
        grid-column: span 6;
    }
    .grid-item-third {
        grid-column: span 4;
    }
    .grid-item-quarter {
        grid-column: span 3;
    }
    /* Filter panel */
    .filter-panel {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'theme' not in st.session_state:
    st.session_state.theme = "light"  # Default theme

# --- Header ---
st.markdown('<div class="dashboard-title">Advanced Analytics Dashboard</div>', unsafe_allow_html=True)

# --- Sidebar: Data Upload & Settings ---
with st.sidebar:
    # PowerBI logo (you can replace with your own logo)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/New_Power_BI_Logo.svg/1200px-New_Power_BI_Logo.svg.png", width=100)
    st.markdown("### Data & Settings")
    
    # Theme selector
    theme = st.radio("Theme", ["Light", "Dark"], horizontal=True)
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # --- Column Mapping ---
        st.markdown("### Column Mapping")
        all_cols = df.columns.tolist()
        target_col = st.selectbox("Target Variable", options=all_cols)
        feature_cols = st.multiselect(
            "Feature Columns",
            options=[c for c in all_cols if c != target_col],
            key="feature_columns_sidebar"
        )
        
        # --- Preprocessing Options ---
        st.markdown("### Preprocessing")
        missing_strategy = st.selectbox("Missing Values", ["mean", "median", "most_frequent"])
        encoding_type = st.selectbox("Categorical Encoding", ["one-hot", "label"])
        scaler_type = st.selectbox("Scaling", ["standard", "minmax", "none"])
        
        # --- Model Selection ---
        st.markdown("### Model Selection")
        model_type = st.selectbox("Model Type", ["auto", "rf", "xgb", "lgbm", "catboost", "linear"])
        
        if not feature_cols:
            st.error("Please select at least one feature column.")
        else:
            # --- Preprocessing Pipeline ---
            try:
                X = preprocess_features(
                    df, 
                    feature_cols=feature_cols,
                    strategy=missing_strategy,
                    encoding_type=encoding_type,
                    scaler_type=scaler_type
                )
                if X.shape[1] == 0:
                    st.error("No features remain after preprocessing.")
                y = df[target_col]
            except Exception as e:
                st.error(f"Preprocessing failed: {str(e)}")
    else:
        st.info("Please upload a CSV file to get started.")

# Main content
if 'df' in locals():
    # --- Dashboard Layout ---
    # Top row: Key metrics
    st.markdown("## Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Records</div>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{df.shape[1]:,}</div>
                <div class="metric-label">Columns</div>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col3:
        missing_percentage = round((df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{missing_percentage}%</div>
                <div class="metric-label">Missing Data</div>
            </div>""", 
            unsafe_allow_html=True
        )
    
    with col4:
        if 'problem_type' in st.session_state and st.session_state.problem_type:
            problem_type = st.session_state.problem_type.capitalize()
        else:
            problem_type = determine_problem_type(y).capitalize()
            st.session_state.problem_type = problem_type.lower()
        
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{problem_type}</div>
                <div class="metric-label">Problem Type</div>
            </div>""", 
            unsafe_allow_html=True
        )

    # --- Main Tabs ---
    tabs = st.tabs(["📊 Data Explorer", "🔍 Feature Analysis", "🏋️ Model Training", "🔮 Predictions & What-If"])
    
    # --- Tab 1: Data Explorer ---
    with tabs[0]:
        st.markdown('<div class="powerbi-card-header">Data Sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(5), use_container_width=True)
        
        # Data statistics in a nice card
        st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
        st.markdown('<div class="powerbi-card-header">Data Statistics</div>', unsafe_allow_html=True)
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            stats_df = df[numeric_cols].describe().T
            st.dataframe(stats_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data visualization in a card
        st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
        st.markdown('<div class="powerbi-card-header">Data Visualization</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if numeric_cols:
                selected_dist_col = st.selectbox("Select column for distribution", options=numeric_cols)
                
                fig_dist = px.histogram(
                    df, 
                    x=selected_dist_col,
                    color_discrete_sequence=["#0078D4"],
                    title=f"Distribution of {selected_dist_col}"
                )
                fig_dist.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="#323130"),
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Time series visualization if date columns exist
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                date_col = st.selectbox("Select date column", options=date_cols)
                value_col = st.selectbox("Select value column", options=numeric_cols)
                
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    trend_df = df.groupby(df[date_col].dt.date)[value_col].mean().reset_index()
                    
                    fig_trend = px.line(
                        trend_df,
                        x=date_col,
                        y=value_col,
                        title=f"{value_col} Trend Over Time",
                        color_discrete_sequence=["#0078D4"]
                    )
                    fig_trend.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(color="#323130"),
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                except:
                    st.info("Select date and value columns to see trends over time")
            else:
                # If no date columns, show a bar chart
                if numeric_cols and len(numeric_cols) >= 2:
                    x_col = st.selectbox("Select X-axis column", options=numeric_cols, index=0)
                    y_col = st.selectbox("Select Y-axis column", options=numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    fig_bar = px.bar(
                        df.sample(min(100, len(df))),  # Sample to avoid overcrowding
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} by {x_col}",
                        color_discrete_sequence=["#0078D4"]
                    )
                    fig_bar.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(color="#323130"),
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Tab 2: Feature Analysis ---
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
            st.markdown('<div class="powerbi-card-header">Correlation Matrix</div>', unsafe_allow_html=True)
            
            numeric_df = df.select_dtypes(include=['number'])
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues"
                )
                fig_corr.update_layout(
                    height=500,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
            st.markdown('<div class="powerbi-card-header">Feature Relationships</div>', unsafe_allow_html=True)
            
            if len(numeric_cols) >= 2:
                x_axis = st.selectbox("X-axis", options=numeric_cols, key="scatter_x")
                y_axis = st.selectbox("Y-axis", options=numeric_cols, key="scatter_y")
                color_option = st.selectbox("Color by", options=["None"] + df.columns.tolist())
                
                if color_option == "None":
                    fig_scatter = px.scatter(
                        df, 
                        x=x_axis, 
                        y=y_axis,
                        color_discrete_sequence=["#0078D4"]
                    )
                else:
                    fig_scatter = px.scatter(
                        df, 
                        x=x_axis, 
                        y=y_axis, 
                        color=color_option,
                        color_continuous_scale="viridis" if df[color_option].dtype.kind in 'ifc' else None
                    )
                
                fig_scatter.update_layout(
                    height=500,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for scatter plot")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # PCA Visualization (similar to third image)
        st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
        st.markdown('<div class="powerbi-card-header">PCA Cluster Analysis</div>', unsafe_allow_html=True)
        
        if len(numeric_cols) > 2:
            col1, col2 = st.columns(2)
            
            with col1:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                # Prepare data for PCA
                numeric_data = df[numeric_cols].copy()
                numeric_data = numeric_data.fillna(numeric_data.mean())
                
                # Scale the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                # Apply PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                
                # Create a DataFrame with PCA results
                pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                
                # Clustering (K-Means)
                from sklearn.cluster import KMeans
                
                n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Add clusters to PCA dataframe
                pca_df['Cluster'] = clusters
                
                # Create scatter plot with clusters
                fig_pca = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title="PCA - Cluster Results",
                    color_continuous_scale="viridis"
                )
                fig_pca.update_layout(
                    height=400,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_pca, use_container_width=True)
            
            with col2:
                # Count by cluster
                cluster_counts = pd.DataFrame(pca_df['Cluster'].value_counts()).reset_index()
                cluster_counts.columns = ['Cluster', 'Number of Records']
                
                fig_counts = px.bar(
                    cluster_counts,
                    x='Cluster',
                    y='Number of Records',
                    title="Number of Records per Cluster",
                    color='Cluster',
                    color_continuous_scale="viridis"
                )
                fig_counts.update_layout(
                    height=400,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_counts, use_container_width=True)
            
            # Add cluster to original dataframe for analysis
            df_with_clusters = df.copy()
            df_with_clusters['Cluster'] = clusters
            
            # Create summary statistics by cluster
            cluster_analysis = df_with_clusters.groupby('Cluster')[numeric_cols].mean().round(2)
            st.markdown('<div class="powerbi-card-header">Cluster Characteristics</div>', unsafe_allow_html=True)
            st.dataframe(cluster_analysis, use_container_width=True)
        else:
            st.info("Need at least 3 numeric columns for PCA cluster analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Tab 3: Model Training ---
    with tabs[2]:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
            st.markdown('<div class="powerbi-card-header">Model Configuration</div>', unsafe_allow_html=True)
            
            st.write(f"Problem Type: **{st.session_state.problem_type.capitalize()}**")
            st.write(f"Selected Model Type: **{model_type.upper()}**")
            
            st.markdown('<div class="powerbi-card-header" style="margin-top: 15px;">Feature Selection</div>', unsafe_allow_html=True)
            st.write(f"Target Variable: **{target_col}**")
            st.write(f"Number of Features: **{len(feature_cols)}**")
            
            st.markdown('<div class="powerbi-card-header" style="margin-top: 15px;">Training Controls</div>', unsafe_allow_html=True)
            train_button = st.button("Train Model", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
            st.markdown('<div class="powerbi-card-header">Model Performance</div>', unsafe_allow_html=True)
            
            if train_button or st.session_state.model:
                with st.spinner("Training model..."):
                    try:
                        if train_button or not st.session_state.model:
                            results = train_optimized_model(
                                X, y, 
                                problem_type=st.session_state.problem_type, 
                                model_type=model_type
                            )
                            st.session_state.model = results['model']
                            st.session_state.results = results
                            save_model(st.session_state.model, "saved_models/latest_model.pkl")
                        
                        # Show results based on problem type
                        if st.session_state.problem_type == 'classification':
                            st.markdown('<div class="powerbi-card-header" style="margin-top: 15px;">Classification Metrics</div>', unsafe_allow_html=True)
                            
                            # Create metrics cards in a row
                            import re
                            report = st.session_state.results['report']
                            
                            # Extract accuracy from report
                            accuracy_match = re.search(r"accuracy\s+(\d+\.\d+)", report)
                            if accuracy_match:
                                accuracy = float(accuracy_match.group(1))
                            else:
                                accuracy = 0
                            
                            # Extract f1 (weighted avg) from report
                            f1_match = re.search(r"weighted avg\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)", report)
                            if f1_match:
                                f1_score = float(f1_match.group(1))
                            else:
                                f1_score = 0
                                
                            # Metrics row
                            mcol1, mcol2, mcol3 = st.columns(3)
                            
                            with mcol1:
                                st.markdown(
                                    f"""<div class="metric-card">
                                        <div class="metric-value">{accuracy:.3f}</div>
                                        <div class="metric-label">Accuracy</div>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                            
                            with mcol2:
                                st.markdown(
                                    f"""<div class="metric-card">
                                        <div class="metric-value">{f1_score:.3f}</div>
                                        <div class="metric-label">F1 Score (Weighted)</div>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                            
                            with mcol3:
                                importance_values = list(st.session_state.results['importance'].values())
                                top_feature_idx = importance_values.index(max(importance_values))
                                top_feature = list(st.session_state.results['importance'].keys())[top_feature_idx]
                                
                                st.markdown(
                                    f"""<div class="metric-card">
                                        <div class="metric-value">{top_feature}</div>
                                        <div class="metric-label">Top Feature</div>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                            
                            # Classification report
                            st.markdown('<div class="powerbi-card-header" style="margin-top: 15px;">Detailed Classification Report</div>', unsafe_allow_html=True)
                            st.text(report)
                            
                        elif st.session_state.problem_type == 'regression':
                            # Metrics row for regression
                            mcol1, mcol2, mcol3 = st.columns(3)
                            
                            with mcol1:
                                st.markdown(
                                    f"""<div class="metric-card">
                                        <div class="metric-value">{st.session_state.results['rmse']:.3f}</div>
                                        <div class="metric-label">RMSE</div>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                            
                            with mcol2:
                                st.markdown(
                                    f"""<div class="metric-card">
                                        <div class="metric-value">{st.session_state.results['r2']:.3f}</div>
                                        <div class="metric-label">R²</div>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                            
                            with mcol3:
                                importance_values = list(st.session_state.results['importance'].values())
                                top_feature_idx = importance_values.index(max(importance_values))
                                top_feature = list(st.session_state.results['importance'].keys())[top_feature_idx]
                                
                                st.markdown(
                                    f"""<div class="metric-card">
                                        <div class="metric-value">{top_feature}</div>
                                        <div class="metric-label">Top Feature</div>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                                
                            # Predicted vs Actual plot
                            st.markdown('<div class="powerbi-card-header" style="margin-top: 15px;">Predicted vs Actual Values</div>', unsafe_allow_html=True)
                            fig_pva = px.scatter(
                                x=y, 
                                y=st.session_state.results['y_pred'],
                                labels={"x": "Actual", "y": "Predicted"},
                                color_discrete_sequence=["#0078D4"]
                            )
                            
                            # Add 45-degree line
                            fig_pva.add_shape(
                                type="line",
                                x0=min(y), y0=min(y),
                                x1=max(y), y1=max(y),
                                line=dict(color="red", dash="dash")
                            )
                            
                            fig_pva.update_layout(
                                plot_bgcolor="white",
                                paper_bgcolor="white",
                                height=400
                            )
                            
                            st.plotly_chart(fig_pva, use_container_width=True)
                        
                        # Feature importance visualization
                        st.markdown('<div class="powerbi-card-header" style="margin-top: 15px;">Feature Importance</div>', unsafe_allow_html=True)
                        importance_df = pd.DataFrame({
                            'Feature': list(st.session_state.results['importance'].keys()),
                            'Importance': list(st.session_state.results['importance'].values())
                        }).sort_values(by='Importance', ascending=False)
                        
                        fig_imp = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale="Blues"
                        )
                        fig_imp.update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            height=400
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.exception(e)
            else:
                st.info("Click 'Train Model' to start training")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Tab 4: Predictions & What-If ---
    with tabs[3]:
        if st.session_state.model:
            st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
            st.markdown('<div class="powerbi-card-header">What-If Analysis</div>', unsafe_allow_html=True)
            st.write("Adjust feature values to see how they affect predictions")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                inputs = {}
                with st.form("prediction_form"):
                    for feature in X.columns:
                        feature_name = feature.split('__')[-1] if '__' in feature else feature
                        
                        # Determine a sensible default value
                        if feature in df.columns:
                            default_val = float(df[feature].mean())
                        else:
                            default_val = 0.0
                            
                        inputs[feature] = st.number_input(
                            f"{feature_name}", 
                            value=default_val,
                            format="%.2f"
                        )
                    
                    predict_button = st.form_submit_button("Predict")
                
                if predict_button:
                    try:
                        X_input = pd.DataFrame([inputs])
                        prediction = st.session_state.model.predict(X_input)
                        
                        st.markdown('<div class="powerbi-card-header" style="margin-top: 15px;">Prediction Result</div>', unsafe_allow_html=True)
                        st.markdown(
                            f"""<div class="metric-card" style="margin-top: 20px">
                                <div class="metric-value">{prediction[0]:.4f}</div>
                                <div class="metric-label">Predicted {target_col}</div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
            
            with col2:
                # What-if visualization
                st.markdown('<div class="powerbi-card-header">Impact Analysis</div>', unsafe_allow_html=True)
                
                if 'results' in st.session_state:
                    importance_df = pd.DataFrame({
                        'Feature': list(st.session_state.results['importance'].keys()),
                        'Importance': list(st.session_state.results['importance'].values())
                    }).sort_values(by='Importance', ascending=False).head(10)
                    
                    # Sensitivity analysis (show top feature impact)
                    if len(importance_df) > 0:
                        top_feature = importance_df.iloc[0]['Feature']
                        top_feature_name = top_feature.split('__')[-1] if '__' in top_feature else top_feature
                        
                        st.write(f"Sensitivity analysis for **{top_feature_name}**")
                        
                        # Get range for the top feature
                        if top_feature in df.columns:
                            min_val = float(df[top_feature].min())
                            max_val = float(df[top_feature].max())
                            step = (max_val - min_val) / 10
                            if step == 0:
                                step = 1.0  # Avoid division by zero
                        else:
                            min_val, max_val, step = 0.0, 1.0, 0.1

                        feature_range = np.arange(min_val, max_val + step, step)
                        whatif_inputs = []
                        for val in feature_range:
                            temp_input = inputs.copy()
                            temp_input[top_feature] = val
                            whatif_inputs.append(temp_input)
                        X_whatif = pd.DataFrame(whatif_inputs)
                        try:
                            preds = st.session_state.model.predict(X_whatif)
                        except Exception as e:
                            st.error(f"Could not compute what-if predictions: {str(e)}")
                            preds = np.zeros_like(feature_range)

                        fig_whatif = go.Figure()
                        fig_whatif.add_trace(go.Scatter(
                            x=feature_range,
                            y=preds,
                            mode='lines+markers',
                            line=dict(color="#0078D4"),
                            name="Prediction"
                        ))
                        fig_whatif.update_layout(
                            title=f"Predicted {target_col} vs {top_feature_name}",
                            xaxis_title=top_feature_name,
                            yaxis_title=f"Predicted {target_col}",
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font=dict(color="#323130"),
                            height=400
                        )
                        st.plotly_chart(fig_whatif, use_container_width=True)
                    else:
                        st.info("No feature importance available for sensitivity analysis.")
                else:
                    st.info("Train a model to enable impact analysis.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Train a model to enable predictions and what-if analysis.")
