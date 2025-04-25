import streamlit as st
import pandas as pd
import plotly.express as px

def display_basic_stats(df):
    st.subheader("Basic Statistics")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Missing Values:", df.isnull().sum().to_dict())

def plot_distributions(df, numeric_cols):
    if len(numeric_cols) == 0:
        return
    selected = st.multiselect("Select numeric columns", options=numeric_cols)
    if selected:
        fig = px.histogram(df, x=selected, nbins=50, title="Distributions")
        st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df, numeric_cols):
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(importance_dict, feature_names):
    fig = px.bar(
        x=feature_names,
        y=importance_dict.values(),
        labels={'x': 'Features', 'y': 'Importance'},
        title="Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_predicted_vs_actual(y_true, y_pred):
    fig = px.scatter(x=y_true, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
    fig.add_shape(type='line', x0=y_true.min(), y0=y_true.min(), x1=y_true.max(), y1=y_true.max())
    st.plotly_chart(fig, use_container_width=True)

def create_what_if_scenario(model, feature_names):
    inputs = {}
    with st.form("prediction_form"):
        for feature in feature_names:
            inputs[feature] = st.number_input(feature)
        if st.form_submit_button("Predict"):
            try:
                X = pd.DataFrame([inputs])
                pred = model.predict(X)
                st.metric("Prediction", pred[0])
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
