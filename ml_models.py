from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import numpy as np
import pandas as pd

def determine_problem_type(y):
    """Determine problem type from target variable"""
    if pd.api.types.is_categorical_dtype(y) or y.nunique() < 10:
        return 'classification'
    elif pd.api.types.is_numeric_dtype(y):
        return 'regression'
    else:
        return 'classification'

def train_optimized_model(X, y, problem_type='auto', model_type='auto'):
    """Train model based on problem type"""
    if problem_type == 'auto':
        problem_type = determine_problem_type(y)
    
    # Model selection
    if problem_type == 'classification':
        model = RandomForestClassifier(n_estimators=100)
    elif problem_type == 'regression':
        model = RandomForestRegressor(n_estimators=100)
    else:
        raise ValueError("Unsupported problem type")
    
    # Train model
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    results = {
        'model': model,
        'importance': dict(zip(X.columns, model.feature_importances_))
    }
    
    if problem_type == 'classification':
        results['report'] = classification_report(y, y_pred)
    elif problem_type == 'regression':
        results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
        results['r2'] = r2_score(y, y_pred)
        results['y_pred'] = y_pred
    
    return results
