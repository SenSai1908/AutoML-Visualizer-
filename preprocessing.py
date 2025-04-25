import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_features(df, feature_cols, strategy='mean', encoding_type='one-hot', scaler_type='standard'):
    """Preprocess feature columns dynamically"""
    if not feature_cols:
        raise ValueError("No feature columns selected for preprocessing.")
    df = df[feature_cols].copy()
    
    # Separate numeric and categorical features
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numeric preprocessing
    numeric_transformer = SimpleImputer(strategy=strategy)
    
    # Categorical preprocessing
    if encoding_type == 'one-hot':
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    else:
        categorical_transformer = SimpleImputer(strategy='most_frequent')
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(df)
    
    # Scaling
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        return pd.DataFrame(X_processed, columns=numeric_features + categorical_features)
    
    X_scaled = scaler.fit_transform(X_processed)
    return pd.DataFrame(X_scaled, columns=preprocessor.get_feature_names_out())
