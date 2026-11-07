"""
Data preprocessing, feature engineering, and preparation pipeline for ChurnGuard-AI.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_clean_data(filepath='data/customer_churn.csv'):
    """
    Loads raw CSV data and performs necessary data sanitization:
    - Handles white-space/empty values in TotalCharges
    - Converts TotalCharges to numeric
    - Encodes binary target Churn (Yes: 1, No: 0)
    """
    df = pd.read_csv(filepath)
    
    # 1. Clean TotalCharges (11 records with empty spaces where tenure=0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    # If customer is new (tenure 0), TotalCharges is logically 0 or equal to MonthlyCharges
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # 2. Encode binary target Churn
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def feature_engineering(df):
    """
    Derives actionable business and behavioral features:
    - TotalServices: count of active add-on digital services
    - AvgMonthlyChargesRatio: TotalCharges / (tenure + 1) vs MonthlyCharges (trend check)
    - HasTechSupportOrSecurity: flag indicating critical support presence
    - LongTermContract: indicator for 1 or 2 year commitment
    - AutoPayment: flag indicating hands-free payments (less friction)
    """
    data = df.copy()
    
    # List of digital add-on services
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for s in services:
        if s in data.columns:
            data[f'has_{s}'] = (data[s] == 'Yes').astype(int)
    
    data['TotalServices'] = data[[f'has_{s}' for s in services]].sum(axis=1)
    
    # Support & Security bundle protection
    data['HasSecuritySupportBundle'] = ((data['OnlineSecurity'] == 'Yes') & (data['TechSupport'] == 'Yes')).astype(int)
    
    # Contract & Payment friction indicators
    data['IsMonthToMonth'] = (data['Contract'] == 'Month-to-month').astype(int)
    data['IsAutoPayment'] = data['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
    data['IsElectronicCheck'] = (data['PaymentMethod'] == 'Electronic check').astype(int)
    
    # Tenure Cohorts (early-life vs mature)
    data['TenureGroup'] = pd.cut(
        data['tenure'], 
        bins=[-1, 6, 12, 24, 48, 72], 
        labels=['0-6m', '6-12m', '1-2y', '2-4y', '4-6y']
    )
    
    # Charge per service efficiency
    data['ChargePerService'] = data['MonthlyCharges'] / (data['TotalServices'] + 1)
    
    return data

def prepare_pipeline_data(filepath='data/customer_churn.csv', test_size=0.2, random_state=42):
    """
    Full preparation workflow returning train/test splits, feature names, and raw reference test dataframe.
    """
    raw_df = load_and_clean_data(filepath)
    df_engineered = feature_engineering(raw_df)
    
    # Drop customerID and engineered helper columns
    drop_cols = ['customerID', 'TenureGroup']
    for s in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        drop_cols.append(f'has_{s}')
        
    features_df = df_engineered.drop(columns=[c for c in drop_cols if c in df_engineered.columns])
    
    y = features_df['Churn']
    X = features_df.drop(columns=['Churn'])
    
    # One-hot encode remaining categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    feature_names = X_encoded.columns.tolist()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'ChargePerService']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Keep reference to raw customer records in test set for dashboard demo
    raw_test_df = raw_df.loc[X_test.index].copy()
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'numeric_cols': numeric_cols,
        'raw_test_df': raw_test_df,
        'raw_df': raw_df
    }

if __name__ == '__main__':
    data_dict = prepare_pipeline_data()
    print("Preprocessing completed successfully!")
    print(f"X_train shape: {data_dict['X_train'].shape}, X_test shape: {data_dict['X_test'].shape}")
    print(f"Engineered features count: {len(data_dict['feature_names'])}")

