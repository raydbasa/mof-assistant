#!/usr/bin/env python3
"""
Quick setup script to create sample data and train basic models..
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb
import joblib

def create_sample_data():
    """Create sample MOF synthesis dataset."""
    data = [
        {
            "Target": "MOF-5",
            "Metal source (mmol)": "Zn(NO3)2·6H2O (0.5)",
            "Linker(s) (mmol)": "H2BDC (0.5)",
            "Solvent(s) (mL)": "DMF (10)",
            "Modulator / Additive": "H2O (0.5)",
            "Temp (°C)": "120",
            "Time (h)": "24",
            "Pressure": "solvothermal",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, methanol, chloroform"
        },
        {
            "Target": "HKUST-1",
            "Metal source (mmol)": "Cu(NO3)2·3H2O (0.3)",
            "Linker(s) (mmol)": "H3BTC (0.2)",
            "Solvent(s) (mL)": "DMF:H2O (8:2)",
            "Modulator / Additive": "",
            "Temp (°C)": "85",
            "Time (h)": "12",
            "Pressure": "ambient",
            "Method": "room temperature",
            "Wash / Activation": "DMF, ethanol"
        },
        {
            "Target": "UiO-66",
            "Metal source (mmol)": "ZrCl4 (0.1)",
            "Linker(s) (mmol)": "H2BDC (0.1)",
            "Solvent(s) (mL)": "DMF (5)",
            "Modulator / Additive": "HCl (0.1)",
            "Temp (°C)": "120",
            "Time (h)": "48",
            "Pressure": "solvothermal",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, methanol, acetone"
        },
        {
            "Target": "MIL-101",
            "Metal source (mmol)": "Cr(NO3)3·9H2O (0.2)",
            "Linker(s) (mmol)": "H2BDC (0.2)",
            "Solvent(s) (mL)": "H2O (10)",
            "Modulator / Additive": "HF (0.1)",
            "Temp (°C)": "220",
            "Time (h)": "8",
            "Pressure": "solvothermal",
            "Method": "hydrothermal",
            "Wash / Activation": "H2O, ethanol"
        },
        {
            "Target": "ZIF-8",
            "Metal source (mmol)": "Zn(NO3)2·6H2O (0.5)",
            "Linker(s) (mmol)": "2-methylimidazole (2.0)",
            "Solvent(s) (mL)": "methanol (20)",
            "Modulator / Additive": "",
            "Temp (°C)": "25",
            "Time (h)": "2",
            "Pressure": "ambient",
            "Method": "room temperature",
            "Wash / Activation": "methanol"
        },
        {
            "Target": "MOF-74",
            "Metal source (mmol)": "Mg(NO3)2·6H2O (0.3)",
            "Linker(s) (mmol)": "H4DOBDC (0.1)",
            "Solvent(s) (mL)": "DMF:H2O:EtOH (6:2:2)",
            "Modulator / Additive": "TEA (0.1)",
            "Temp (°C)": "100",
            "Time (h)": "36",
            "Pressure": "solvothermal",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, methanol"
        },
        {
            "Target": "NU-1000",
            "Metal source (mmol)": "ZrCl4 (0.05)",
            "Linker(s) (mmol)": "H4TBAPy (0.05)",
            "Solvent(s) (mL)": "DMF (3)",
            "Modulator / Additive": "benzoic acid (0.1)",
            "Temp (°C)": "120",
            "Time (h)": "72",
            "Pressure": "solvothermal",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, methanol, acetone"
        },
        {
            "Target": "PCN-250",
            "Metal source (mmol)": "FeCl3·6H2O (0.2)",
            "Linker(s) (mmol)": "H3BTC (0.1)",
            "Solvent(s) (mL)": "DMF:H2O (8:2)",
            "Modulator / Additive": "HCl (0.05)",
            "Temp (°C)": "100",
            "Time (h)": "24",
            "Pressure": "solvothermal",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, methanol"
        },
        {
            "Target": "MOF-177",
            "Metal source (mmol)": "Zn(NO3)2·6H2O (0.1)",
            "Linker(s) (mmol)": "H3BTB (0.05)",
            "Solvent(s) (mL)": "DMF (5)",
            "Modulator / Additive": "H2O (0.1)",
            "Temp (°C)": "85",
            "Time (h)": "48",
            "Pressure": "solvothermal",
            "Method": "solvothermal",
            "Wash / Activation": "DMF, chloroform"
        },
        {
            "Target": "MIL-53",
            "Metal source (mmol)": "Al(NO3)3·9H2O (0.2)",
            "Linker(s) (mmol)": "H2BDC (0.2)",
            "Solvent(s) (mL)": "H2O (10)",
            "Modulator / Additive": "",
            "Temp (°C)": "150",
            "Time (h)": "12",
            "Pressure": "solvothermal",
            "Method": "hydrothermal",
            "Wash / Activation": "H2O, ethanol"
        }
    ]
    
    return pd.DataFrame(data)

def extract_numerical_features(df):
    """Extract numerical features from text columns."""
    df_processed = df.copy()
    
    # Extract temperature values
    temp_values = []
    for temp_str in df['Temp (°C)']:
        if pd.isna(temp_str) or temp_str == '':
            temp_values.append(np.nan)
        else:
            numbers = re.findall(r'\d+\.?\d*', str(temp_str))
            temp_values.append(float(numbers[0]) if numbers else np.nan)
    
    df_processed['temp_numeric'] = temp_values
    
    # Extract time values
    time_values = []
    for time_str in df['Time (h)']:
        if pd.isna(time_str) or time_str == '':
            time_values.append(np.nan)
        else:
            numbers = re.findall(r'\d+\.?\d*', str(time_str))
            time_values.append(float(numbers[0]) if numbers else np.nan)
    
    df_processed['time_numeric'] = time_values
    
    return df_processed

def train_models(df):
    """Train XGBoost models."""
    # Extract numerical features
    df_processed = extract_numerical_features(df)
    
    # Define feature columns
    feature_cols = [
        'Metal source (mmol)', 'Linker(s) (mmol)', 'Solvent(s) (mL)', 
        'Modulator / Additive', 'Pressure'
    ]
    
    X = df_processed[feature_cols].copy()
    y_temp = df_processed['temp_numeric']
    y_time = df_processed['time_numeric'] 
    y_method = df_processed['Method']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Pressure']),
            ('text', 'passthrough', ['Metal source (mmol)', 'Linker(s) (mmol)', 'Solvent(s) (mL)', 'Modulator / Additive'])
        ],
        remainder='drop'
    )
    
    # Train temperature model
    temp_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=42, n_estimators=50))
    ])
    
    temp_mask = ~y_temp.isna()
    temp_model.fit(X[temp_mask], y_temp[temp_mask])
    
    # Train time model
    time_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=42, n_estimators=50))
    ])
    
    time_mask = ~y_time.isna()
    time_model.fit(X[time_mask], y_time[time_mask])
    
    # Train method model
    method_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42, n_estimators=50))
    ])
    
    method_model.fit(X, y_method)
    
    return {
        'preprocessor': preprocessor,
        'temp_model': temp_model,
        'time_model': time_model,
        'method_model': method_model
    }

def main():
    """Main setup function."""
    print("Setting up MOF Synthesis Assistant...")
    
    # Create sample data
    print("Creating sample dataset...")
    df = create_sample_data()
    
    # Ensure directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save dataset
    df.to_csv('data/processed/mof_runs.csv', index=False)
    print(f"✅ Sample dataset created with {len(df)} rows")
    
    # Train models
    print("Training ML models...")
    models = train_models(df)
    
    # Save models
    joblib.dump(models['preprocessor'], 'models/preproc.joblib')
    joblib.dump(models['temp_model'], 'models/xgb_temp.joblib')
    joblib.dump(models['time_model'], 'models/xgb_time.joblib')
    joblib.dump(models['method_model'], 'models/xgb_method.joblib')
    
    print("✅ Models trained and saved")
    print("\nSetup complete! You can now run:")
    print("streamlit run app/app.py")

if __name__ == "__main__":
    main()
