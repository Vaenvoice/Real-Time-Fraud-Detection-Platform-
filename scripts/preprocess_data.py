import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(data_path, output_dir):
    print(f"Reading engineered data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Define features and target
    X = df.drop(['Class', 'Time'], axis=1) # Drop original Time, use Hour
    y = df['Class']
    
    # 1. Train-Test Split (with stratification to preserve fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nInitial Train-Test Split:")
    print(f"Train size: {len(X_train)} (Fraud: {y_train.sum()})")
    print(f"Test size: {len(X_test)} (Fraud: {y_test.sum()})")
    
    # 2. Feature Scaling
    # Crucial for models like Logistic Regression and SVN.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Handling Class Imbalance (SMOTE)
    # Perform oversampling only on the TRAINING set to avoid data leakage
    print("\nApplying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Resampled Train size: {len(X_train_resampled)} (Fraud: {y_train_resampled.sum()})")
    print(f"Fraud Ratio after SMOTE: {y_train_resampled.mean():.0%}")
    
    # 4. Save the artifacts
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save processed data
    joblib.dump((X_train_resampled, X_test_scaled, y_train_resampled, y_test), os.path.join(output_dir, 'processed_data.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    print(f"\nPreprocessing complete. Artifacts saved in {output_dir}")

if __name__ == "__main__":
    preprocess_data("data/creditcard_engineered.csv", "data/processed")
