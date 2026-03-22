import pandas as pd
import numpy as np
import os

def engineer_features(data_path, output_path):
    print(f"Reading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Time-based Features
    # The 'Time' feature is seconds from the first transaction in the dataset.
    # We can convert it to 'Hour of Day' (assuming a 24-hour cycle)
    df['Hour'] = (df['Time'] // 3600) % 24
    
    # 2. Amount-based Features
    # 'Amount' is often highly skewed, so log-transformation helps models.
    # Adding a small constant (0.001) to avoid log(0)
    df['Log_Amount'] = np.log1p(df['Amount'])
    
    # 3. Transaction Velocity (Simulated)
    # Since we don't have UserID, we can't do true per-user velocity.
    # However, we can track the "Global Velocity" or "Amount standard deviation" 
    # over a sliding window to capture bursts of activity.
    df['Amount_to_Mean_Ratio'] = df['Amount'] / (df['Amount'].mean() + 1e-5)
    
    # 4. Interaction Features (Top PCA features)
    # V17, V14, V12, V10 are typically the most important features in this dataset.
    df['V17_V14'] = df['V17'] * df['V14']
    df['V12_V10'] = df['V12'] * df['V10']

    # Drop the original 'Time' since 'Hour' is more representative for a periodic model
    # We keep 'Amount' for now but might use 'Log_Amount' for training.
    
    print(f"Feature engineering complete. Original features: 31, New features: {len(df.columns)}")
    
    # Save the engineered dataset
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    df.to_csv(output_path, index=False)
    print(f"Engineered data saved to {output_path}")

if __name__ == "__main__":
    engineer_features("data/creditcard.csv", "data/creditcard_engineered.csv")
