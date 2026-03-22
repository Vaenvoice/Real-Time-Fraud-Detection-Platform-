import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visual style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def perform_eda(data_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Basic Info
    print("\n--- Data Overview ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    # Class Imbalance
    print("\n--- Class Distribution ---")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"Fraud Ratio: {class_counts[1]/len(df):.4%}")
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df, palette='viridis')
    plt.title('Transaction Class Distribution (0: Normal, 1: Fraud)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # Amount Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, kde=True, color='red', label='Fraud')
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, kde=True, color='blue', label='Normal', alpha=0.1)
    plt.title('Transaction Amount Distribution')
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'amount_distribution.png'))
    plt.close()
    
    # Time Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['Class'] == 1]['Time'], bins=50, color='red', label='Fraud')
    plt.title('Fraudulent Transactions over Time')
    plt.savefig(os.path.join(output_dir, 'fraud_time_distribution.png'))
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(15, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='RdBu', annot=False)
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print(f"\nEDA completed. Visualizations saved in {output_dir}")

if __name__ == "__main__":
    data_path = "data/creditcard.csv"
    output_dir = "reports/eda"
    perform_eda(data_path, output_dir)
