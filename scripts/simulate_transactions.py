import pandas as pd
import requests
import time
import json
import os
import random

def simulate_realtime(data_path, api_url):
    print(f"Loading test transactions from {data_path}...")
    df = pd.read_csv(data_path)
    # Filter for some fraud examples to make it interesting
    fraud_indices = df[df['Class'] == 1].index.tolist()
    normal_indices = df[df['Class'] == 0].sample(100).index.tolist()
    
    # Mix them up
    test_indices = random.sample(fraud_indices[:10] + normal_indices, 50)
    
    print(f"\nStarting Simulation on {api_url}/predict...")
    print("-" * 50)
    
    for idx in test_indices:
        row = df.loc[idx]
        payload = row.drop('Class').to_dict()
        
        try:
            start_time = time.time()
            response = requests.post(f"{api_url}/predict", json=payload)
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                label_str = "FRAUD" if result['fraud_label'] == 1 else "NORMAL"
                actual_str = "FRAUD" if row['Class'] == 1 else "NORMAL"
                
                print(f"TX {idx: >6} | Amount: ${row['Amount']: >8.2f} | Status: {label_str: <7} (Actual: {actual_str: <7}) | Latency: {latency: >6.2f}ms")
                
                if result['fraud_label'] == 1:
                    print(f"   ㄴ Explanation: {json.dumps(result['explanation'], indent=2)}")
            else:
                print(f"Error for TX {idx}: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Simulation failed for TX {idx}: {str(e)}")
            
        time.sleep(0.5) # Wait between transactions

if __name__ == "__main__":
    # Ensure raw data is available
    data_path = "data/creditcard.csv"
    api_url = "http://localhost:8000"
    simulate_realtime(data_path, api_url)
