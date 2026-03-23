import httpx
import json
import time

API_URL = "http://localhost:8000"

test_cases = [
    {"Amount": 10.0, "V1": 0.5, "V2": -0.2},
    {"Amount": 2500.0, "V1": -3.5, "V2": 2.1}, # Potential fraud (High amount + extremes)
    {"Amount": 55.20, "V1": 0.1, "V2": 0.1},
    {"Amount": 5000.0, "V1": -1.0, "V2": -1.0}, # High amount
    {"Amount": 1.0, "V1": 0.0, "V2": 0.0},
    {"Amount": 999.99, "V1": 1.5, "V2": -1.5},
    {"Amount": 42.42, "V1": -0.5, "V2": 0.5},
    {"Amount": 8800.0, "V1": 2.0, "V2": 2.0},
    {"Amount": 15.0, "V1": -2.0, "V2": 2.0},
    {"Amount": 300.0, "V1": 0.3, "V2": -0.3}
]

def run_sweep():
    print(f"--- Starting Integration Sweep at {API_URL} ---")
    results = []
    
    with httpx.Client() as client:
        # Check Health
        try:
            health = client.get(f"{API_URL}/health")
            print(f"Health Check: {health.status_code}")
        except Exception as e:
            print(f"Server is not running: {e}")
            return

        for i, case in enumerate(test_cases):
            # Fill V3-V28
            full_case = {f"V{j}": 0.0 for j in range(3, 29)}
            full_case.update(case)
            full_case["Time"] = int(time.time()) % 100000
            
            print(f"Test case {i+1}: Amount=${case['Amount']}...", end=" ")
            try:
                resp = client.post(f"{API_URL}/predict", json=full_case)
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"SUCCESS | Label: {data['fraud_label']} | Prob: {data['fraud_probability']:.4f}")
                    results.append(data)
                else:
                    print(f"FAILED | Status: {resp.status_code}")
            except Exception as e:
                print(f"ERROR: {e}")
                
    print(f"\n--- Sweep Complete: {len(results)}/10 passed ---")

if __name__ == "__main__":
    run_sweep()
