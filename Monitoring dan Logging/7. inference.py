import requests
import time
import random

EXPORTER_URL = "http://localhost:8000/predict"

def generate_dummy_data():
    return {
        "dataframe_split": {
            "columns": ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"],
            "data": [
                [
                    random.randint(500, 800),
                    random.choice([0, 1, 2]),
                    random.choice([0, 1]),
                    random.randint(25, 65),
                    random.randint(1, 10),
                    random.uniform(0.0, 150000.0),
                    random.randint(1, 4),
                    random.choice([0, 1]),
                    random.choice([0, 1]),
                    random.uniform(30000.0, 120000.0)
                ]
            ]
        }
    }

print("Starting simulation...")
print("Press Ctrl+C to stop.")

try:
    for i in range(1, 201):
        data = generate_dummy_data()
        try:
            response = requests.post(EXPORTER_URL, json=data)
            print(f"Request {i}: {response.json()}")
        except Exception as e:
            print(f"Failed to send request {i}")
        
        time.sleep(random.uniform(0.5, 2.0))
        
except KeyboardInterrupt:
    print("\nSimulation stopped.")