import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(BASE_DIR, 'bank_churn_preprocessing', 'train.csv')
    test_path = os.path.join(BASE_DIR, 'bank_churn_preprocessing', 'test.csv')
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.drop('Exited', axis=1)
    y_train = train_data['Exited']
    X_test = test_data.drop('Exited', axis=1)
    y_test = test_data['Exited']

    mlflow.set_experiment("Bank_Churn_Prediction")

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="CI_Pipeline_Run") as run:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        score = rf_model.score(X_test, y_test)
        print(f"Model training completed in CI Pipeline! Accuracy: {score:.4f}")
        
        mlflow.sklearn.log_model(rf_model, "model")
        
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    train_model()