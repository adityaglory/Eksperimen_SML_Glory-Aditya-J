import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

REPO_OWNER = "adityaglory" 
REPO_NAME = "SML-repo"

dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_experiment("Bank_Churn_Prediction_Advance")

def train_and_tune_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(BASE_DIR, 'bank_churn_preprocessing', 'train.csv')
    test_path = os.path.join(BASE_DIR, 'bank_churn_preprocessing', 'test.csv')

    print("Loading dataset for tuning...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop('Exited', axis=1)
    y_train = train_data['Exited']
    X_test = test_data.drop('Exited', axis=1)
    y_test = test_data['Exited']

    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [10, 20, 50, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    print("Starting Hyperparameter Tuning...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    
    with mlflow.start_run(run_name="RandomForest_Tuned_DagsHub"):
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        mlflow.log_params(grid_search.best_params_)
        
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        
        print(f"Tuning completed! Accuracy: {acc:.4f}")
        
        report = classification_report(y_test, y_pred)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        
        plt.figure(figsize=(10, 6))
        feature_importances = best_model.feature_importances_
        sns.barplot(x=feature_importances, y=X_train.columns)
        plt.title('Feature Importance - Random Forest')
        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        
        signature = mlflow.models.signature.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model", signature=signature)
        
        print("All logs (params, metrics, model, artifacts) sent to DagsHub successfully!")

if __name__ == "__main__":
    train_and_tune_model()