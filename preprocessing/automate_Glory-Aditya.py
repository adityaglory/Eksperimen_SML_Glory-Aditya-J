import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_preprocessing(input_path, output_dir):
    print(f"Reading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    
    cols_to_drop = ['id', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    categorical_cols = ['Gender', 'Geography']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    print("-> Data Encoded Completely")
            
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    for col in numeric_cols:
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("-> Outlier Handling and Feature Scaling Completed")
    
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    print("-> Data Split Complete")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Preprocessing Done! Data saved to folder: {output_dir}")
    return train_data, test_data

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    RAW_DATA_PATH = os.path.join(BASE_DIR, '..', 'bank_churn_raw', 'bank_churn_raw.csv')
    
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'bank_churn_preprocessing')
    
    run_preprocessing(RAW_DATA_PATH, OUTPUT_FOLDER)