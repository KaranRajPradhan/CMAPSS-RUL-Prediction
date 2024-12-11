import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os

# Base path and files
base_path = "CMAPSSData/"
train_files = ["train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt"]
test_files = ["test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt"]
rul_files = ["RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt"]

columns = [
    "EngineID", "Cycle", "Op1", "Op2", "Op3",
    "Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5",
    "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10",
    "Sensor11", "Sensor12", "Sensor13", "Sensor14", "Sensor15",
    "Sensor16", "Sensor17", "Sensor18", "Sensor19", "Sensor20",
    "Sensor21"
]

def load_data(train_files, test_files, rul_files):
    all_train_data = []
    all_test_data = []

    for train_file, test_file, rul_file in zip(train_files, test_files, rul_files):
        # Load train data
        train_data = pd.read_csv(os.path.join(base_path, train_file), sep="\s+", header=None, names=columns)
        max_cycles = train_data.groupby("EngineID")["Cycle"].transform("max")
        train_data["RUL"] = max_cycles - train_data["Cycle"]
        all_train_data.append(train_data)

        # Load test data
        test_data = pd.read_csv(os.path.join(base_path, test_file), sep="\s+", header=None, names=columns)

        # Load RUL data
        rul_data = pd.read_csv(os.path.join(base_path, rul_file), sep="\s+", header=None, names=["RUL"])
        unique_engines = test_data["EngineID"].unique()
        rul_mapping = dict(zip(unique_engines, rul_data["RUL"]))
        test_data["RUL"] = test_data["EngineID"].map(rul_mapping)

        all_test_data.append(test_data)

    # Combine all train and test data
    train_data_combined = pd.concat(all_train_data, ignore_index=True)
    test_data_combined = pd.concat(all_test_data, ignore_index=True)

    return train_data_combined, test_data_combined

def preprocess_data(train_data, test_data):
    # Normalize sensor data
    scaler = MinMaxScaler()
    sensor_columns = [f"Sensor{i}" for i in range(1, 22)]
    train_data[sensor_columns] = scaler.fit_transform(train_data[sensor_columns])
    test_data[sensor_columns] = scaler.transform(test_data[sensor_columns])

    # Prepare features
    X_train = train_data[sensor_columns]
    y_train = train_data["RUL"]
    X_test = test_data[sensor_columns]
    y_test = test_data["RUL"]

    return X_train, y_train, X_test, y_test, test_data

def train_xgb_model(X_train, y_train):
    # Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, test_data):
    # Predict RUL
    y_pred = model.predict(X_test)

    # Evaluate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

    # Save predictions
    results = test_data.copy()
    results["Predicted_RUL"] = y_pred
    results[["EngineID", "Cycle", "RUL", "Predicted_RUL"]].to_csv("predicted_rul_results.csv", index=False)

    print("Predicted RUL results saved to 'predicted_rul_results.csv'.")

    return mae, rmse, r2

def main():
    # Load data
    train_data, test_data = load_data(train_files, test_files, rul_files)

    # Preprocess data
    X_train, y_train, X_test, y_test, test_data = preprocess_data(train_data, test_data)

    # Train XGBoost model
    model = train_xgb_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, test_data)

if __name__ == "__main__":
    main()
