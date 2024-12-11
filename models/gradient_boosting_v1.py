import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import os
from pickle import dump, load


base_path = "CMAPSSData/"
# train_files = ["train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt"]
# test_files = ["test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt"]
# rul_files = ["RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt"]

train_files = ["train_FD001.txt"]
test_files = ["test_FD001.txt"]
rul_files = ["RUL_FD001.txt"]

columns = [
        "EngineID", "Cycle", "Op1", "Op2", "Op3",
        "Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5",
        "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10",
        "Sensor11", "Sensor12", "Sensor13", "Sensor14", "Sensor15",
        "Sensor16", "Sensor17", "Sensor18", "Sensor19", "Sensor20",
        "Sensor21"
    ]

def load_train_data(train_files):
    train_data_list = []

    for file in train_files:
        data = pd.read_csv(base_path + file, sep="\s+", header=None, names=columns)
        train_data_list.append(data)

    # Concatenate all train data
    train_data = pd.concat(train_data_list, axis=0).reset_index(drop=True)

    # Compute RUL for training target
    max_cycles = train_data.groupby("EngineID")["Cycle"].transform("max")
    train_data["RUL"] = max_cycles - train_data["Cycle"]

    # Normalize sensor data
    scaler = MinMaxScaler()
    sensor_columns = [f"Sensor{i}" for i in range(1, 22)]
    train_data[sensor_columns] = scaler.fit_transform(train_data[sensor_columns])

    X_train = train_data.drop(["EngineID", "Cycle", "RUL", "Sensor14", "Sensor11"], axis=1)
    y_train = train_data["RUL"]
    
    return X_train, y_train

def load_test_data(test_files, rul_files):
    test_data_list = []

    for file in test_files:
        data = pd.read_csv(base_path + file, sep="\s+", header=None, names=columns)
        test_data_list.append(data)

    # Concatenate all test data
    test_data = pd.concat(test_data_list, axis=0).reset_index(drop=True)

    # Load RUL data and concatenate if multiple files
    rul_data_list = []
    for file in rul_files:
        data = pd.read_csv(base_path + file, sep="\s+", header=None, names=["RUL"])
        rul_data_list.append(data)

    rul_data = pd.concat(rul_data_list, axis=0).reset_index(drop=True)

    # Normalize sensor data using the scaler fitted on train data
    scaler = MinMaxScaler()
    sensor_columns = [f"Sensor{i}" for i in range(1, 22)]
    test_data[sensor_columns] = scaler.fit_transform(test_data[sensor_columns])

    # Assign RUL to all rows in test data based on EngineID
    test_data["RUL"] = test_data["EngineID"].apply(lambda engine_id: rul_data.iloc[engine_id - 1, 0])

    # Prepare features and target
    X_test = test_data.drop(["EngineID", "Cycle", "RUL", "Sensor14", "Sensor11"], axis=1)
    y_test = test_data["RUL"]

    return X_test, y_test

def train_xgb_model(X_train, y_train):
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        tree_method="hist"
    )
    xgb_model.fit(X_train, y_train)
    model_save = open("model_xgboost", "wb")
    dump(xgb_model, model_save, protocol=5)
    return xgb_model

def test_xgb_model(model, X_test, y_test):
    # Predict RUL
    y_pred = model.predict(X_test)

    # Create a new DataFrame with Actual RUL and Predicted RUL
    results = pd.DataFrame({
        "Actual_RUL": y_test.values,
        "Predicted_RUL": y_pred
    })

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = model.score(X_test, y_test)

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R^2: {r2:.2f}")

    # Save the results to a CSV file
    results.to_csv("predicted_vs_actual.csv", index=False)

    return mae, rmse, r2, results

def get_single_prediction(test_data_file):
    column_names = [
        "EngineID", "Cycle", "Op1", "Op2", "Op3",
        "Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5",
        "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10",
        "Sensor11", "Sensor12", "Sensor13", "Sensor14", "Sensor15",
        "Sensor16", "Sensor17", "Sensor18", "Sensor19", "Sensor20",
        "Sensor21"
        ]
    if type(test_data_file) == str:
        test_data = pd.read_csv("data/"+test_data_file, sep='\s+',header=None, names=column_names)
        model_save = open("model_xgboost", "rb")
    else:
        test_data = pd.read_csv(test_data_file, sep='\s+',header=None, names=column_names)
        model_save = open("models/model_xgboost", "rb")
    test_data = test_data.drop(["EngineID", "Cycle", "Sensor14", "Sensor11"], axis=1)
    model = load(model_save)
    rul_est = model.predict(test_data)
    test_data["RULPrediction"] = rul_est
    return(int(rul_est[0]))

# def test_model():
X_train, y_train = load_train_data(train_files)
X_test, y_test = load_test_data(test_files, rul_files)
model = train_xgb_model(X_train, y_train)
mae, rmse, r2, results = test_xgb_model(model, X_test, y_test)
