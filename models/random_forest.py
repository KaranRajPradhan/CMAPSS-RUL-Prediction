import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load Dataset
train_path = "CMAPSSData/train_FD001.txt"
test_path = "CMAPSSData/test_FD001.txt"
rul_path = "CMAPSSData/RUL_FD001.txt"

# Column names
columns = [
    "EngineID", "Cycle", "Op1", "Op2", "Op3",
    "Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5",
    "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10",
    "Sensor11", "Sensor12", "Sensor13", "Sensor14", "Sensor15",
    "Sensor16", "Sensor17", "Sensor18", "Sensor19", "Sensor20",
    "Sensor21"
]

# Load data with automatic whitespace handling
train_data = pd.read_csv(train_path, delim_whitespace=True, header=None, names=columns)
test_data = pd.read_csv(test_path, delim_whitespace=True, header=None, names=columns)
rul_data = pd.read_csv(rul_path, header=None, names=['RUL'])

# Add Remaining Useful Life (RUL) to training data
train_data['RUL'] = train_data.groupby('EngineID')['Cycle'].transform("max") - train_data['Cycle']

# Feature Engineering
def add_features(df):
    """Add rolling statistics and normalized features."""
    sensor_cols = [col for col in df.columns if 'Sensor' in col]
    df[sensor_cols] = df[sensor_cols].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return df

train_data = add_features(train_data)
test_data = add_features(test_data)

# Prepare Features and Target
train_data = train_data.drop(columns=['EngineID', 'Cycle'])
test_data = test_data.drop(columns=['EngineID', 'Cycle'])
feature_cols = [col for col in train_data.columns if col != 'RUL']
X_train = train_data[feature_cols]
y_train = train_data['RUL']

# Map RUL to each test data point
def map_test_rul(test_df, rul_df):
    test_df = test_df.copy()
    engine_rul_map = rul_df['RUL'].to_dict()
    test_df['RUL'] = test_df['EngineID'].map(lambda x: engine_rul_map[x - 1])
    return test_df

test_data['EngineID'] = test_data.index // len(test_data) + 1  # Restore EngineID
mapped_test_data = map_test_rul(test_data, rul_data)
y_test = mapped_test_data['RUL']

X_test = test_data[feature_cols]

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate on Test Set
y_test_pred = rf.predict(X_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")
