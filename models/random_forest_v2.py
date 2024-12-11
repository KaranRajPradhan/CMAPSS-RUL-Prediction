import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

class CMAPSSRULPredictor:
    def __init__(self, dataset_path='./CMAPSSData'):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=20, 
            min_samples_split=5
        )

    def load_data(self, dataset_id='FD001'):
        # Load training data
        train_file = os.path.join(self.dataset_path, f'train_{dataset_id}.txt')
        train_columns = [
            'unit_number', 'cycle', 'setting1', 'setting2', 'setting3',
            *[f'sensor_{i}' for i in range(1, 22)]
        ]
        self.train_data = pd.read_csv(train_file, sep='\s+', header=None, names=train_columns)

        # Load test data
        test_file = os.path.join(self.dataset_path, f'test_{dataset_id}.txt')
        self.test_data = pd.read_csv(test_file, sep='\s+', header=None, names=train_columns)

        # Load RUL data
        rul_file = os.path.join(self.dataset_path, f'RUL_{dataset_id}.txt')
        self.test_rul = pd.read_csv(rul_file, header=None)[0].values

    def preprocess_data(self):
        # Calculate RUL for training data
        def calculate_rul(group):
            max_cycle = group['cycle'].max()
            group['RUL'] = max_cycle - group['cycle']
            return group

        self.train_data = self.train_data.groupby('unit_number').apply(calculate_rul).reset_index(drop=True)
        
        # Select features (excluding unit number, cycle)
        feature_columns = [
            'setting1', 'setting2', 'setting3', 
            *[f'sensor_{i}' for i in range(1, 22)]
        ]
        
        # Prepare features and target
        X = self.train_data[feature_columns]
        y = self.train_data['RUL']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def train_model(self):
        X, y = self.preprocess_data()
        self.model.fit(X, y)

    def predict_rul(self):
        test_last_cycles = self.test_data.groupby('unit_number').last().reset_index()
        
        test_features = test_last_cycles[
            ['setting1', 'setting2', 'setting3', 
             *[f'sensor_{i}' for i in range(1, 22)]]
        ]
        
        # Scale test features
        X_test_scaled = self.scaler.transform(test_features)
        
        # Predict RUL
        predicted_rul = self.model.predict(X_test_scaled)
        return predicted_rul

    def evaluate_model(self, predicted_rul):
        mae = mean_absolute_error(self.test_rul, predicted_rul)
        mse = mean_squared_error(self.test_rul, predicted_rul)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.test_rul, predicted_rul)
        
        return {
            'Mean Absolute Error': mae,
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'R-squared': r2
        }

def main():
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    
    for dataset in datasets:
        print(f"\nAnalyzing Dataset: {dataset}")
        predictor = CMAPSSRULPredictor()
        predictor.load_data(dataset)
        predictor.train_model()
        
        # Predict and evaluate
        predicted_rul = predictor.predict_rul()
        metrics = predictor.evaluate_model(predicted_rul)
        
        print("Model Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
