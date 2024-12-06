import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from scipy.stats import pearsonr
import seaborn as sns
import statsmodels.api as sm
from time import time
from pickle import dump, load

def prepare_csv(filename):    
    column_names = [
    "EngineID", "Cycle", "Op1", "Op2", "Op3",
    "Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5",
    "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10",
    "Sensor11", "Sensor12", "Sensor13", "Sensor14", "Sensor15",
    "Sensor16", "Sensor17", "Sensor18", "Sensor19", "Sensor20",
    "Sensor21"
    ]

    # Load the training data
    train_df1 = pd.read_csv("data/"+filename, sep='\s+', header=None, names=column_names)
    
    failure_data = train_df1.loc[train_df1.groupby('EngineID')['Cycle'].idxmax()]
    train_df1["CyclesToFailure"] = train_df1.apply(lambda row: failure_data["Cycle"].loc[failure_data["EngineID"] == row.EngineID].values[0]-row.Cycle, axis=1)
    
    train_df1.to_csv("data/"+filename, index=False)
    
def generate_model(training_data):
    train_df1 = pd.read_csv("data/"+training_data)
    
    Y = train_df1["CyclesToFailure"]
    X = train_df1.drop("CyclesToFailure", axis=1)
    X = sm.add_constant(X)
    model_lin = sm.OLS(Y,X)
    model_lin_fit = model_lin.fit()
    print(model_lin_fit.summary())
    model_save = open("model_lin_fit.pkl", "wb")
    dump(model_lin_fit, model_save, protocol=5)
    
def generate_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    print(df.head())
    return(3)

def prediction_test():
    model_save = open("model_lin_fit.pkl", "rb")
    model_lin_fit = load(model_save)
    print(model_lin_fit.summary())
    