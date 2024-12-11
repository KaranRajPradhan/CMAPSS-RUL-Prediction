import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from time import time
from pickle import dump, load
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

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
    
def generate_model(training_data, model_name):
    train_df1 = pd.read_csv("data/"+training_data)
    train_df1 = train_df1.drop(["EngineID","Op1", "Op2", "Op3","Sensor5","Sensor6","Sensor10","Sensor16","Sensor18","Sensor19"], axis=1)
    Y = train_df1["CyclesToFailure"]
    X = train_df1.drop("CyclesToFailure", axis=1)
    model_lin = sm.OLS(Y,X)
    model_lin_fit = model_lin.fit()
    print(model_lin_fit.summary())
    model_save = open(model_name, "wb")
    dump(model_lin_fit, model_save, protocol=5)
    
def generate_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    print(df.head())
    return(3)

def prediction_test(test_data_file, model, do_plot = 0, const_adj = 0):
    column_names = [
    "EngineID", "Cycle", "Op1", "Op2", "Op3",
    "Sensor1", "Sensor2", "Sensor3", "Sensor4", "Sensor5",
    "Sensor6", "Sensor7", "Sensor8", "Sensor9", "Sensor10",
    "Sensor11", "Sensor12", "Sensor13", "Sensor14", "Sensor15",
    "Sensor16", "Sensor17", "Sensor18", "Sensor19", "Sensor20",
    "Sensor21"
    ]
    test_data = pd.read_csv("data/"+test_data_file, sep='\s+',header=None, names=column_names)
    test_data_no_id = test_data.drop(["EngineID","Op1", "Op2", "Op3","Sensor5","Sensor6","Sensor10","Sensor16","Sensor18","Sensor19"], axis=1)
    model_save = open(model, "rb")
    model_lin_fit = load(model_save)
    rul_est = model_lin_fit.predict(test_data_no_id)
    test_data["RULPrediction"] = rul_est
    test_data["RULPrediction"] = test_data["RULPrediction"].astype(int)
    if const_adj == 1:
        test_data["RULPrediction"] = test_data["RULPrediction"].sub(125) #just a guess, could be optimized later
    failure_data = test_data.loc[test_data.groupby('EngineID')['Cycle'].idxmax()]
    test_data["CyclesToFailure"] = test_data.apply(lambda row: failure_data["Cycle"].loc[failure_data["EngineID"] == row.EngineID].values[0]-row.Cycle, axis=1)
    test_data = test_data[test_data["CyclesToFailure"] > 0]
    test_data["Error"] = test_data["CyclesToFailure"].sub(test_data["RULPrediction"])
    test_data["PercentError"] = test_data["Error"].div(test_data["CyclesToFailure"])
    MAE = mean_absolute_error(test_data["CyclesToFailure"], test_data["RULPrediction"])
    MAPE = mean_absolute_percentage_error(test_data["CyclesToFailure"], test_data["RULPrediction"])
    if do_plot == 1:
        plt.figure(figsize=(10,4))
        plt.plot(test_data.loc[test_data["EngineID"]==69]["CyclesToFailure"])
        plt.plot(test_data.loc[test_data["EngineID"]==69]["RULPrediction"])
        plt.legend(("RUL (Actual)","RUL (Predicted)"), fontsize = 12)
        plt.show()
    print(MAE,MAPE)
    return (MAE,MAPE)
    
def test_models():
    models = ["model_lin_fit_01.pkl","model_lin_fit_02.pkl","model_lin_fit_03.pkl","model_lin_fit_04.pkl"]
    test_datasets = ["test_FD001.txt","test_FD002.txt","test_FD003.txt","test_FD004.txt"]
    results = []
    for i in models:
        for j in test_datasets:
            results = results + [(prediction_test(j, i,0,1))]
    avg_MAE = 0
    avg_MAPE = 0
    for i in results:
        avg_MAE = avg_MAE + i[0]
        avg_MAPE = avg_MAPE + i[1]
    avg_MAE = avg_MAE / len(results)
    avg_MAPE = avg_MAPE / len(results)
    return(avg_MAE, avg_MAPE)
    
def get_single_prediction(test_data_file, model):
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
        model_save = open(model, "rb")
    else:
        test_data = pd.read_csv(test_data_file, sep='\s+',header=None, names=column_names)
        model_save = open("models/"+model, "rb")
    test_data = test_data.drop(["EngineID","Op1", "Op2", "Op3","Sensor5","Sensor6","Sensor10","Sensor16","Sensor18","Sensor19"], axis=1)
    model_lin_fit = load(model_save)
    rul_est = model_lin_fit.predict(test_data)
    test_data["RULPrediction"] = rul_est
    return(int(rul_est.iloc[0]))