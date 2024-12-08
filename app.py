from flask import Flask, redirect, url_for, render_template, request, flash
from markupsafe import escape
from models import linear_regression


app = Flask(__name__)
app.secret_key = "SECRET KEY"

@app.route("/index.html", methods = ["POST","GET"])
def index():
    if request.method == "GET": #first loading the page
        rul_est_lin = "" #leave results box empty
        rul_est_2 = ""
        rul_est_3 = ""
        return render_template('index.html', rul_est_lin=rul_est_lin,rul_est_2=rul_est_2,rul_est_3=rul_est_3)
    elif request.method == "POST":
        file_contents = request.files["file_upload"].read().decode("utf-8") # file_contents contains the string read from the file. This can be fed into a pandas dataframe
        if len(file_contents) == 0:
            rul_est_lin = "Please upload a data file" # Checks for a non-empty file upload. Does not check for correct csv formatting
            rul_est_2 = ""
            rul_est_3 = ""
        else:
            data_file = request.files["file_upload"]
            #calculate estimates from different models
            data_file.seek(0) #linear
            rul_est_lin_01 = str(linear_regression.get_single_prediction(data_file,"model_lin_fit_01.pkl"))
            data_file.seek(0)
            rul_est_lin_02 = str(linear_regression.get_single_prediction(data_file,"model_lin_fit_02.pkl"))
            data_file.seek(0)
            rul_est_lin_03 = str(linear_regression.get_single_prediction(data_file,"model_lin_fit_03.pkl"))
            data_file.seek(0)
            rul_est_lin_04 = str(linear_regression.get_single_prediction(data_file,"model_lin_fit_04.pkl"))
            rul_est_lin = rul_est_lin_01 + ", " + rul_est_lin_02 + ", " + rul_est_lin_03 + ", " + rul_est_lin_04 + " Cycles"
            rul_est_2 = str(42) + " Hours"
            rul_est_3 = str(42) + " Hours"
        return render_template('index.html', rul_est_lin=rul_est_lin,rul_est_2=rul_est_2,rul_est_3=rul_est_3)
