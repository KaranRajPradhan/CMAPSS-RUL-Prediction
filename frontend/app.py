from flask import Flask, redirect, url_for, render_template
from markupsafe import escape

app = Flask(__name__)

rul_est = 2

@app.route("/")
def index():
    return render_template('index.html', rul_est=rul_est)

@app.route("/data_input")
def data_input():
    return "Data input page"
