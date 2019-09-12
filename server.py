import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 14)
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    result = loaded_model.predict(to_predict)
    return result

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = ValuePredictor(to_predict_list)
    return render_template("result.html", prediction=result)