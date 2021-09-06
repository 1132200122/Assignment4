from flask import Flask,render_template,request
import Logistic
import numpy as np


app = Flask(__name__)

@app.route('/')
def load():
    Logistic.train()
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predictions():
    a = request.form['slength']
    b = request.form['swidth']
    c = request.form['plength']
    d = request.form['pwidth']
    value = Logistic.prediction(a,b,c,d)
    return render_template('index.html', value='Predicted Class: {}'.format(value))
if __name__=='__main__':
    app.run(debug=True,port=4000)