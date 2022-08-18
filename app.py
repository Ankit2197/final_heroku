import numpy as np
from flask import Flask, request,render_template
import stockmodel

app = Flask(__name__,template_folder='template')

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    final_features = int_features[0]
    prediction = stockmodel.stockpred(final_features)

    output = prediction

    return render_template('index.html', prediction_text='Predicted Stock price for next day is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)