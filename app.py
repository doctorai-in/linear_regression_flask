import numpy as np
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.externals import joblib
from house_price_prediction import house_price

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
model = joblib.load("my_model.pkl")

@app.route('/')
def home():
    return render_template('index_upload.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    folder_name = "files"#request.form['upload']
    # this is to verify that folder to upload to exists.
    if os.path.isdir(os.path.join(APP_ROOT, '{}/'.format(folder_name))):
        print("folder exist")
    
    target = os.path.join(APP_ROOT, '{}/'.format(folder_name))
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".csv") or (ext == ".csv"):
            print("File supported moving on...")
            destination = "/".join([target, filename])
            print("Accept incoming file:", filename)
            print("Save it to:", destination)
            upload.save(destination)
        else:
            print("Files uploaded are not supported...")
        

    test = 'files/input.csv'
    house_price(test, model)
    return render_template('output.html')



if __name__ == "__main__":
    app.run(debug=True, port=90)
