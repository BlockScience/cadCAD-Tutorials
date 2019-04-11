from flask import Flask,request,jsonify,Response,render_template
from functools import wraps
from werkzeug import secure_filename
import os
import re


app = Flask(__name__)
app.secret_key = 'mysecretkey'


def run_cadCAD(input1,input2,input3):
    '''function for text summarization'''
    # bytes to string

    return Summary

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/health")
def health():
    return 'Healthy'

@app.route('/result', methods=['POST'])
def result():
    '''
    Route for UI
    '''

    try:
        file = request.files['upload']
        # Replace spaces with underscores to prevent errors
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        text = textract.process('data/' + filename)
        print(text)
    except:
        # Get the data from the index page
        text=request.form['textToSummarize']

 
    return render_template('result.html', Summary=Summary)

@app.route('/run', methods=['POST'])
@requires_auth
def summary():
    '''
    Example API call:
        Curl:

        curl -X POST -H "Content-Type: application/json"  -u "username:password" -d '{
        "text": "Handling file upload in Flask is very easy. It needs an HTML form with its enctype attribute set to ‘multipart/form-data’, posting the file to a URL. The URL handler fetches file from request.files[] object and saves it to the desired location. Each uploaded file is first saved in a temporary location on the server, before it is actually saved to its ultimate location. Name of destination file can be hard-coded or can be obtained from filename property of request.files[file] object. However, it is recommended to obtain a secure version of it using the secure_filename() function. It is possible to define the path of default upload folder and maximum size of uploaded file in configuration settings of Flask object.",
        "ratio": ".5"
        }' http://localhost:80/summarize-text

        
        Python:
        import requests

        headers = {
            'Content-Type': 'application/json',
            }

        data = '{"text": "Linear model fitted by minimizing a regularized empirical loss with SGD SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). The regularizer is a penalty added to the loss function that shrinks model parameters towards the zero vector using either the squared euclidean norm L2 or the absolute norm L1 or a combination of both (Elastic Net). If the parameter update crosses the 0.0 value because of the regularizer, the update is truncated to 0.0 to allow for learning sparse models and achieve online feature selection. This implementation works with data represented as dense numpy arrays of floating point values for the features.","ratio": ".90"}'

        response = requests.post('http://localhost:80/summarize-text', headers=headers, data=data,auth=('user', 'password')) 
        response.json()[0]['Summary']

    '''

    # Get the data from the upload
    text = request.get_json()['text']
    ratio = request.get_json()['ratio']
    print(text)
    print(ratio)
    ratio = float(ratio)
    # If user puts in 25 for .25, convert
    if ratio > 1:
        ratio = ratio/100
    else:
        pass

    resultSummary = [{'Summary':Summary} ]
    return jsonify(resultSummary)

@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('error.html'), 404

@app.errorhandler(405)
def page_not_found2(e):
    # note that we set the 405 status explicitly
    return render_template('error.html'), 405

@app.errorhandler(500)
def page_not_found3(e):
    # note that we set the 500 status explicitly
    return render_template('error.html'), 500

if __name__ == "__main__":
     app.run(host='localhost', port=8000,debug=True)
