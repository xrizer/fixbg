from testing import *
from flask import Flask, jsonify, render_template,request,send_file,Response
import os

model1, model2, model3 = load_models()

app = Flask(__name__)
@app.route('/',methods=['GET'])
def hello_world():
    return jsonify({'test':'berhasil'})


@app.route('/predict',methods=['POST'])
def coba():
    imagefile = request.files['']
    img_path = 'tmp/' + imagefile.filename
    imagefile.save(img_path)
    result, peluang, count = main(img_path, model1, model2, model3)
    hasil = report(result, peluang, count)
    return jsonify({'diagnose': hasil})

if __name__ == '__main__':
    app.run(debug=True)