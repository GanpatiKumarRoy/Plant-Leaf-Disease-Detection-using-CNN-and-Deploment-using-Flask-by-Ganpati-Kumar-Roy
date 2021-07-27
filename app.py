from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='Model__Roy.h5'


# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    
    if preds==0:
        preds="Apple Disease_Black_Root"
    elif preds==1:
        preds="Apple Disease_Rust"
    elif preds==2:
        preds="Apple_Healthy"
    elif preds==3:
        preds="Cherry_Disease"
    elif preds==4:
        preds="Cherry_Healthy"
    elif preds==5:
        preds="Maize or Sugercane_Disease_Blight"
    elif preds==6:
        preds="Maize_Disease_Gray_Spot"
    elif preds==7:
        preds="Maize_Healthy"
    elif preds==8:
        preds="Peach_Disease_Bacerial"
    elif preds==9:
        preds="Peach_Healthy"
    elif preds==10:
        preds="Potato_Disease_Early_Blight"
    elif preds==11:
        preds="Potato_Disease_Late_Blight"
    elif preds==12:
        preds="Potato_Healthy"
    elif preds==13:
        preds="Tomato_Disease_Blight"
    elif preds==14:
        preds="Tomato_Disease_Target_Spot"
    elif preds==15:
        preds="Tomato_Disease_Yellow_Virus"
    else:
        preds="Tomato_Healthy"


    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=8080)
