# predict.py
import argparse
import sys
import os
import glob
import numpy as np
import urllib
import cv2
 
from keras.models import load_model as load_keras_model
from keras.preprocessing.image import img_to_array, load_img
from flask import Flask, jsonify

app = Flask(__name__)
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
 
model_filename = 'cntk-model.h5'
class_to_name = [
    "agave blue",
    "aztec gold sunburst",
    "aztec gold sparkle",
    "black",
    "black sunburst",
    "blue sparkle",
    "burgundy mist",
    "candy apple red",
    "candy green",
    "cherry burst",
    "cherry sunburst",
    "coral pink",
    "daphne blue",
    "desert sand",
    "fiesta red",
    "lake placid blue",
    "ocean turquoise",
    "olympic white",
    "sage green metallic",
    "sea foam green",
    "sea foam green sparkle",
    "vintage white",
    "vintage blonde",
    "amber",
    "antigua",
    "antique burst"
]

def load_model():
    if os.path.exists(model_filename):
        return load_keras_model(model_filename)
    else:
        print("File {} not found!".format(model_filename))
        exit()

def load_image(filename):
    img_arr = img_to_array(load_img(filename, False, target_size=(256,256)))
    return np.asarray([img_arr])

@app.route('/predict')
def predict():
    result = np.argmax(keras_model.predict(image))
    return jsonify({'prediction': class_to_name[result]})

 
if __name__ == '__main__':
    filename = sys.argv[1]
    keras_model = load_model()
    image = load_image(filename)
    app.run(host='0.0.0.0', port=5000)
