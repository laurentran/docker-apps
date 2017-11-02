# predict.py
import argparse
import sys
import os
import glob
import numpy as np
 
from keras.models import load_model as load_keras_model
from keras.preprocessing.image import img_to_array, load_img
from flask import Flask, jsonify

app = Flask(__name__)
 
# disable TF debugging info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
 
# our saved model file
# may be refactored to be taken from command line
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
 
def get_filenames():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='*', default=['**/*.*'])
    args = parser.parse_args()
    print (args, args.filename)
    return args.filename
 
    # for pattern in args.filename:
    #     # here we recursively look for input
    #     # files using provided glob patterns
    #     for filename in glob.iglob('data/' + pattern, recursive=True):
    #         yield filename
 
 
def load_model():
    if os.path.exists(model_filename):
        return load_keras_model(model_filename)
    else:
        print("File {} not found!".format(model_filename))
        exit()
 
 
def load_image(filename):
    img_arr = img_to_array(load_img(filename, False, target_size=(256,256)))
    return np.asarray([img_arr])
 
#@app.route('/predict/<path:url>', methods=['POST'])
@app.route('/')
#def predict(image, model):
def predict():
    image_class = predict(image, keras_model)
    result = np.argmax(model.predict(image))
    #return class_to_name[result]
    return jsonify({'prediction': class_to_name[result]})
 
 
if __name__ == '__main__':
    filenames = get_filenames()
    keras_model = load_model()
    for filename in filenames:
        image = load_image(filename)
        #image_class = predict(image, keras_model)
        #print("{:30}   {}".format(filename, image_class))

    app.run(debug=True)