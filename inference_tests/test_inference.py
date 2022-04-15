import os
import datetime
import time
import argparse

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from dataio import *
from callbacks import *
from model_saver import *
from loss import *
from tflite_evaluate import *
import hyperparameters
import models


import warnings
warnings.filterwarnings("ignore")

# build the argument parser to get the inputs from command line
ap = argparse.ArgumentParser()
ap.add_argument("-tf", "--tflite_file",
                required=True,
                type=str,
                help="model, .tflite file")

ap.add_argument("-f", "--file_name",
                required=True,
                type=str,
                help="name of the file to predict")

tflite_file = args["tflite_file"]
file_name = []
file_name.append(args["file_name"])
file_name = tf.concat(file_name, axis=0).numpy()
input_type = "mfcc"

# input preprocessing
processed_audio = preprocess_input(file_name, input_type)

# run the prediction
predictions = run_tflite_model(tflite_file, processed_audio)

print(predictions)