# run with this:
# python single_test_inference.py -fn "11a04Fd.wav" -id 3 -at "all" -ln "cross_entropy" -v 0 -it "mfcc"



from fileinput import filename
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


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.random.set_seed(hyperparameters.SEED)
np.random.seed(hyperparameters.SEED)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-fn", "--file_name",
                required=True,
                type=str,
                help="file name")

ap.add_argument("-id", "--input_durations",
                required=True,
                type=float,
                help="input durations(sec)")

ap.add_argument("-at", "--audio_type",
                default="all",
                type=str,
                help="auido type to filter dataset(IEMOCAP)")

ap.add_argument("-ln", "--loss_name",
                default="cross_entropy",
                type=str,
                help="cost function name for training")

ap.add_argument("-v", "--verbose",
                default=1,
                type=int,
                help="verbose for training bar")

ap.add_argument("-it", "--input_type",
                default="mfcc",
                type=str,
                help="type of input(mfcc, spectrogram, mel_spectrogram)")


args = vars(ap.parse_args())


file_name = args["file_name"]  # name of the file inside inference_tests/
input_durations = args["input_durations"]
audio_type = args["audio_type"]
loss_name = args["loss_name"]
verbose = args["verbose"]
input_type = args["input_type"]

print ("\n***** OBTAINING FULL FILE PATH *****")
filename = f"inference_tests/{file_name}"
print (filename)

print ("\n***** INPUT PREPROCESSING *****")
preprocessed_input = preprocess_input(filename, input_type="mfcc")
print (preprocessed_input)

'''
print ("\n***** CREATING BUFFX *****")
BuffX = []
BuffX.append(preprocessed_input)
BuffX = tf.concat(BuffX, axis=0).numpy()
print (BuffX)
'''

print ("\n***** LOAD AND RUN THE MODEL *****")
model_path = f"inference_tests/EMO-DB_3.0s_Segmented_cross_entropy_float32.tflite"
predictions = run_model(model_path, preprocessed_input)
print (predictions)

