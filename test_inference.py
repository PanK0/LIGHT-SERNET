mport os
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
ap.add_argument("-dn", "--dataset_name",
                required=True,
                type=str,
                help="dataset name")

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


dataset_name = args["dataset_name"]  # Use EMO-DB_3.0s_Inference_Tests as dataset name
dataset_name = "EMO-DB_3.0s_Inference_Tests"
input_durations = args["input_durations"]
audio_type = args["audio_type"]
loss_name = args["loss_name"]
verbose = args["verbose"]
input_type = args["input_type"]

'''
NO NEED TO SEGMENT DATASET
'''
print ("Dataset is already segmented\n")

threshold = 0

Result = []
Reports = []
Predicted_targets = np.array([])
Actual_targets = np.array([])

print ("Preparing for dataset spliting")

Filenames, Splited_Index, Labels_list = split_dataset(dataset_name, audio_type=audio_type)

print ("\n***** FILENAMES *****")
print (Filenames)

print ("\n***** SPLITED INDEX *****")
print (Splited_Index)

print ("\n***** LABELS LIST *****")
print (Labels_list)