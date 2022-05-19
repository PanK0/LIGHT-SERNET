# run with this:
# python inference_single_file.py -id 3 -at "all" -ln "cross_entropy" -it "mfcc" -fn "happiness_dataset.wav"



from fileinput import filename
import argparse

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from dataio import *
from inference_data_processing import *
from callbacks import *
from model_saver import *
from loss import *
from tflite_evaluate import *
import hyperparameters


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

ap.add_argument("-it", "--input_type",
                default="mfcc",
                type=str,
                help="type of input(mfcc, spectrogram, mel_spectrogram)")


args = vars(ap.parse_args())


file_name = args["file_name"]  # name of the file inside inference_tests/
input_duration = int(args["input_durations"])
audio_type = args["audio_type"]
loss_name = args["loss_name"]
input_type = args["input_type"]

# print ("\n***** OBTAINING FULL FILE PATH *****")
filename = f"inference_tests/{file_name}"

# print ("\n***** SEGMENTING DATA *****")
segmented_file = segment_file(filename, input_duration, segment_mode=1)

# print ("\n***** INPUT PREPROCESSING *****")
preprocessed_input = preprocess_input(segmented_file, input_type)

# print ("\n***** LOAD AND RUN THE MODEL *****")
model_path = f"inference_tests/EMO-DB_3.0s_Segmented_cross_entropy_float32.tflite"
predictions = run_model(model_path, preprocessed_input)

# Classes - only valid for this specific model
# !!! IF YOU CHANGE MODEL, YOU ALSO HAVE TO REASSIGN THE CLASSES, DUE TO THE MODEL TRAINING
classdict  = {
    0 : "Boredom",
    1 : "Neutral",
    2 : "Sadness",
    3 : "Anxiety/Fear",
    4 : "Anger",
    5 : "Disgust",
    6 : "Happiness"
}

print ("\n****** RESULT ******")
print (classdict[predictions])
