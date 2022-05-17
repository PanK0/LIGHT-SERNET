from dataio import *

"""
Segment the file in order to obtain a segmente with a fixed length
"""
def segment_file(filename, segment_length=3, segment_mode=1):
    raw_data, fs = read_wave(filename)
    trimed_data = trim_wave(raw_data, segment_length=int(segment_length*fs), segment_mode=segment_mode)

    for counter in range(0, len(trimed_data), int(segment_length*fs)):
        segment_data = trimed_data[counter:counter+int(segment_length*fs)]
        segment_data = normalize(segment_data, segment_length=int(segment_length*fs))

        splited_filename = filename.split("/")
        splited_filename[-1] = f"segmented_{splited_filename[-1]}"
        filename = "/".join(splited_filename)
        write_wave(segment_data, filename, fs)
    return filename

def get_input_id(audio, input_type="mfcc") :
    if input_type == "spectrogram":
        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        return spectrogram
    elif input_type == "mel_spectrogram":
        spectrogram = get_spectrogram(audio)
        mel_spectrogram = get_mel_spectrogram(spectrogram)
        mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)
        return mel_spectrogram
    elif input_type == "mfcc":
        spectrogram = get_spectrogram(audio)
        mel_spectrogram = get_mel_spectrogram(spectrogram)
        mfcc = get_mfcc(mel_spectrogram)
        mfcc = tf.expand_dims(mfcc, -1)
        return mfcc
    else:
        raise ValueError('input_type not Valid!')


'''
Preprocess the input
'''
# Get the preprocessed input as result
def preprocess_input(filename, input_type="mfcc") :
    output_file, _ = get_waveform_and_label(filename)
    output_file = get_input_id(output_file)
    return output_file 


'''
Load and run the model
'''
def run_model(tflite_file, test_audio) :

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros(1, dtype=int)

    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        test_audio = test_audio / input_scale + input_zero_point

    test_audio = np.expand_dims(test_audio, axis=0).astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], test_audio)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions = output.argmax()
    
    return predictions