# Light-SERNet

This is the Tensorflow 2.x implementation of our paper ["Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition"](https://arxiv.org/abs/2110.03435), accepted in ICASSP 2022. 

<div align=center>
<img width=95% src="./pics/Architecture.png"/>
</div>
In this paper, we propose an efficient and lightweight fully convolutional neural network(FCNN) for speech emotion recognition in systems with limited hardware resources. In the proposed FCNN model, various feature maps are extracted via three parallel paths with different filter sizes. This helps deep convolution blocks to extract high-level features, while ensuring sufficient separability. The extracted features are used to classify the emotion of the input speech segment. While our model has a smaller size than that of the state-of-the-art models, it achieves a higher performance on the IEMOCAP and EMO-DB datasets.

* * *

# Training and Testing

## Demo
Demo on EMO-DB dataset: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AryaAftab/LIGHT-SERNET/blob/master/Demo_Light_SERNet.ipynb)


## Run
### 1. Clone Repository
```bash
$ git clone https://github.com/AryaAftab/LIGHT-SERNET.git
$ cd LIGHT-SERNET/
```
### 2. Requirements
- Tensorflow >= 2.3.0
- Numpy >= 1.19.2
- Tqdm >= 4.50.2
- Matplotlib> = 3.3.1
- Scikit-learn >= 0.23.2

```bash
$ pip install -r requirements.txt
```

### 3. Data:
* Download **[EMO-DB](http://emodb.bilderbar.info/download/download.zip)** and **[IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)**(requires permission to access) datasets
* extract them in [data](./data) folder

### 4. Set hyperparameters and training config :
You only need to change the constants in the [hyperparameters.py](./hyperparameters.py) to set the hyperparameters and the training config.

### 6. Strat training:
Use the following code to train the model on the desired dataset, cost function, and input length(second).
- Note 1: The input is automatically cut or padded to the desired size and stored in the [data](./data) folder.
- Note 2: The best model are saved in the [result](./result) folder.
- Note 3: The results for the confusion matrix are saved in the [result](./result) folder.
```bash
$ python train.py -dn {dataset_name} \
                  -id {input durations} \
                  -at {audio_type} \
                  -ln {cost function name} \
                  -v {verbose for training bar} \
                  -it {type of input(mfcc, spectrogram, mel_spectrogram)}
```
#### Example:

EMO-DB Dataset:
```bash
python train.py -dn "EMO-DB" \
                -id 3 \
                -at "all" \
                -ln "focal" \
                -v 1 \
                -it "mfcc"
```

IEMOCAP Dataset:
```bash
python train.py -dn "IEMOCAP" \
                -id 7 \
                -at "impro" \
                -ln "cross_entropy" \
                -v 1 \
                -it "mfcc"
```
**Note : For all experiments just run ```run.sh```**
```bash
sh run.sh
```

## Citation

If you find our code useful for your research, please consider citing:
```bibtex
@article{aftab2021light,
  title={Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition},
  author={Aftab, Arya and Morsali, Alireza and Ghaemmaghami, Shahrokh and Champagne, Benoit},
  journal={arXiv preprint arXiv:2110.03435},
  year={2021}
}
```

* * *

# Single File Inference

In the folder *[inference_tests](https://github.com/PanK0/LIGHT-SERNET/tree/master/inference_tests)*  are present:
- **trained model** (named: *EMO-DB_3.0s_Segmented_cross_entropy_float32*) with the relative confusion matrix and report for the performances. The model is trained using the EMO-DB dataset;
- some **audio samples** named with the corresponding label.


## Content of LIGHT-SERNET/inference_tests folder

Except for the four files named


*   `EMO-DB_3.0s_Segmented_cross_entropy_float32.tflite`
*   `EMO-DB_3.0s_Segmented_cross_entropy_Report.txt`
*   `EMO-DB_3.0s_Segmented_cross_entropy_TotalConfusionMatrixNormalized.pdf`
*   `EMO-DB_3.0s_Segmented_cross_entropy_TotalConfusionMatrix.pdf`

the other files in the folder contain in the name some information like


*   The **label** of the audio file
*   Whether the file **belongs to the EMO-DB Dataset** (simply named with `dataset`)
*   Whether the file is a **phrase from the dataset** but it has been recorded by me from a german speaker (simply named with `rec`)
*   Whether the file is **NOT a phrase from the dataset** and it has been recorded by me from a german speaker (simply named with `external`)

**! ! ! PLEASE CAREFUL :** audio tagged with `rec` and `external` are recorded by me with non-professional instruments with two german speakers that are **NOT ACTRESSES**, so the classification of the content may encounter some impediments.

## Run Inference Experiments

Colab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ESVAMto8Fu65l2uSakStkt6OgQfbc6VH?usp=sharing)

The main functions used for the experiments, took from the original source and modified a little for my scopes, are contained into the file *[inference_data_processing.py](https://github.com/PanK0/LIGHT-SERNET/blob/master/inference_data_processing.py)*

- Install the requirements

- Run the file *[inference_single_file.py](https://github.com/PanK0/LIGHT-SERNET/blob/master/inference_single_file.py)* with

`$ python inference_single_file.py -id 3 -at "all" -ln "cross_entropy" -it "mfcc" -fn "happiness_dataset.wav"`

where "happiness_dataset.wav" is the name of the file you want to classify, stored in the folder *[inference_tests](https://github.com/PanK0/LIGHT-SERNET/tree/master/inference_tests)*;
- **CAREFUL** : the file must be a **.wav** file sampled at **16 kHz**, otherwise the program will not work. 


## The used model

The used model is also present in the folder *[inference_tests](https://github.com/PanK0/LIGHT-SERNET/tree/master/inference_tests)* with the name `EMO-DB_3.0s_Segmented_cross_entropy_float32.tflite`. 

To change the model train again the Neural Network and then modify the model path in the variable `model_path` in the code in `inference_single_file.py`.

**! ! ! PLEASE CAREFUL:** if you train a new model, the classes will be mixed, and so you'll have to reassign the correct correspondend classes in the dictionary.