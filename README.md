# A-multimodal-model-for-detecting-depression-using-text-and-speech-data
This project proposes a multi-scale spatial and temporal convolution fusion method that combines text and speech modal data for depression recognition.
Code running process：
#### 1. Prepare vocal data  ####
preprocess data(CMDC_preprocess.py)
#### 2. Prepare text data  ####
txt data split(txt_split)
#### 3. Train and Evaluate  ####
run.py


#### Requirement: ####
scikit-learn	1.0.2
numba	0.48.0
librosa	0.7.2
torch	1.12.1+cu116
transformers	4.18.0
