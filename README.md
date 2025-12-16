# smart-home-kws
**EE 475 Machine Learning Project:** An embedded keyword-spotting system that controls peripherals based on voice commands.

- Eric Oliveira, Justin Ansell, Vivek Matta

[YouTube Demo](https://www.youtube.com/watch?v=afolWC9hfCI)

# Overview
This project is a proof of concept for a smart-home voice-controlled system that identifies a selected set of commands and controls peripherals in real time. We deployed the model on a **Raspberry Pi Zero W** housed in a 3D-printed enclosure featuring:

- Microphone  
- Push button (starts an inference-window recording)  
- RGB LED  
- LCD

# Data Pipeline

## Gathering Data
We curated our own dataset by recording ourselves, friends, and family repeatedly uttering **7 voice commands**:

- **LED color control:** `red`, `green`, `blue`, `white`, `off`  
- **LCD display:** `time`, `temperature`

To improve robustness, we added two additional classes:
- `noise` (ambient noise)
- `unknown` (non-command words)

This resulted in **9 total classes**. For the `unknown` class, we recorded speakers saying random words and also incorporated samples from [Qualcomm Keyword Speech Dataset](https://www.qualcomm.com/developer/software/keyword-speech-dataset) *(Copyright (c) 2019 Qualcomm Technologies, Inc. All rights reserved.).* For background noise, we used a Background Noise Dataset available in Kaggle ( [Moaz Abdeljalil. (2025). Background Noise [Data set]. Kaggle.]( https://doi.org/10.34740/KAGGLE/DSV/10892389)

For labelling purposes, we chose the following naming convention: `label_speaker_num_full.wav`, where label is one of our voice commands or "unknown" and "noise", speaker is the name of the speaker in the recording, num is the count of which iteration of that speaker saying that label, and full indicates that this is a full-length recording, not an extracted utterance. Example: `blue_eric_03_ful.wav`. 

## Preprocessing

### Sanitation

Since we asked speakers to record on whatever means they had (mainly cellphone recordins) and mixed this with our own recordings (cellphones and a diverse group of microphones we had available) and outsourced datasets, our file formats, sampling rates, and datatypes were diverse. So the first step in preprocessing our files was to force a standard to all audio recordings. For this end, we wrote the `sanitize_vanilla_dataset()` function inside the `data_augmentation.py` script. This function parses through a directory and rewrites all audio files in place while forcing the following standard:

- Format: .wav
- Bit Depth: 16
- Sample Rate: 22.05 kHz
- Number of Channels: 1 (Mono)

This function also normalizes the audio files. 

### Extracting Utterances

Our speech recordings consisted of speakers repeating a keyword dozens of times, but we needed an efficient way for extracting each utterance for building a useful labelled dataset. To that end, we wrote the `parse()` function in `file_chopper.py`. This file parses a wav file and extracts individual utterances based on a few key parameters that must be set by the user. Since audio recordings varied considerably in the utterance length, signal to noise ratio, and formant shapes, we needed to adapt our extraction settings for each recording. The parameters are `threshold`, `window_length`, and `wait`. The threshold determines the loudness the waveform must reach to be considered a spoken word, the window length is the amount of time after the threshold is reached that will fill in the extracted audio file, and wait is how much time before a threshold is reach will also be added to the extracted utterance (particularly useful for words like "time" and "temperature", where the 't' sound is almost always below reasonable thresholds). The function also handles other housekeeping tasks, such as fade-in and fade-out, normalization, writing to the proper directory, using our naming convention, skipping non-audio files, etc.

### Feature Extraction

Before extracting features, we needed to fix an input length for the model. We chose to use the size of the longest extracted utterance from all of our data plus an increment of 10%. To achieve this, we zero padded all our files (centered). 

We then split all our data into `train`, `val`, and `test` sets, for training, validation, and test datasets. The training data consists of 70% of our data points, while the validation and test sets consist of 15% each. These and many other helpful functions are contained within `wrapper.py`.

We used mel-frequency cepstral coefficients (MFCCs) as our audio feature, extracting them with `librosa`'s implementation. This is essenctially a compressed spectrogram of our audio files based on psychoacoustic approximations. We chose 32 MFCCs, yielding an output format of (32, 154) matrices - 154 is tied to the audio duration (39243 samples). The `extract_mfccs()` function in `feature_extraction.py` is responsible for outputting an MFCC matrix for a single audio file. We also vectorized our labels with the `vectorize_label()` function, which extracts the true label of the audio file based on its path name and maps it to a vector in $R^{9}$. 

The most important function in `feature_extraction.py` is the `build_feature_dict()` function, which parses a directory and, for each extracted utterance (.wav files), generates a tuple `(flat_MFCCs, vec_label)`. Having separated our data into into `train`, `val`, and `test` directories previously, we used this function to build our JSON feature dictionaries, the final form of our datasets that was fed into our model. 

### Data Augmentation

To enhance model robustness, we also artificially expanded our **training** dataset. This expansion consists of transformation to the extracted audio utterances, such as filtering (built filter taps with pyfda [2]), pitch-shifting (librosa's effects), and noise addition. All these functions and the master function `augment_dataset()` are contained in the `data_augmentation.py` script. We then build the `train_augmented` dataset. 

## Model

Our model is a Convolutional Neural Network (CNN) trained on Google Colab with TensorFlow. For details on the training and model architecture, consult our [colab script](https://colab.research.google.com/drive/1zg0SPHg8Gk4uLPPREuo5_4F8gLJBUEz0?authuser=0). 

Here is a model summary:

| Layer (type) | Output Shape | Param # |
|---|---:|---:|
| `input_layer (InputLayer)` | `(None, 32, 154, 1)` | `0` |
| `conv2d (Conv2D)` | `(None, 32, 154, 64)` | `1,088` |
| `batch_normalization (BatchNormalization)` | `(None, 32, 154, 64)` | `256` |
| `max_pooling2d (MaxPooling2D)` | `(None, 16, 77, 64)` | `0` |
| `spatial_dropout2d (SpatialDropout2D)` | `(None, 16, 77, 64)` | `0` |
| `conv2d_1 (Conv2D)` | `(None, 16, 77, 64)` | `36,928` |
| `batch_normalization_1 (BatchNormalization)` | `(None, 16, 77, 64)` | `256` |
| `max_pooling2d_1 (MaxPooling2D)` | `(None, 8, 38, 64)` | `0` |
| `spatial_dropout2d_1 (SpatialDropout2D)` | `(None, 8, 38, 64)` | `0` |
| `conv2d_2 (Conv2D)` | `(None, 8, 38, 128)` | `73,856` |
| `batch_normalization_2 (BatchNormalization)` | `(None, 8, 38, 128)` | `512` |
| `max_pooling2d_2 (MaxPooling2D)` | `(None, 4, 19, 128)` | `0` |
| `spatial_dropout2d_2 (SpatialDropout2D)` | `(None, 4, 19, 128)` | `0` |
| `global_average_pooling2d (GlobalAveragePooling2D)` | `(None, 128)` | `0` |
| `dense (Dense)` | `(None, 128)` | `16,512` |
| `dropout (Dropout)` | `(None, 128)` | `0` |
| `dense_1 (Dense)` | `(None, 9)` | `1,161` |

**Total params:** `130,569` (510.04 KB)  
**Trainable params:** `130,057` (508.04 KB)  
**Non-trainable params:** `512` (2.00 KB)

This model is the final result of a great deal of experimentation, and it reached 100% accuracy on our 473 test points. It uses dropout to avoid overfitting (both with the kernels and with the dense layer at the output). We used early stopping and learning schedule rules based on validation accuracy to optimize our training. Our numerical optimizer was the ADAM, and we user a multi-class cross-entropy cost function. 

## Real-Time Inference

Before flashing our model onto our board, we experimented with real-time algorithms in `inference_test.py`. The pipeline is as follows:

1) Record input buffer (4s)
2) Extract optimal window to capture utterance (use a leaky integrator on the input buffer and pick its maximum value as the midpoint for the fixed input window)
3) Zero-out silence based on threshold to mimic the training data (similar process as in the `parse()` function, since files without zero-padding were usually noise)
4) Normalize, ensure float32 data, and extract window's MFCCs with the same `extract_mfccs()` function.
5) Feed reshaped MFCC tensor to the loaded model
6) Convert prediction into a label

After confirming the functionality of this pipeline, we then coded `main.py`, which is the script used in the Pi Zero W board on real-time inference. The pipeline for that script is:

1) Setup: Get weather data, load model, setup LEDs and LCD, flashing the RGB LED in purple to display readiness
2) Look for a button press and record to input buffer when detected
3) Normalize, extract optimal window to capture utterance (use a leaky integrator on the input buffer and pick its maximum value as the midpoint for the fixed input window)
4) Zero-out silence based on threshold to mimic the training data's zero-padding (similar process as in the `parse()` function, since files without zero-padding were usually noise)
5) Extract window's MFCCs with the same `extract_mfccs()` function
6) Feed reshaped MFCC tensor to the loaded model
7) Convert prediction into a command
8) Execute a command based on ENUM (control the RGB LED, display time with `datetime`, or display fetched temperature from `python_weather`).
9) Look for button press again and continue loop from step 3.


## References

[1] Suksri, Siwat and Thaweesak Yingthawornsuk. “Speech Recognition using MFCC.” (2012).

[2] [pyfda (Python Filter Design & Analysis Tool)](https://github.com/chipmuenk/pyfda)

[3] [Qualcomm Keyword Speech Dataset](https://www.qualcomm.com/developer/software/keyword-speech-dataset)

[4] [Moaz Abdeljalil. (2025). Background Noise \[Data set\]. Kaggle.](https://doi.org/10.34740/KAGGLE/DSV/10892389)

[5] [neonwatty. *machine-learning-refined* (GitHub repository, see LICENSE).](https://github.com/neonwatty/machine-learning-refined?tab=License-1-ov-file)



