import numpy as np
import json
import os
import librosa

"""
This script will apply simple transformations to the vanilla keyword-spotting dataset and
generate new training points for model robustness. 
"""

VANILLA_DATA_PATH = ""
AUGMENTED_DATAPATH = ""
DURATION = 0

def augment_data_set():
    pass

def sanitize_vanilla_dataset():
    """
    --> Edit dataset in place. 

    - Ensure all files are:
        * Mono
        * 44.1 kHz
        * m4a
        * DURATION s long 

    - Normalize all files

    """
    pass

def low_pass():
    pass

def high_pass():
    pass

def band_pass():
    pass

def pitch_up():
    pass

def pitch_down():
    pass

def add_noise():
    pass
