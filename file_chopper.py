import numpy as np
import os
import matplotlib.pyplot as plt
import librosa

"""
This script chops long audio samples into pre-determined length windows based 
on the signal level. It attemopts to extract relevant utterances from speech. 
"""

FULL_DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_dataset_copy2"
SAMPLE_RATE = 44100
DURATION = SAMPLE_RATE * 0.8 #800 ms split 
CHOPPED_DIR = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_dataset_copy2/chops"

def parse(plot_first=True):
    """
    This function assumes the dataset is uniform (i.e. already sanitezed)
    """
    # Check if chops exists. If not, create it
    if not os.path.exists(CHOPPED_DIR):
        os.makedirs(CHOPPED_DIR)

    #Loop through directory
    for name in os.listdir(FULL_DATA_PATH):
        file_name = os.path.join(FULL_DATA_PATH, name)       


if __name__ == "__main__":
    max_size = find_largest_length(FULL_DATA_PATH)
    print(f"Max size = {max_size}")