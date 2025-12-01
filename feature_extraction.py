import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Optional  
import os
import librosa.display
import matplotlib.pyplot as plt
import json

"""
This file defines the feature extraction processes and builds JSON feature dictionary.
"""

DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_Datasets/Chops1"
FEATURE_DICT_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_Datasets/dict1"
SAMPLE_RATE = 22050  # default target SR for feature extraction
N_MFCC = 32
T_FRAMES = 154

def extract_mfccs(
    audio_data: np.ndarray,
    n_mfcc: int = 32,
    # STFT / framing
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    # Mel / magnitude
    power: float = 2.0,
    n_mels: int = 64,
    htk: bool = False,
    fmin: float = 20.0,
    fmax: Optional[float] = 16000.0,
    # dB scaling
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    # DCT / MFCC post-processing
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: int = 0,
    # sampling rate override
    sr: Optional[int] = None,
) -> np.ndarray:
    """Return MFCCs of shape (n_mfcc, T)."""
    sr_eff = SAMPLE_RATE if sr is None else sr
    nyq = 0.5 * sr_eff
    fmax_eff = min(fmax if fmax is not None else nyq, nyq)

    # 1) Mel spectrogram (power)
    S_mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr_eff,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        power=power,
        n_mels=n_mels,
        htk=htk,
        fmin=fmin,
        fmax=fmax_eff,
    )

    # 2) Convert to dB
    S_db = librosa.power_to_db(S_mel, ref=ref, amin=amin, top_db=top_db)

    # 3) MFCCs from Mel dB
    mfccs = librosa.feature.mfcc(
        S=S_db,
        n_mfcc=n_mfcc,
        dct_type=dct_type,
        norm=norm,
        lifter=lifter,
    )
    return mfccs

def vectorize_label(label):
    
    """
    Turns string label into vectorized label.
    """

    # Dictionary for mapping string label to dimension in vectorized label
    label_maping = {'red': 0, 'green': 1, 'blue': 2, 'white': 3, 'off': 4, 
                     'time': 5, 'temperature': 6, 'unknown': 7, 'noise': 8}
    
    # Get basis vector for this label
    e = np.zeros((9, 1))

    # Handle chunks, which are noise
    if label == "chunk":
        label = 'noise'

    e[label_maping[label]] = 1

    return e


def build_feature_dict(dir_name=DATA_PATH, write_dir_name=FEATURE_DICT_PATH, dict_name='feature_dict'):
    """
    Parse dir_name, extract features, vectorize labels, and dump a JSON list of
    [flat_feature_vector, label_vector] pairs to write_dir_name/dict_name.json.
    """

    # Ensure output directory exists 
    os.makedirs(write_dir_name, exist_ok=True)

    feature_tuples = []
    for name in os.listdir(dir_name):
        # Ignore hidden / DS files early
        if name.startswith(".") or name.startswith("._"):
            continue

        file_name = os.path.join(dir_name, name)
        if not os.path.isfile(file_name):
            continue

        print(f"name = {name}\nLooping through file {name}")

        # Load 
        audio_data, sr = librosa.load(file_name, sr=SAMPLE_RATE, mono=True)
        print(f"    Loaded data of shape {audio_data.shape} and type {type(audio_data[0])}")

        if sr != SAMPLE_RATE:
            raise ValueError("Sample rate is not correct. Ensure data is sanitized.")

        # Get label
        label = name.split('_')[0]
        print(f"\n  Label = {label}\n")
        vec_label = vectorize_label(label)

        # Get MFCCs and flatten to list 
        mfccs = extract_mfccs(audio_data=audio_data)
        print(f"  MFCCs shape = {mfccs.shape}\n")

        mfcc_flat = mfccs.flatten().tolist()
        vec_label_list = np.asarray(vec_label).astype(float).flatten().tolist()

        # Store as JSON friendly pair
        feature_tuples.append([mfcc_flat, vec_label_list])

    print(f"Build feature list of length {len(feature_tuples)}")

    # Dump data
    json_file_path = os.path.join(write_dir_name, f"{dict_name}.json")
    with open(json_file_path, 'w') as f:
        json.dump(feature_tuples, f, indent=4)

    print(f"Wrote feature dictionary to: {json_file_path}")
    return


if __name__ == "__main__":

    print("\n\nEntered main all good.\n\n")