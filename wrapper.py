import os
import shutil
import numpy as np
import soundfile as sf
import librosa
import data_augmentation as da
import json

"""
This script contains helper functions for moving, removing, splitting, and loading 
already-cleaned and processed audio files. 
"""

DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Mixed_Data"
TRAIN_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/train"
TEST_PATH  = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/test"
VAL_PATH   = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/val"
SAMPLE_RATE = 22050

# Used only for label->index; we derive the label string from the filename
LABEL_MAPPING = {
    'red': 0, 'green': 1, 'blue': 2, 'white': 3, 'off': 4,
    'time': 5, 'temperature': 6, 'unknown': 7, 'noise': 8
}
AUDIO_EXTS = {".wav", ".m4a", ".flac", ".mp3", ".ogg", ".opus", ".aiff", ".aif"}

def write_wav(file_name, audio_data):
    """
    Write mono WAV at SAMPLE_RATE (16-bit PCM).
    Returns the actual path written (ensures .wav extension).
    """
    root_out, ext_out = os.path.splitext(file_name)
    if ext_out.lower() != ".wav":
        file_name = root_out + ".wav"
    sf.write(file_name, audio_data.astype(np.float32), samplerate=SAMPLE_RATE, subtype="PCM_16")
    print(f"    Wrote wavfile: {file_name}")
    return file_name


def find_largest_length(dir_name):
    """
    Returs the largest data length in samples in a given directory.
    """
    max_size = 0.0
    for name in os.listdir(dir_name):

        #build full path
        file_name = os.path.join(dir_name, name)

        #Ignore non-audio data
        print(f"name = {name}")
        if name.startswith(".") or name.startswith("._"):
            continue  # skip .DS_Store and AppleDouble

        file_name = os.path.join(dir_name, name)
        print(f"Looping through file {name}")

        if not os.path.isfile(file_name):
            continue
    
        #load audio data
        audio_data, sr = librosa.load(file_name)
        print(f"    Loaded data of shape {audio_data.shape} and type {type(audio_data[0])}")

        size = 0
        #if stereo, keep track of the channel's length
        if len(audio_data.shape) != 1:
            size = audio_data.shape[1]
        
        #If mono, simply get the length
        else:
            size = len(audio_data)
        
        #Keep track of the largest file
        if size > max_size:
            print(f"\n\nNew max at {size}\n\n")
            max_size = size

    return max_size  

def force_standard_size(dirname, size):
    """"
    Parse through audio files in a directory and ensure they all have the same 
    length by either cutting them smaller or zero-padding them.

    Assumes all files were sanitized.

    dirname (str): Directory where the files of interest live
    size (int): Desired file size in samples
    """

    print(f"\nForcing standard size for files in {dirname}...")

    for name in os.listdir(dirname):
        # Ignore non-audio data
        print(f"\nname = {name}")
        if name.startswith(".") or name.startswith("._"):
            print("    (1) Not an audio file, continuing...")
            continue  # skip .DS_Store and AppleDouble

        # Build full file name
        file_name = os.path.join(dirname, name)

        # Split safely
        root, ext = os.path.splitext(file_name)
        ext = ext.lower().lstrip(".")
        print(f"    Split name successfully")

        # Expect WAVs only
        if ext != 'wav':
            print("     (2) Not a wav file:")
            print(f"    ext = {ext}")
            print(f"    Exiting function. Sanitize the data in {dirname} before calling force_standard_size() again.")
            raise ValueError()

        if not os.path.isfile(file_name):
            print("     (3) Not an audio file, continuing...")
            continue

        # Load at native rate to check against SAMPLE_RATE later
        audio_data, sr = librosa.load(file_name, sr=None, mono=True)
        print(f"    Loaded data of shape {audio_data.shape} and type {type(audio_data[0])}")

        if sr != SAMPLE_RATE:
            print(f"\nWrong sample rate. This script expects {SAMPLE_RATE}, but {name} is {sr}")
            print(f"    Exiting function. Sanitize the data in {dirname} before calling force_standard_size() again. \n")
            raise ValueError()

        if audio_data.ndim != 1:
            print(f"\n{name} is a stereo file. This script expects mono files.")
            print(f"    Exiting function. Sanitize the data in {dirname} before calling force_standard_size() again. \n")
            raise ValueError()

        # Get this file's size
        curr_size = int(audio_data.size)

        # Build fade-in and fade-out
        N = min(100, size)
        fade_in = np.linspace(0.0, 1.0, N, endpoint=True)
        fade_out = np.linspace(1.0, 0.0, N, endpoint=True)

        if curr_size < size:
            # center-pad with zeros, distributing the odd sample to the right
            size_diff = size - curr_size
            left = size_diff // 2
            right = size_diff - left
            fixed_audio_data = np.pad(audio_data, (left, right), mode='constant')

        elif curr_size > size:
            # center-crop
            start = (curr_size - size) // 2
            end = start + size
            fixed_audio_data = audio_data[start:end]

        else:
            # already the right size
            fixed_audio_data = audio_data.copy()

        # apply fades on the final, fixed-length buffer
        if N > 0:
            fixed_audio_data[:N] *= fade_in
            fixed_audio_data[-N:] *= fade_out

        # Write in place
        tmp_path = root + ".__tmp.wav"
        sf.write(tmp_path, fixed_audio_data.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
        os.replace(tmp_path, file_name)  # atomic on POSIX
        # (If something goes wrong before replace, tmp file may remain.)
        print(f"    Wrote fixed file: {file_name}")
        


def split_dataset(data_dir=DATA_PATH, train=70, test=15, val=15):
    """
    Split the dataset into train / test / val, preserving each label's proportion.
    The label is taken from the filename prefix before the first underscore,
    with a special case: files starting with 'chunk_' are treated as 'noise'.
    """
    if train + test + val != 100:
        print("Warning: percentages don't sum to 100. Using 70/15/15.")
        train, test, val = 70, 15, 15

    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(TEST_PATH,  exist_ok=True)
    os.makedirs(VAL_PATH,   exist_ok=True)

    by_label: dict[str, list[str]] = {lbl: [] for lbl in LABEL_MAPPING.keys()}

    for name in os.listdir(data_dir):

        #Skip bad data/nested dirs
        if name.startswith(".") or name.startswith("._"):
            continue
        src = os.path.join(data_dir, name)
        if not os.path.isfile(src):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() not in AUDIO_EXTS:
            continue

        # Get label
        raw_label = name.split("_", 1)[0]
        label = "noise" if raw_label == "chunk" else raw_label

        # Handle exceptions in the naming convention
        if label not in by_label:
            label = "unknown"

        # Append current file
        by_label[label].append(src)   

    # Get a random number generator with deterministic split (seeded)
    rng = np.random.default_rng(7)

    # Split the data acording to input
    split_counts = {}
    for label, files in by_label.items():
        if not files:
            split_counts[label] = (0, 0, 0)
            continue

        # Copy and shuffle files
        files = files.copy()
        rng.shuffle(files)

        # Determine how many files go to each location
        n = len(files)
        n_train = int(np.floor(n * train / 100.0))
        n_test  = int(np.floor(n * test  / 100.0))
        n_val   = n - n_train - n_test  

        split_counts[label] = (n_train, n_test, n_val)

        #Populate files 
        train_files = files[:n_train]
        test_files  = files[n_train:n_train + n_test]
        val_files   = files[n_train + n_test:]

        # Copy data
        for src in train_files:
            shutil.copy2(src, os.path.join(TRAIN_PATH, os.path.basename(src)))
        for src in test_files:
            shutil.copy2(src, os.path.join(TEST_PATH,  os.path.basename(src)))
        for src in val_files:
            shutil.copy2(src, os.path.join(VAL_PATH,   os.path.basename(src)))

    # Report
    print("\nSplit summary (train, test, val) per label:")
    total_train = total_test = total_val = 0
    for label in LABEL_MAPPING.keys():
        t, te, v = split_counts.get(label, (0, 0, 0))
        total_train += t; total_test += te; total_val += v
        print(f"  {label:12s}: {t:4d}, {te:4d}, {v:4d}")
    print(f"\nTotals -> train: {total_train}, test: {total_test}, val: {total_val}")
    print("Done.")
    return


def write_n_random(write_dir, n=350, read_dir=DATA_PATH, write_all=False):
    """
    Randomly copy n .wav files from read_dir into write_dir.
    Files are copied as-is (same name, data, metadata).
    """

    # Make sure destination exists
    os.makedirs(write_dir, exist_ok=True)

    # If write all files flag is true, reset n
    # CAUTION: This requires that there are no directories on write_dir
    if write_all:
        lst =os.listdir(read_dir)
        n = len(lst)

    # Collect candidate .wav files
    wav_files = []
    for name in os.listdir(read_dir):
        # Skip hidden/AppleDouble files
        if name.startswith(".") or name.startswith("._"):
            continue

        src = os.path.join(read_dir, name)
        if not os.path.isfile(src):
            continue

        _, ext = os.path.splitext(name)
        if ext.lower() != ".wav":
            continue

        wav_files.append(src)

    if not wav_files:
        print(f"No .wav files found in '{read_dir}'. Nothing to do.")
        return

    # If n is larger than available, cap it
    if n > len(wav_files):
        print(f"Requested {n} files but only found {len(wav_files)} .wav files.")
        print("Copying all available files instead.")
        n = len(wav_files)

    # Randomly choose n distinct files
    rng = np.random.default_rng()  # no fixed seed → different each run
    indices = rng.choice(len(wav_files), size=n, replace=False)
    chosen_files = [wav_files[i] for i in indices]

    # Copy them over
    for src in chosen_files:
        dst = os.path.join(write_dir, os.path.basename(src))
        shutil.copy2(src, dst)

    print(f"Copied {n} .wav files from '{read_dir}' to '{write_dir}'.")

def chunk_2_noise(directory):

    """
    Renames files in 'directory' that were labelled 'chunk' to 'noise'. Keep
    the same rest of the name.
    """

    print(f"Renaming chunks to noise in {directory}...\n")

    for name in os.listdir(directory):
        # Skip hidden/AppleDouble files
        if name.startswith(".") or name.startswith("._"):
            continue

        old_path = os.path.join(directory, name)
        if not os.path.isfile(old_path):
            continue

        stem, ext = os.path.splitext(name)
        if ext.lower() != ".wav":
            continue

        parts = stem.split("_")
        if len(parts) < 2:
            continue  # unexpected format, skip

        label = parts[0]
        num   = parts[1]

        print(f"    Found label ({label})")
        print(f"    With number label ({num})")

        if label == "chunk":
            new_name = f"noise_{num}{ext}"
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"    Renamed {name} -> {new_name}")


def clean_directory(directory):
    """
    Remove any non-wav data from directory. Ensure there is only relevant data. 
    """

    print(f"Cleaning directory {directory}...")

    for name in os.listdir(directory):

        # Get full file name
        file_name = os.path.join(directory, name)

        # Skip directories entirely 
        if not os.path.isfile(file_name):
            continue

        # Remove Apple hidden files
        if name.startswith(".") or name.startswith("._"):
            print(f"Removing {name}, as it was identified as a hidden file.")
            os.remove(file_name)
            continue

        # Remove non-wav files
        _, ext = os.path.splitext(name)
        if ext.lower() != ".wav":
            print(f"Removing {name}, as it was identified as not being a .wav file.")
            os.remove(file_name)
            continue
        
    print("\nDone.\n")
    return

def remove_label(directory, bad_label):
    """
    Removes all instances of a given label for the proper naming convention in 'directory'. 
    """

    print(f"Removing label {bad_label} from {directory}...")

    for name in os.listdir(directory):

        # Get full file name
        file_name = os.path.join(directory, name)

        # Skip directories entirely
        if not os.path.isfile(file_name):
            continue

        # Get label
        label = name.split("_", 1)[0]

        # Remove if the label is the bad label
        if label == bad_label:
            print(f"Removing {name}")
            os.remove(file_name)
        
    print(f"Removed all files flagged as '{bad_label}' from {directory}.")
    return

def remove_n_random(directory, n=100):
    """
    Remove n random .wav files from `directory`.
    - Only operates on regular files (not subdirectories).
    - Only considers files with .wav extension (case-insensitive).
    """

    # Collect candidate .wav files
    wav_files = []
    for name in os.listdir(directory):
        file_path = os.path.join(directory, name)

        # Skip non-files (directories, etc.)
        if not os.path.isfile(file_path):
            continue

        # Skip hidden/AppleDouble files
        if name.startswith(".") or name.startswith("._"):
            continue

        # Only keep .wav files
        _, ext = os.path.splitext(name)
        if ext.lower() != ".wav":
            continue

        wav_files.append(file_path)

    if not wav_files:
        print(f"No .wav files found in '{directory}'. Nothing to remove.")
        return

    # If n is larger than available, cap it
    if n > len(wav_files):
        print(f"Requested to remove {n} files but only found {len(wav_files)} .wav files.")
        print("Removing all available .wav files instead.")
        n = len(wav_files)

    # Randomly choose n distinct files
    rng = np.random.default_rng()  # different result each call
    indices = rng.choice(len(wav_files), size=n, replace=False)
    chosen_files = [wav_files[i] for i in indices]

    # Remove chosen files
    for file_path in chosen_files:
        print(f"Removing {os.path.basename(file_path)}")
        os.remove(file_path)

    print(f"Removed {n} .wav files from '{directory}'.")

import os

def rename_points_per_label(directory, label):
    """
    In a given `directory` corresponding to a specific `label`, rename any .wav file
    that does NOT follow the convention:  '{label}_RESTOFNAME'
    
    Mis-labeled files are renamed to: '{label}_mistery_NUM.wav', where NUM runs from
    1 to N in the order they are found during the directory scan.
    """
    print(f"Renaming poorly labeled files in '{directory}' for label '{label}'...")

    mystery_count = 0
    expected_prefix = f"{label}_"

    for name in os.listdir(directory):
        old_path = os.path.join(directory, name)

        # Skip non-files (directories, etc.)
        if not os.path.isfile(old_path):
            continue

        # Skip hidden/AppleDouble files
        if name.startswith(".") or name.startswith("._"):
            continue

        # Only operate on .wav files
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".wav":
            continue

        # If it already follows label_RESTOFNAME, leave it alone
        if name.startswith(expected_prefix):
            continue

        # Otherwise, it's a "mistery" file for this label
        mystery_count += 1
        new_name = f"{label}_mistery_{mystery_count}.wav"
        new_path = os.path.join(directory, new_name)

        print(f"  Renaming '{name}' -> '{new_name}'")
        os.rename(old_path, new_path)

    print(f"Done. Relabeled {mystery_count} file(s) in '{directory}' as '{label}_mistery_NUM.wav'.")
    return

import os
import json
import numpy as np

def wrapper(dirname, augmented=False):
    """
    Assumes feature dictionaries are located in subfolders of `dirname`.
    Each subfolder contains a JSON file with a list of
        [flat_feature_vector, label_vector]
    pairs, as produced by `build_feature_dict`.

    Parameters
    ----------
    dirname : str
        Path to the directory that contains 'train', 'train_augmented',
        'test', and 'val' subdirectories.
    augmented : bool
        If True, use 'train_augmented' as the training source.
        If False, use 'train'.

    Returns
    -------
    (X_train, y_train), (X_test, y_test), (X_val, y_val)
        Each X_* is an array of shape (N, D),
        each y_* is an array of shape (N, C) (whatever your label vector length is).
    """

    # Which subfolders correspond to which splits
    classes = ["train", "test", "val"]
    if augmented:
        classes[0] = "train_augmented"   # use augmented train

    def load_split(split_name):
        """
        Helper to load one split ('train', 'train_augmented', 'test', 'val').
        """
        split_dir = os.path.join(dirname, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Directory not found: {split_dir}")

        # Default expected JSON name: '<split_name>_feature_dict.json'
        expected_json = f"{split_name}_feature_dict.json"
        json_path = os.path.join(split_dir, expected_json)

        # If the expected file is not there, fall back to "first .json in folder"
        if not os.path.isfile(json_path):
            json_files = [f for f in os.listdir(split_dir) if f.endswith(".json")]
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {split_dir}")
            # Take the first one as a fallback
            json_path = os.path.join(split_dir, json_files[0])

        print(f"Loading feature dict from: {json_path}")
        with open(json_path, 'r') as f:
            feature_tuples = json.load(f)

        # Handle empty dict case
        if not feature_tuples:
            X = np.empty((0, 0), dtype=float)
            y = np.empty((0, 0), dtype=float)
            return X, y

        # feature_tuples is list of [mfcc_flat, vec_label_list]
        X_list = [pair[0] for pair in feature_tuples]
        y_list = [pair[1] for pair in feature_tuples]

        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=float)

        print(f"  -> Loaded {split_name}: X.shape = {X.shape}, y.shape = {y.shape}")
        return X, y

    # Load each split
    split_train_name = classes[0]  # 'train' or 'train_augmented'
    split_test_name  = classes[1]  # 'test'
    split_val_name   = classes[2]  # 'val'

    X_train, y_train = load_split(split_train_name)
    X_test,  y_test  = load_split(split_test_name)
    X_val,   y_val   = load_split(split_val_name)

    print("Loaded all data.")

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


if __name__ == "__main__":

    # Split dataset
    if False:
        split_dataset()


    # Rename chunks to noise
    if False:
        chunk_2_noise("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/noise")

    # Write n files to a dir
    if False:
        write_n_random(write_dir="/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/noise", 
                        read_dir="/Users/ericoliviera/Downloads/noise_chunks", n=600)

    # Clean Data per label
    if False:

        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/red")
        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/green")
        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/blue")
        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/white")
        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/off")
        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/time")
        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/temperature")
        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/unknown")
        clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/noise")

    if False:
        rename_points_per_label(directory="/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data/unknown", 
                                 label= "unknown")

    # Write all label files to Mixed
    labels = ["red", "green", "blue", "white", "off", "time", "temperature", "unknown", "noise"]
    if False:
        base_path = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data"

        for l in labels:
            curr_label_path = os.path.join(base_path, l)
            write_n_random(write_dir="/Users/ericoliviera/Desktop/Data/smart-home-ksw/Mixed_Data", 
                           read_dir=curr_label_path, 
                           write_all=True)
            
    # Split Dataset:
    if False:
        split_dataset()

    if True:
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = wrapper(
            dirname="/Users/ericoliviera/Desktop/Data/smart-home-ksw/Feature_dicts/", 
            augmented=False)
        
        print(f"Train's x shape: {x_train.shape}")
        print(f"Train's y shape: {y_train.shape}")

        print(f"Test's x shape: {x_test.shape}")
        print(f"Test's y shape: {y_test.shape}")

        print(f"Val's x shape: {x_val.shape}")
        print(f"Val's y shape: {y_val.shape}")

    pass