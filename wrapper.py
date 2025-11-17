import os
import shutil
import numpy as np
import soundfile as sf
import librosa

"""
This script contains helper functions for moving, removing, splitting, and loading 
already-cleaned and processed audio files. 
"""

DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/All_Data"
TRAIN_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/train"
TEST_PATH  = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/test"
VAL_PATH   = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/val"
SAMPLE_RATE = 22050

# Used only for label->index; we derive the label string from the filename.
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

def split_dataset(data_dir=DATA_PATH, train=70, test=15, val=15):
    """
    Split the dataset into train / test / val, preserving each label's proportion.
    The label is taken from the filename prefix before the first underscore,
    with a special case: files starting with 'chunk_' are treated as 'noise'.
    """
    # Check if percentages are coherent and force a standard value if not
    if train + test + val != 100:
        print("Warning: percentages don't sum to 100. Using 70/15/15.")
        train, test, val = 70, 15, 15

    # Ensure output directories exist
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(TEST_PATH,  exist_ok=True)
    os.makedirs(VAL_PATH,   exist_ok=True)

    # Collect files by label
    by_label: dict[str, list[str]] = {lbl: [] for lbl in LABEL_MAPPING.keys()}

    # Loop through directory 
    for name in os.listdir(data_dir):

        #Avoid bad files
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

        # skip unknown labels not in mapping (optional: add them to 'unknown')
        if label not in by_label:
            # treat truly unseen prefixes as 'unknown'
            label = "unknown"
        by_label[label].append(src)

        by_label[label].append(src)

    # Random generator 
    rng = np.random.default_rng(7)  # deterministic shuffle

    split_counts = {}
    for label, files in by_label.items():
        if not files:
            split_counts[label] = (0, 0, 0)
            continue

        # Shuffle
        files = files.copy()
        rng.shuffle(files)

        # Get the number of files in each class per label
        n = len(files)
        n_train = int(np.floor(n * train / 100.0))
        n_test  = int(np.floor(n * test  / 100.0))
        n_val   = n - n_train - n_test  

        split_counts[label] = (n_train, n_test, n_val)

        # Split files 
        train_files = files[:n_train]
        test_files  = files[n_train:n_train + n_test]
        val_files   = files[n_train + n_test:]

        # Copy (overwrite if already exists)
        for src in train_files:
            shutil.copy2(src, os.path.join(TRAIN_PATH, os.path.basename(src)))
        for src in test_files:
            shutil.copy2(src, os.path.join(TEST_PATH,  os.path.basename(src)))
        for src in val_files:
            shutil.copy2(src, os.path.join(VAL_PATH,   os.path.basename(src)))

    # Print info
    print("\nSplit summary (train, test, val) per label:")
    total_train = total_test = total_val = 0
    for label in LABEL_MAPPING.keys():
        t, te, v = split_counts.get(label, (0, 0, 0))
        total_train += t; total_test += te; total_val += v
        print(f"  {label:12s}: {t:4d}, {te:4d}, {v:4d}")
    print(f"\nTotals -> train: {total_train}, test: {total_test}, val: {total_val}")
    print("Done.")

def write_n_random(write_dir, n=350, read_dir=DATA_PATH):
    """
    Randomly copy n .wav files from read_dir into write_dir.
    Files are copied as-is (same name, data, metadata).
    """

    # Make sure destination exists
    os.makedirs(write_dir, exist_ok=True)

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

    print(f"Removing label {bad_label} from {directory}...")

    for name in os.listdir(directory):

        # Get full file name
        file_name = os.path.join(directory, name)

        # Skip directories entirely
        if not os.path.isfile(file_name):
            continue

        # Get label
        label = name.split("_", 1)[0]

        if label == bad_label:
            print(f"Removing {name}")
            os.remove(file_name)
        
    print(f"Removed all files flagged as '{bad_label}' from {directory}.")
    return




if __name__ == "__main__":
    #split_dataset()
    #chunk_2_noise("/Users/ericoliviera/Desktop/Data/smart-home-ksw/chops/noise")
    #write_n_random(write_dir="/Users/ericoliviera/Desktop/Data/smart-home-ksw/All_Data")

    clean_directory("/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_Data/val")

    pass