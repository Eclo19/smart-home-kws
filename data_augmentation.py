import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import wrapper as wp
from scipy.signal import lfilter, filtfilt, sosfiltfilt

VANILLA_DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/train"
AUGMENTED_DATAPATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/train_augmented"  
SAMPLE_RATE = 22050
AUDIO_EXTS = {"wav", "m4a", "flac", "mp3", "ogg", "opus", "aiff", "aif"}


def sanitize_vanilla_dataset():
    """
    Edit dataset in place:
      - mono
      - SAMPLE_RATE kHz
      - .wav
      - normalized (peak)
    """

    print("\nStarting data sanitation...")

    # Loop through every file in the directory
    for name in os.listdir(VANILLA_DATA_PATH):

        # Ignore non-audio data
        print(f"\nname = {name}")
        if name.startswith(".") or name.startswith("._"):
            print("    (1) Not an audio file, continuing...")
            continue  # skip .DS_Store and AppleDouble

        file_name = os.path.join(VANILLA_DATA_PATH, name)

        # Split safely
        root, ext = os.path.splitext(file_name)
        ext = ext.lower().lstrip(".")
        print(f"    Splited name succesfully")

        if ext.lower() not in AUDIO_EXTS:
            print("     (2) Not an audio file, continuing...")
            print(f"    ext = {ext}")
            continue

        if not os.path.isfile(file_name):
            print("     (3) Not an audio file, continuing...")
            continue

        # Load
        audio_data, sr = librosa.load(file_name, sr=None)
        print(f"    Loaded data of shape {audio_data.shape} and type {type(audio_data[0])}")
        
        if False:   
            sd.play(audio_data, samplerate=sr)
            sd.wait()

        # Already mono+wav+sr? (length may still need adjusting)
        good_format = (ext == "wav" and audio_data.shape[0] == 1 and sr == SAMPLE_RATE)

        #Continue if the format is good
        if good_format:
            continue

        # Force mono
        if len(audio_data.shape) != 1:
            audio_data = np.sum(audio_data, axis=1)/2
            print(f"    Converted {file_name} to mono succesfully")

        # Resample 
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            print(f"    Converted {file_name}'s sample rate to {SAMPLE_RATE}")

        # Peak normalize
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            audio_data = audio_data / peak
            print(f"    Normalized {file_name}'s data")

        # Write helper
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

        # Force WAV on disk
        if ext == "wav":
            # In-place rewrite via temp, then atomic replace
            tmp = root + "._temp.wav"
            written = write_wav(tmp, audio_data)
            os.replace(written, file_name)
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            print(f"    {name}: forced to standards.")
        else:
            # Convert to .wav, then delete original
            out_path = root + ".wav"
            write_wav(out_path, audio_data)
            os.remove(file_name)
            print(f"    {name}: converted -> {os.path.basename(out_path)} (mono) and deleted original.")
        
    print(f"\n\nSuccesfully sanitized all files in {VANILLA_DATA_PATH}\n\n")


def augment_data_set(roll=True):
    """
    Generates an artificially augmented dataset in the global path 'AUGMENTED_DATAPATH' from
    files in the global 'VANILLA_DATA_PATH' by calling all the relevant transformations.
    """
    print("\nStarting artificial data augmentation.\n ")
    print("     Sanitizing vanilla data... ")
    sanitize_vanilla_dataset()

    os.makedirs(AUGMENTED_DATAPATH, exist_ok=True)  # in case it doesn't exist

    print("     Starting augmentation: ")
    for name in os.listdir(VANILLA_DATA_PATH):

        # Ignore non-audio data / hidden files
        print(f"\nname = {name}")
        if name.startswith(".") or name.startswith("._"):
            print("    (1) Not an audio file, continuing...")
            continue

        in_path = os.path.join(VANILLA_DATA_PATH, name)
        root_name, ext = os.path.splitext(name)
        ext = ext.lower().lstrip(".")
        print("    Splited name succesfully")

        if ext not in AUDIO_EXTS or not os.path.isfile(in_path):
            print("    (2) Not an audio file, continuing...")
            print(f"    ext = {ext}")
            continue

        # Load
        audio_data, sr = librosa.load(in_path, sr=None)
        print(f"    Loaded data of shape {audio_data.shape} and type {type(audio_data[0])}")

        # Writer
        def write_wav(out_path, audio):
            root_out, ext_out = os.path.splitext(out_path)
            if ext_out.lower() != ".wav":
                out_path = root_out + ".wav"
            sf.write(out_path, audio.astype(np.float32), samplerate=SAMPLE_RATE, subtype="PCM_16")
            print(f"    Wrote wavfile: {out_path}")
            return out_path

         # Write ORIGINAL into augmented dir
        vanilla_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}.wav")
        write_wav(vanilla_out, audio_data)

        # Low-pass, then write with _lp suffix 
        lp_audio_data = low_pass(audio_data)
        lp_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_lp.wav")
        write_wav(lp_out, lp_audio_data)

        # High-pass, then write with _hp suffix 
        hp_audio_data = high_pass(audio_data)
        hp_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_hp.wav")
        write_wav(hp_out, hp_audio_data)

        # Band-pass, then write with _bp suffix 
        bp_audio_data = band_pass(audio_data)
        bp_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_bp.wav")
        write_wav(bp_out, bp_audio_data)

        # Pitch-shift up, then write with _pu suffix 
        pu_audio_data = pitch_up(audio_data)
        pu_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_pu.wav")
        write_wav(pu_out, pu_audio_data)

        # Pitch-shift down, then write with _pd suffix 
        pd_audio_data = pitch_down(audio_data)
        pd_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_pd.wav")
        write_wav(pd_out, pd_audio_data)

        # Add noise, then write with _nz suffix 
        nz_audio_data = add_noise(audio_data)
        nz_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_nz.wav")
        write_wav(nz_out, nz_audio_data)

        # Rolling the array
        if roll:

            min_roll = 200

            # Find zero-padded regions by looking at first/last non-zero samples
            eps = 1e-8
            nonzero_idx = np.where(np.abs(audio_data) > eps)[0]

            if nonzero_idx.size == 0:
                print("    (3) All-zero file, skipping roll augmentations...")
            else:
                first_nz = nonzero_idx[0]
                last_nz  = nonzero_idx[-1]

                left_pad  = first_nz
                right_pad = len(audio_data) - 1 - last_nz
                max_zero_padded_length = max(left_pad, right_pad)

                print(f"    Zero padding: left={left_pad}, right={right_pad}, "
                      f"max_zero_padded_length={max_zero_padded_length}")

                # Require at least 400 samples of zero padding to roll into
                if max_zero_padded_length <= min_roll:
                    print("    (4) Not enough zero padding to do meaningful roll, skipping...")
                else:
                    low = min_roll
                    high = max_zero_padded_length  # [400, max_zero_padded_length)

                    # Roll to the right x2
                    rr_constant = np.random.randint(low, high)
                    rolled_right_audio_data = np.roll(audio_data, rr_constant)
                    rr_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_rr.wav")
                    write_wav(rr_out, rolled_right_audio_data)

                    rr_constant = np.random.randint(low, high)
                    rolled_right_audio_data = np.roll(audio_data, rr_constant)
                    rr_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_rr2.wav")
                    write_wav(rr_out, rolled_right_audio_data)

                    # Roll to the left x2
                    rl_constant = -np.random.randint(low, high)
                    rolled_left_audio_data = np.roll(audio_data, rl_constant)
                    rl_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_rl.wav")
                    write_wav(rl_out, rolled_left_audio_data)

                    rl_constant = -np.random.randint(low, high)
                    rolled_left_audio_data = np.roll(audio_data, rl_constant)
                    rl_out = os.path.join(AUGMENTED_DATAPATH, f"{root_name}_rl2.wav")
                    write_wav(rl_out, rolled_left_audio_data)

    return

def low_pass(audio_data, path="Filters/lp7.csv"):

    """
    Assumes audio_data is sanitized. 

    audio_data(ndarray): data to be filtered

    Returns: Filtered audio_data
    """

    print("\nApplying lowpass filter...")

    # Load filter taps 
    try:
        b = np.loadtxt(path, delimiter=',', dtype=float)
    except Exception:
        lines, vals = [], []
        with open(path, 'r') as f:
            for line in f:
                try:
                    # take all comma-separated tokens that parse as floats
                    nums = [float(tok) for tok in line.strip().split(',') if tok.strip()]
                    if nums:
                        vals.append(nums)
                except ValueError:
                    continue
        b = np.array(vals, dtype=float)
    b = np.squeeze(b)
    if b.ndim != 1:
        # if b is a matrix with 1 row/col, flatten
        if 1 in b.shape:
            b = b.flatten()
        else:
            raise ValueError("FIR CSV doesn't look like a single list of taps.")
    
    print("Loaded filter taps")

    # Work with float32
    audio_data = np.asarray(audio_data, dtype=np.float32)

    if audio_data.ndim != 1:
        raise ValueError("audio must be 1-D mono array.")

    # causal filtering
    y = lfilter(b, [1.0], audio_data)
    return y.astype(np.float32)
    

def high_pass(audio_data, path="Filters/hp8.csv"):
    """
    Assumes audio_data is sanitized. 

    audio_data(ndarray): data to be filtered

    path (str): Path to the filter coefficients

    Returns: Filtered audio_data
    """

    print("\nApplying highpass filter...")

    # Load filter taps 
    try:
        b = np.loadtxt(path, delimiter=',', dtype=float)
    except Exception:
        lines, vals = [], []
        with open(path, 'r') as f:
            for line in f:
                try:
                    # take all comma-separated tokens that parse as floats
                    nums = [float(tok) for tok in line.strip().split(',') if tok.strip()]
                    if nums:
                        vals.append(nums)
                except ValueError:
                    continue
        b = np.array(vals, dtype=float)
    b = np.squeeze(b)
    if b.ndim != 1:
        # if b is a matrix with 1 row/col, flatten
        if 1 in b.shape:
            b = b.flatten()
        else:
            raise ValueError("FIR CSV doesn't look like a single list of taps.")
    
    print("Loaded filter taps")

    # Work with float32
    audio_data = np.asarray(audio_data, dtype=np.float32)

    if audio_data.ndim != 1:
        raise ValueError("audio must be 1-D mono array.")

    # causal filtering
    y = lfilter(b, [1.0], audio_data)
    return y.astype(np.float32)

def band_pass(audio_data,path="Filters/bp1.csv"):

    """
    Assumes audio_data is sanitized. 

    audio_data(ndarray): data to be filtered

    path (str): Path to the filter coefficients

    Returns: Filtered audio_data
    """

    print("\nApplying bandpass filter...")

    # Load filter taps 
    try:
        b = np.loadtxt(path, delimiter=',', dtype=float)
    except Exception:
        lines, vals = [], []
        with open(path, 'r') as f:
            for line in f:
                try:
                    # take all comma-separated tokens that parse as floats
                    nums = [float(tok) for tok in line.strip().split(',') if tok.strip()]
                    if nums:
                        vals.append(nums)
                except ValueError:
                    continue
        b = np.array(vals, dtype=float)
    b = np.squeeze(b)
    if b.ndim != 1:
        # if b is a matrix with 1 row/col, flatten
        if 1 in b.shape:
            b = b.flatten()
        else:
            raise ValueError("FIR CSV doesn't look like a single list of taps.")
    
    print("Loaded filter taps")

    # Work with float32
    audio_data = np.asarray(audio_data, dtype=np.float32)

    if audio_data.ndim != 1:
        raise ValueError("audio must be 1-D mono array.")

    # causal filtering
    y = lfilter(b, [1.0], audio_data)
    return y.astype(np.float32)

def pitch_up(audio_data):
    pitched_audio_data = librosa.effects.pitch_shift(audio_data, sr=SAMPLE_RATE, n_steps=1)
    return pitched_audio_data


def pitch_down(audio_data):
    pitched_audio_data = librosa.effects.pitch_shift(audio_data, sr=SAMPLE_RATE, n_steps=-1)
    return pitched_audio_data
    pass

def add_noise(audio_data, scale=0.35):
    """
    Add red noise (Low-passed white noise) to audio_data
    """

    white_noise = np.random.randn(audio_data.size)
    red_noise = low_pass(white_noise, path="/Users/ericoliviera/Desktop/My_Repositories/smart-home-kws/Filters/lp1.csv")

    if False:
        sd.play(red_noise, SAMPLE_RATE)
        sd.wait()

    noisy_audio_data = audio_data + red_noise * scale
    return noisy_audio_data

if __name__ == "__main__":

    #force_standard_size('/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_dataset_4', size)

    # Test lowpass
    if False:

        for name in os.listdir(VANILLA_DATA_PATH):

            file_name = os.path.join(VANILLA_DATA_PATH, name)

            # Ignore non-audio data
            print(f"\nname = {name}")
            if name.startswith(".") or name.startswith("._"):
                print("    (1) Not an audio file, continuing...")
                continue  # skip .DS_Store and AppleDouble

            audio_data, sr = librosa.load(file_name, sr=None, mono=True)

            filt_audio_data = low_pass(audio_data)

            print(f"\nNow playing vanilla and filtered {name}:\n ")

            sd.play(audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

            sd.play(filt_audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

    # Test highpass
    if False:

        for name in os.listdir(VANILLA_DATA_PATH):

            file_name = os.path.join(VANILLA_DATA_PATH, name)

            # Ignore non-audio data
            print(f"\nname = {name}")
            if name.startswith(".") or name.startswith("._"):
                print("    (1) Not an audio file, continuing...")
                continue  # skip .DS_Store and AppleDouble

            audio_data, sr = librosa.load(file_name, sr=None, mono=True)

            filt_audio_data = high_pass(audio_data)

            print(f"Now playing vanilla and filtered {name}:\n ")

            sd.play(audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

            sd.play(filt_audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

    # Test bandpass
    if False:

        for name in os.listdir(VANILLA_DATA_PATH):

            file_name = os.path.join(VANILLA_DATA_PATH, name)

            # Ignore non-audio data
            print(f"\nname = {name}")
            if name.startswith(".") or name.startswith("._"):
                print("    (1) Not an audio file, continuing...")
                continue  # skip .DS_Store and AppleDouble

            audio_data, sr = librosa.load(file_name, sr=None, mono=True)

            filt_audio_data = band_pass(audio_data)

            print(f"Now playing vanilla and filtered {name}:\n ")

            sd.play(audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

            sd.play(filt_audio_data, samplerate=SAMPLE_RATE)
            sd.wait()


    # Test pitch_up
    if False:

        for name in os.listdir(VANILLA_DATA_PATH):

            file_name = os.path.join(VANILLA_DATA_PATH, name)

            # Ignore non-audio data
            print(f"\nname = {name}")
            if name.startswith(".") or name.startswith("._"):
                print("    (1) Not an audio file, continuing...")
                continue  # skip .DS_Store and AppleDouble

            audio_data, sr = librosa.load(file_name, sr=None, mono=True)

            shifted_audio_data = pitch_up(audio_data)

            print(f"Now playing vanilla and pitch-shifted {name}:\n ")

            sd.play(audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

            sd.play(shifted_audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

    # Test pitch_down
    if False:

        for name in os.listdir(VANILLA_DATA_PATH):

            file_name = os.path.join(VANILLA_DATA_PATH, name)

            # Ignore non-audio data
            print(f"\nname = {name}")
            if name.startswith(".") or name.startswith("._"):
                print("    (1) Not an audio file, continuing...")
                continue  # skip .DS_Store and AppleDouble

            audio_data, sr = librosa.load(file_name, sr=None, mono=True)

            shifted_audio_data = pitch_down(audio_data)

            print(f"Now playing vanilla and pitch-shifted {name}:\n ")

            sd.play(audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

            sd.play(shifted_audio_data, samplerate=SAMPLE_RATE)
            sd.wait()
    
    #Test red noise
    if False:
        for name in os.listdir(VANILLA_DATA_PATH):

            file_name = os.path.join(VANILLA_DATA_PATH, name)

            # Ignore non-audio data
            print(f"\nname = {name}")
            if name.startswith(".") or name.startswith("._"):
                print("    (1) Not an audio file, continuing...")
                continue  # skip .DS_Store and AppleDouble

            audio_data, sr = librosa.load(file_name, sr=None, mono=True)
            noisy_audio = add_noise(audio_data)

            print(f"Now playing vanilla and noisy {name}:\n ")

            sd.play(audio_data, samplerate=SAMPLE_RATE)
            sd.wait()

            sd.play(noisy_audio, samplerate=SAMPLE_RATE)
            sd.wait()

    # Sanitize all data points per label
    labels = ["red", "green", "blue", "white", "off", "time", "temperature", "unknown", "noise"]
    if False:
        base_path = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data"

        for l in labels:
            print(f"\n---Sanitizing {l}: \n")
            curr_label_path = os.path.join(base_path, l)
            VANILLA_DATA_PATH = curr_label_path
            sanitize_vanilla_dataset()

    # Find largest audio data per label
    size = 0
    if False:
        base_path = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data"
        largest_lengths = []
        for l in labels:
            curr_label_path = os.path.join(base_path, l)
            largest_len = find_largest_length(dir_name=curr_label_path)
            largest_lengths.append(int(largest_len))
        print("\n Printing largest lengths: \n",largest_lengths)
        print(f"In order: {sorted(largest_lengths)}") # [11025, 15434, 15434, 18301, 18742, 19845, 23624, 35676, 66150]
        size = sorted(largest_lengths)[-2]
        print(f"Size = {size}") # 35676 (from unknown)


    # Force standard size in all files in the data
    if False:
        base_path = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Full_Data"

        for l in labels:
            curr_label_path = os.path.join(base_path, l)
            force_standard_size(dirname=curr_label_path, size=int(35676*1.1)) # increase size by 10% for safety
            # All files will be of size 39243.

    # Augment training dataset
    if False:
        augment_data_set()

    print("All good")
    