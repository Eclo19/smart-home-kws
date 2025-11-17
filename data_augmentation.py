import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import file_chopper
from scipy.signal import lfilter, filtfilt, sosfiltfilt

VANILLA_DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/All_Data"
AUGMENTED_DATAPATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/augmented"   
SAMPLE_RATE = 22050
DURATION_S = 20
DURATION = int(DURATION_S * SAMPLE_RATE) # Average duration in samples
AUDIO_EXTS = {"wav", "m4a", "flac", "mp3", "ogg", "opus", "aiff", "aif"}


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
        

def sanitize_vanilla_dataset(duration=DURATION):
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


def augment_data_set():
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

    return


def low_pass(audio_data, path="/Users/ericoliviera/Desktop/My_Repositories/smart-home-kws/Filters/lp7.csv"):

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
    

def high_pass(audio_data, path="/Users/ericoliviera/Desktop/My_Repositories/smart-home-kws/Filters/hp8.csv"):
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

def band_pass(audio_data,path="/Users/ericoliviera/Desktop/My_Repositories/smart-home-kws/Filters/bp1.csv"):

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
    sanitize_vanilla_dataset()

    size = SAMPLE_RATE * 2 #2s
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


    #Test augmentation
    #augment_data_set()

    

    # Get toydataset good
    #size = find_largest_length(VANILLA_DATA_PATH)
    #print(f"Largest file length is ({size}), in seconds: {float(size/SAMPLE_RATE)}")

    #new_size = SAMPLE_RATE # 1 second
    #force_standard_size(VANILLA_DATA_PATH, new_size)