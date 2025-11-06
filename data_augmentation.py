import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import file_chopper

VANILLA_DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_dataset_4"
AUGMENTED_DATAPATH = ""   
SAMPLE_RATE = 44100
DURATION_S = 20
DURATION = int(DURATION_S * SAMPLE_RATE) # Average duration in samples
AUDIO_EXTS = {"wav", "m4a", "flac", "mp3", "ogg", "opus", "aiff", "aif"}

def find_largest_length(dir_name):

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

        size = 0
        #if stereo, keep track of the channel's length
        if len(audio_data.shape) != 1:
            size = audio_data.shape[1]/sr
        
        #If mono, simply get the length
        else:
            size = len(audio_data)/sr
        
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
      - 44.1 kHz
      - .m4a
      - length = DURATION samples
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
        audio_data, sr = librosa.load(file_name)
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

if __name__ == "__main__":
    sanitize_vanilla_dataset()
    #file_chopper.parse()
    size = SAMPLE_RATE * 2 #2s
    force_standard_size('/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_dataset_4', size)
