import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
from pydub import AudioSegment

VANILLA_DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_dataset_copy"
AUGMENTED_DATAPATH = ""   
SAMPLE_RATE = 44100
DURATION = int(20 * SAMPLE_RATE) # Average duration in samples
BIT_RATE = 192000
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

def sanitize_vanilla_dataset(duration=DURATION):
    """
    Edit dataset in place:
      - mono
      - 44.1 kHz
      - .m4a
      - length = DURATION samples
      - normalized (peak)
    """

    print("Starting sanitation...")

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

        # Force mono
        if len(audio_data.shape) != 1:
            audio_data = np.sum(audio_data, axis=1)/2
            print(f"    Converted {file_name} to mono succesfully")

        # Resample 
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            print(f"    Converted {file_name}'s sample rate to {SAMPLE_RATE}")

        # Trim/pad to fixed duration
        cur_len = len(audio_data)
        if cur_len > duration:
            audio_data = audio_data[ :DURATION]
            print(f"    Trimmed {file_name}'s size")

        elif cur_len < duration:
            pad_size = duration - cur_len
            # Zero-pad (centered)
            left = pad_size // 2
            right = pad_size - left
            audio_data = np.pad(audio_data, (left, right), mode='constant', constant_values=0)
            print(f"    Extended {file_name}'s size")

        # Peak normalize
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            audio_data = audio_data / peak
            print(f"    Normalized {file_name}'s data")

        #Do not touch anything above this line.

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