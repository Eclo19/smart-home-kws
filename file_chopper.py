import numpy as np
import os
import matplotlib.pyplot as plt
import librosa

"""
This script chops long audio samples into pre-determined length windows based 
on the signal level. It attemopts to extract relevant utterances from speech. 
"""

FULL_DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_dataset_copy2"
CHOPPED_DIR = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_dataset_copy2/chops"
SAMPLE_RATE = 22050
DURATION = 0.75 # Duration of an utterance in seconds
WAIT = 0.01 #10ms of silence before determins a word

#Define valid audio extentions
AUDIO_EXTS = {"wav", "m4a", "flac", "mp3", "ogg", "opus", "aiff", "aif"}


def parse(thresh = 0.1, wait= int(SAMPLE_RATE*0.01), duration=int(SAMPLE_RATE*DURATION), plot_first=True):
    """
    This function assumes the dataset is uniform (i.e. already sanitezed).

    Inputs:
        * thresh (float): The threshold that separates an utterance from silence [0,1]
        * wait (int): The wait time in samples before a threshold. To avoid chopping utterances
        * duration (int): How long the utterances are. Changes per window according to user input. 
    """
    # Check if chops exists. If not, create it
    if not os.path.exists(CHOPPED_DIR):
        os.makedirs(CHOPPED_DIR)

    #Loop through directory
    for name in os.listdir(FULL_DATA_PATH):
        # Ignore non-audio data
        print(f"\nname = {name}")
        if name.startswith(".") or name.startswith("._"):
            print("     Not an audio file (likely a DS file), continuing...")
            continue  # skip .DS_Store and AppleDouble

        file_name = os.path.join(FULL_DATA_PATH, name)

        # Split safely
        root, ext = os.path.splitext(file_name)
        ext = ext.lower().lstrip(".")
        print(f"    Splited name succesfully")

        if ext.lower() not in AUDIO_EXTS:
            print("      Not an audio file, continuing...")
            print(f"    ext = {ext}")
            continue

        if not os.path.isfile(file_name):
            print("     Not a valid file, continuing...")
            continue

        #Load audio data (Assumed to be normalized)
        audio_data, sr = librosa.load(file_name)
        print(f"audio_data.shape = {audio_data.shape}")
        print(f"sr = {sr}")

        # linear fade-in over first 100 samples & fade-out over last 100 samples
        N = min(wait, audio_data.shape[0])  # handle very short clips
        fade_in = np.linspace(0.0, 1.0, N, endpoint=True)
        audio_data[:N] *= fade_in
        fade_out = np.linspace(1.0, 0.0, N, endpoint=True)
        audio_data[len(audio_data) -N:] *= fade_out
        
        #Loop through audio_data and find markers
        abs_audio = np.abs(audio_data)

        markers = []

        i = max(int(wait), 1)
        while i < len(abs_audio):
            if abs_audio[i] > thresh:
                # check silence just before this point
                start = max(0, i - int(wait))
                if np.all(abs_audio[start:i] < thresh):
                    beg = start                                  # include the silence window
                    end = min(beg + int(wait) + int(duration), len(abs_audio))
                    markers.append(beg)                          # begin marker (silence start)
                    markers.append(end)                          # end marker (silence + utterance)
                    i = end                                      # skip past this whole span
                    continue
            i += 1

        if plot_first:

            # time axis in seconds
            t = np.linspace(0, (len(audio_data) - 1)/sr, len(audio_data))

            # Simple envelope (moving average of absolute value over 'wait' samples)
            wait_samp = int(wait)
            dur_samp  = int(duration)

            plt.figure(figsize=(12, 6))
            plt.plot(t, audio_data, alpha=0.5, label="waveform")

            # Threshold 
            plt.axhline(y=thresh, color="red", linestyle="--", label=f"threshold = {thresh:g}")
            plt.axhline(y=-thresh, color="red", linestyle="--", alpha=0.5)

            # Shade windows for each detection: [beg - wait, beg] = silence, [beg, beg + duration] = utterance
            for k in range(0, len(markers), 2):
                beg = markers[k]
                end = markers[k+1] if k+1 < len(markers) else min(beg + dur_samp, len(audio_data))

                # silence window checked before onset
                s0 = max(0, beg) / sr
                s1 = (beg + wait_samp) / sr
                plt.axvspan(s0, s1, color="orange", alpha=0.25, label="silence window" if k == 0 else None)

                # utterance window
                u0 = beg / sr
                u1 = end / sr
                plt.axvspan(u0, u1, color="green", alpha=0.25, label=f"utterance window ({int(duration/sr)})" if k == 0 else None)

            # Also draw the detection markers exactly
            for m in markers:
                plt.axvline(m / sr, color="green", linestyle="-", alpha=0.6)

            plt.xlabel("time (s)")
            plt.ylabel("amplitude")
            plt.title(f"Detected windows in {name}")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.show()




if __name__ == "__main__":
    parse()