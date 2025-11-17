import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

"""
This script chops long audio samples into pre-determined length windows based 
on the signal level. It attempts to extract relevant utterances from speech. 
Each "full" recording file is parse according to specific and iterative user
input in order to determine the best utterance extraction for every recording 
setting. 

"""

FULL_DATA_PATH = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_Datasets/Full1"
CHOPPED_DIR = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Toy_Datasets/Full1/chops"
SAMPLE_RATE = 22050
DURATION = 0.75 # s. Duration of an utterance in seconds
WAIT = 0.15 # s. silence buffer before threshold for a given utterance 

#Define valid audio extentions
AUDIO_EXTS = {"wav", "m4a", "flac", "mp3", "ogg", "opus", "aiff", "aif"}

def parse(thresh = 0.1, wait= int(SAMPLE_RATE*WAIT), duration=int(SAMPLE_RATE*DURATION), plot_first=True):
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

        # Skip hidden files fast
        if name.startswith(".") or name.startswith("._"):
            print(f"\nname = {name}. Hidden/DS file, skipping…")
            continue

        # Skip noise chunks fast
        if name.split('_')[0] == "chunk":
            print(f"\nname = {name}. Noise chunk, skipping…")
            continue

        # from here on we only handle labeled speech files
        thresh_bad = True
        duration_bad = True
        markers = []            
        audio_data = None        
        sr = None

        #Loop through each file until a good threshold and duration is set
        while (thresh_bad or duration_bad):
                
            #Build full file path
            file_name = os.path.join(FULL_DATA_PATH, name)

            # Split safely
            root, ext = os.path.splitext(file_name)
            ext = ext.lower().lstrip(".")
            print(f"name '{name}")
            print(f"    Splited name succesfully")

            #Check if it has an audio file extension
            if ext.lower() not in AUDIO_EXTS:
                print("      Not an audio file, continuing...")
                print(f"    ext = {ext}")
                thresh_bad = False
                duration_bad = False
                continue
            
            #Check if path is valid
            if not os.path.isfile(file_name):
                print("     Not a valid file, continuing...")
                thresh_bad = False
                duration_bad = False
                continue

            #Load audio data (Assumed to be normalized)
            audio_data, sr = librosa.load(file_name, sr=None, mono=True)
            print(f"audio_data.shape = {audio_data.shape}")
            print(f"sr = {sr}")

            # linear fade-in over first 'N' samples & fade-out over last 'N' samples
            N = min(100, audio_data.shape[0])  # handle very short clips
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

            # Plot the splits for user input 
            if plot_first:

                # time axis in seconds
                t = np.linspace(0, (len(audio_data) - 1)/sr, len(audio_data))

                wait_samp = int(wait)
                dur_samp  = int(duration)

                plt.figure(figsize=(13, 7))
                plt.plot(t, audio_data, alpha=0.5, label="waveform")

                # Threshold 
                plt.axhline(y=thresh, color="red", linestyle="--", label=f"threshold = {thresh:g}")
                plt.axhline(y=-thresh, color="red", linestyle="--", alpha=0.5)

                # Shade windows for each detection: [beg - wait, beg] = silence, [beg, beg + duration] = utterance
                for k in range(0, len(markers), 2):
                    beg = markers[k]
                    end = markers[k+1] if k+1 < len(markers) else min(beg + dur_samp, len(audio_data))

                    # silence window display
                    s0 = max(0, beg) / sr
                    s1 = (beg + wait_samp) / sr
                    plt.axvspan(s0, s1, color="orange", alpha=0.25, label=f"silence window ({(1000*float(wait)/SAMPLE_RATE):.2f} ms)" if k == 0 else None)

                    # utterance window
                    u0 = beg / sr
                    u1 = end / sr
                    plt.axvspan(u0, u1, color="green", alpha=0.25, label=f"utterance window ({(1000*duration/sr):.2f} ms)" if k == 0 else None)

                # Also draw the detection markers exactly
                for m in markers:
                    plt.axvline(m / sr, color="green", linestyle="-", alpha=0.6)

                plt.xlabel("time (s)")
                plt.ylabel("amplitude")
                plt.title(f"Detected windows in {name}")
                plt.legend(loc="upper right")
                plt.grid(True)
                plt.minorticks_on()
                plt.tight_layout()
                plt.show()

                # Threshold prompt with robust handling
                try:
                    # Anything not '1' (including empty, letters, etc.) becomes '0' → adjust
                    thres_choice = int(input("\nIf the threshold is good, press 1. Else, press anything else to reset: ").strip() or '0')
                except Exception:
                    # Treat any parsing problem as "adjust"
                    thres_choice = 0
                finally:
                    if thres_choice == 1:
                        thresh_bad = False
                    else:
                        # Let errors propagate here if not convertible to float (as requested)
                        thresh = float(input("    New threshold value ([0,1]): ").strip())
                        thresh_bad = True

                # Duration prompt with robust handling
                try:
                    duration_choice = int(input("If the windowing is good, press 1. Else, press anything else to reset: ").strip() or '0')
                except Exception:
                    duration_choice = 0
                finally:
                    if duration_choice == 1:
                        duration_bad = False
                    else:
                        # Let errors propagate here if not convertible to float
                        duration = int(float(input("    New window value (in seconds): ").strip()) * SAMPLE_RATE)
                        duration_bad = True
            
        # Populate chops
        if markers and audio_data is not None:
            base = name.rsplit(".", 1)[0]
            try:
                label, speaker, num, full = base.split("_")
            except ValueError:
                print("------NAME ERROR------")
                print(f"File {name} is not in the proper annotation convention; skipping chops.")
            else:
                chop_count = 1
                wait_samp = int(wait)
                dur_samp  = int(duration)

                for k in range(0, len(markers), 2):
                    beg = int(markers[k])                             # start of silence+utterance
                    # We only care about the utterance chunk of exact 'duration' length:
                    u_beg = beg + wait_samp                           # skip the silence we required
                    u_end = u_beg + dur_samp

                    # Bounds check + exact length requirement
                    if 0 <= u_beg < u_end <= len(audio_data):
                        # Build chop nam,e
                        chop_name = f"{label}_{speaker}_{num}_{chop_count:02d}.wav"
                        chop_full_name = os.path.join(CHOPPED_DIR, chop_name)

                        # Add fade in and out to avoid pops
                        to_write_data = audio_data[u_beg:u_end]
                        N = min(50, to_write_data.shape[0])  # handle very short clips
                        fade_in = np.linspace(0.0, 1.0, N, endpoint=True)
                        to_write_data[:N] *= fade_in
                        fade_out = np.linspace(1.0, 0.0, N, endpoint=True)
                        to_write_data[len(to_write_data) -N:] *= fade_out

                        # Write file
                        sf.write(chop_full_name, to_write_data, SAMPLE_RATE, subtype="PCM_16")
                        print(f"Writting utterance {chop_count} of {name} in {CHOPPED_DIR}")
                        chop_count += 1



if __name__ == "__main__":
    parse()