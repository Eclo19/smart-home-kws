import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import feature_extraction
import keras
import librosa
import os 

print("\nAll good\n")

SAMPLE_RATE = 22050
DATA_TYPE = np.float32
INPUT_LENGTH = 39243  # number of samples expected by your MFCC setup
MODEL_PATH = "Models/best_cnn_run4_padrobust-epoch15-valloss0.0025-valacc1.0000.keras"
TEST_DIR = "/Users/ericoliviera/Desktop/Data/smart-home-ksw/Split_data/test"
THRESH = 0.12

# Mirror the training-time constants
N_MFCC   = feature_extraction.N_MFCC
T_FRAMES = feature_extraction.T_FRAMES

def leaky_integrator(input_signal, dt, tau):
    """
    Simulates a leaky integrator over a 1D input signal.

    Args:
        input_signal (np.array): The input signal over time (1D).
        dt (float): Time step between samples (seconds).
        tau (float): Time constant of the leak (seconds).

    Returns:
        np.array: The leaky-integrated output over time.
    """
    num_steps = len(input_signal)
    output = np.zeros(num_steps, dtype=np.float32)

    alpha = 1.0 - dt / tau
    if alpha < 0:
        raise ValueError(f"dt/tau is too large: dt={dt}, tau={tau} -> alpha={alpha} < 0")

    for i in range(1, num_steps):
        output[i] = output[i-1] * alpha + input_signal[i]

    return output


def plot_signal_and_loudest_window_leaky(
    audio_data,
    sr,
    tau_ms=200.0,
    win_size=INPUT_LENGTH,
    title="Signal + Leaky Integrator Loudest Window", 
    plot=False
):
    """
    Plot a signal, its leaky-integrated energy, and highlight the region
    around the peak of the leaky integrator (loudest window). Returns
    a *fixed-length* window (win_size samples) whenever possible.
    """
    # Ensure 1D signal
    audio_data = np.asarray(audio_data, dtype=float).ravel()
    N = len(audio_data)
    t = np.arange(N) / sr

    # Parameters
    dt  = 1.0 / sr
    tau = tau_ms / 1000.0

    # 1) Compute energy
    energy = audio_data ** 2

    # 2) Run leaky integrator on energy
    leaky_out = leaky_integrator(energy, dt=dt, tau=tau)

    # 3) Find the loudest point (max of leaky-integrated energy)
    peak_idx = int(np.argmax(leaky_out))

    # 4) Choose a fixed-length window around that peak
    if N >= win_size:
        # Try to center around peak, but keep window inside [0, N - win_size]
        start = max(0, min(peak_idx - win_size // 2, N - win_size))
        end   = start + win_size
    else:
        # Recording is shorter than win_size: just use the whole signal
        start = 0
        end   = N

    t_start = start / sr
    t_end   = end / sr

    print(f"Loudest region (by leaky integrator):")
    print(f"  Peak index: {peak_idx}")
    print(f"  Window samples: [{start}, {end}) (len={end-start})")
    print(f"  Window time: {t_start:.3f}s to {t_end:.3f}s")

    # 5) Scale leaky output for plotting on same axes as signal
    if np.max(np.abs(leaky_out)) > 0:
        leaky_scaled = leaky_out / np.max(leaky_out) * np.max(np.abs(audio_data))
    else:
        leaky_scaled = leaky_out

    # 6) Plot

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(t, audio_data, label="Signal", linewidth=1)
        plt.plot(t, leaky_scaled, label="Leaky integrator (scaled)", linewidth=2)
        plt.axvspan(t_start, t_end, alpha=0.3, label="Detected loudest window")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude / (scaled curves)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 7) Extract window
    window = audio_data[start:end].astype(np.float32)

    # Fade in and fade out (just like before, but now on fixed-length window)
    num_samples = min(300, len(window) // 2)  # safe if window is short
    if num_samples > 0:
        fade_in  = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, num_samples, dtype=np.float32)
        window[:num_samples]  *= fade_in
        window[-num_samples:] *= fade_out

    return window


def zero_out(audio_data, threshold=THRESH, fade_len=300, wait_ms=80, sr=SAMPLE_RATE):
    """
    Force artificial silence at beginning/end and keep the middle region where
    |audio| >= threshold, with a linear fade in/out.

    - Anything below `threshold` is considered silence to find the true
      non-silent region.
    - We find the first and last sample above `threshold` (true nonzero region).
    - `wait_ms` (in milliseconds) defines how much time BEFORE and AFTER
      that true region is *kept* (not zeroed out), even if it is below threshold.
    - Everything before (first - wait) and after (last + wait) is set to 0.
    - Fades apply ONLY to the true nonzero part [first:last]:
        * fade-in over `fade_len` samples at the start (from `first`)
        * fade-out over `fade_len` samples at the end (up to `last`)
    """

    # Ensure float32 copy so we don't modify the original in-place
    x = np.asarray(audio_data, dtype=np.float32).copy()
    n = len(x)

    # Find indices where the signal is above threshold
    non_silent_idx = np.where(np.abs(x) >= threshold)[0]

    first = int(non_silent_idx[0])   # start of true non-silent region
    last  = int(non_silent_idx[-1])  # end of true non-silent region

    # Convert wait from ms to samples
    wait_samples = int(round(wait_ms * sr / 1000.0))

    # Extended region we keep (may include some below-threshold samples)
    ext_start = max(0, first - wait_samples)
    ext_end   = min(n - 1, last + wait_samples)

    # Zero out everything outside the extended region
    x[:ext_start] = 0.0
    x[ext_end + 1:] = 0.0

    # Length of the true active (non-silent) region
    active_len = last - first + 1

    # Fade length cannot exceed half the active region
    L = min(fade_len, active_len // 2)
    if L > 0:
        # Linear fades
        fade_in = np.linspace(0.0, 1.0, L, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, L, dtype=np.float32)

        # Apply fade in on the first L samples of the TRUE non-silent region
        x[first:first + L] *= fade_in

        # Apply fade out on the last L samples of the TRUE non-silent region
        x[last - L + 1:last + 1] *= fade_out

    return x


if __name__ == "__main__":

    # Check devices
    lst = sd.query_devices()
    print(lst)

    # Load model ONCE, like in Colab
    print(f"\nLoading model from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Loaded model succesfully.\n")

    # Label mapping (same as training script)
    label_mapping = {
        'red': 0, 'green': 1, 'blue': 2, 'white': 3, 'off': 4,
        'time': 5, 'temperature': 6, 'unknown': 7, 'noise': 8
    }
    idx_to_label = {v: k for k, v in label_mapping.items()}

    duration = 4  # seconds

    # Bool for recording. If False, load random test .wav file
    rec = True  

    while True:

        # This will hold the 1D window we actually feed to the model
        input_window = np.zeros(INPUT_LENGTH, dtype=np.float32)

        if rec:
            # --- MICROPHONE BRANCH: mimic training pipeline ---
            print("\n\nRecording from microphone...\n\n")
            # Record as 16-bit PCM at 22050 Hz, mono
            recording = sd.rec(
                frames=int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16'   # 16-bit like your .wav training data
            )
            sd.wait()
            print("Recorded audio.")
            print(f"Recording dtype = {recording.dtype}, shape = {recording.shape}")

            # Convert to float32, mono 1D
            mono = recording.ravel().astype(np.float32)
            print(f"\nMono's shape = {mono.shape} , dtype = {type(mono)} of {type(mono[0])}")

            # Use leaky-integrator windowing on this clip
            input_window = plot_signal_and_loudest_window_leaky(
                audio_data=mono,
                sr=SAMPLE_RATE,
                tau_ms=200
            )

        else:
            # --- TEST-DATA BRANCH: random .wav from TEST_DIR ---
            test_list = [f for f in os.listdir(TEST_DIR)
                         if f.lower().endswith(".wav")]
            if not test_list:
                print(f"No .wav files found in {TEST_DIR}")
                continue

            test_file = np.random.choice(test_list)
            file_path = os.path.join(TEST_DIR, test_file)

            print(f"\nUsing test file: {file_path}")

            # librosa.load returns (y, sr). Force sr=SAMPLE_RATE to match training.
            audio_data, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Use the same leaky-integrator windowing
            input_window = plot_signal_and_loudest_window_leaky(
                audio_data=audio_data.astype(np.float32),
                sr=sr,
                tau_ms=200
            )

        # ---- Common pipeline after we have input_window ----

        # Normalize exactly like training/test code: float32 + divide by max |x|
        max_val = np.max(np.abs(input_window))
        if max_val > 0:
            input_window = input_window.astype(np.float32) / max_val
        else:
            print("Silent input (max amplitude = 0). Skipping.")
            continue

        # Zero parts under threshold 
        input_window = zero_out(input_window)

        # Plot for sanity check
        plt.figure(figsize=(8, 6))
        plt.title("Captured Input Window Audio")
        t = np.arange(len(input_window)) / SAMPLE_RATE
        plt.plot(t, input_window)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

        # Optionally hear what the model is seeing
        sd.play(input_window, samplerate=SAMPLE_RATE)
        sd.wait()

        # Extract MFCCs (same function as training)
        mfccs = feature_extraction.extract_mfccs(audio_data=input_window)
        print("MFCCs shape:", mfccs.shape)

        # Enforce same shape as training code
        if mfccs.shape != (N_MFCC, T_FRAMES):
            print(f"Skipping this sample: unexpected MFCC shape {mfccs.shape}")
            continue

        # Prepare input for CNN (add batch + channel dims)
        input_matrix = mfccs.reshape(
            1,
            N_MFCC,
            T_FRAMES,
            1
        ).astype(np.float32)

        print("input_matrix shape:", input_matrix.shape, "dtype:", input_matrix.dtype)

        # Run Inference
        y_pred = model.predict(input_matrix, verbose=1)[0]
        print(f"Model Output: \n{y_pred}")

        pred_idx = int(np.argmax(y_pred))
        pred_label = idx_to_label[pred_idx]

        print(f"\n--- Prediction: {pred_label} ---\n")

