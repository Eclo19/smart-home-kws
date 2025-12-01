import numpy as np
import feature_extraction_pi as feature_extraction
import pyaudio
import ctypes
from ctypes import cdll, CFUNCTYPE, c_char_p, c_int
import tflite_runtime.interpreter as tflite

# ==== Global constants ====

# Audio / model I/O
SAMPLE_RATE   = 22050           # Must match training
INPUT_LENGTH  = 39243           # Length used when building MFCCs
THRESH        = 0.12            # Silence threshold for zero_out()
DATA_TYPE     = np.float32

# PyAudio / capture settings
CHUNK          = 1024
FORMAT         = pyaudio.paInt16
CHANNELS       = 1
RATE           = SAMPLE_RATE    # Keep this equal to SAMPLE_RATE
RECORD_SECONDS = 4              # Duration of each capture window (seconds)

# TFLite model path on the Pi (same dir as this script)
MODEL_PATH = "cnn_kws_model.tflite"

# Mirror training-time MFCC dimensions
N_MFCC   = feature_extraction.N_MFCC
T_FRAMES = feature_extraction.T_FRAMES


# Helpers

def leaky_integrator(input_signal, dt, tau):
    """
    Simple leaky integrator over a 1D input signal.
    V[i] = V[i-1] * (1 - dt/tau) + input_signal[i]
    """
    x = np.asarray(input_signal, dtype=np.float32).ravel()
    num_steps = len(x)
    out = np.zeros(num_steps, dtype=np.float32)

    alpha = 1.0 - dt / tau
    if alpha < 0:
        raise ValueError(f"dt/tau too large: dt={dt}, tau={tau} -> alpha={alpha} < 0")

    for i in range(1, num_steps):
        out[i] = out[i - 1] * alpha + x[i]

    return out


def extract_loudest_window_leaky(
    audio_data,
    sr,
    tau_ms=200.0,
    win_size=INPUT_LENGTH,
):
    """
    Compute a leaky-integrated energy trace and return a fixed-length window
    around the loudest point. No plotting, just the window (float32).
    """
    audio_data = np.asarray(audio_data, dtype=np.float32).ravel()
    N = len(audio_data)

    if N == 0:
        # Just return zeros if something went very wrong
        return np.zeros(win_size, dtype=np.float32)

    dt  = 1.0 / sr
    tau = tau_ms / 1000.0

    # Energy
    energy = audio_data ** 2

    # Leaky integrator
    leaky_out = leaky_integrator(energy, dt=dt, tau=tau)

    # Loudest index
    peak_idx = int(np.argmax(leaky_out))

    # Choose fixed-length window around that peak
    if N >= win_size:
        start = max(0, min(peak_idx - win_size // 2, N - win_size))
        end   = start + win_size
    else:
        # If recording shorter than win_size, use whole signal and pad
        start = 0
        end   = N

    # Slice
    window = audio_data[start:end].astype(np.float32)

    # If needed, pad to win_size with zeros (centered-ish)
    if len(window) < win_size:
        pad_total = win_size - len(window)
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        window = np.pad(window, (pad_left, pad_right), mode='constant', constant_values=0.0)

    # Fade in/out
    num_samples = min(300, len(window) // 2)
    if num_samples > 0:
        fade_in  = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, num_samples, dtype=np.float32)
        window[:num_samples]  *= fade_in
        window[-num_samples:] *= fade_out

    print(f"Loudest region (by leaky integrator): peak_idx={peak_idx}, "
          f"window samples [{start}, {end}) (len={end-start})")
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

    x = np.asarray(audio_data, dtype=np.float32).copy()
    n = len(x)

    if n == 0:
        return x

    # Indices above threshold
    non_silent_idx = np.where(np.abs(x) >= threshold)[0]

    if non_silent_idx.size == 0:
        # Entire thing is below threshold → all zeros
        return np.zeros_like(x, dtype=np.float32)

    first = int(non_silent_idx[0])   # start of true non-silent region
    last  = int(non_silent_idx[-1])  # end of true non-silent region

    # Convert wait from ms to samples
    wait_samples = int(round(wait_ms * sr / 1000.0))

    # Extended region we keep
    ext_start = max(0, first - wait_samples)
    ext_end   = min(n - 1, last + wait_samples)

    # Zero out everything outside the extended region
    x[:ext_start]   = 0.0
    x[ext_end + 1:] = 0.0

    # Length of the true active (non-silent) region
    active_len = last - first + 1

    # Fade length cannot exceed half the active region
    L = min(fade_len, active_len // 2)
    if L > 0:
        fade_in  = np.linspace(0.0, 1.0, L, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, L, dtype=np.float32)

        x[first:first + L]        *= fade_in
        x[last - L + 1:last + 1]  *= fade_out

    return x



if __name__ == "__main__":

    # ALSA ERROR HANDLER (from test_microphone.py) 
    ERROR_HANDLER_FUNC = CFUNCTYPE(
        None,
        c_char_p, c_int, c_char_p, c_int, c_char_p
    )

    def py_error_handler(filename, line, function, err, fmt):
        # Swallow ALSA errors
        return

    try:
        asound = cdll.LoadLibrary("libasound.so")
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        asound.snd_lib_error_set_handler(c_error_handler)
        print("ALSA error handler installed.")
    except OSError:
        asound = None
        print("Could not load libasound.so; continuing without custom error handler.")

    # ---- LOAD TFLITE MODEL ----
    print(f"\nLoading TFLite model from: {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Loaded TFLite model successfully.\n")
    print("Input tensor details:", input_details)
    print("Output tensor details:", output_details, "\n")

    # Check we are feeding the right dtype
    model_input_dtype = input_details[0]["dtype"]
    print("Model expects input dtype:", model_input_dtype)

    # Label mapping (same as training script)
    label_mapping = {
        'red': 0, 'green': 1, 'blue': 2, 'white': 3, 'off': 4,
        'time': 5, 'temperature': 6, 'unknown': 7, 'noise': 8
    }
    idx_to_label = {v: k for k, v in label_mapping.items()}

    # ---- AUDIO + INFERENCE LOOP ----
    p = pyaudio.PyAudio()
    stream = None

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("PyAudio input stream opened.")

        while True:
            print("\nRecording from microphone...\n")

            frames = []
            num_chunks = int(RATE / CHUNK * RECORD_SECONDS)

            for _ in range(num_chunks):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            print("Recording done. Got", len(frames), "chunks.")

            # ---- Convert frames -> mono float32 ----
            audio_bytes = b"".join(frames)
            mono = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

            # ---- Leaky-integrator loudest window ----
            input_window = extract_loudest_window_leaky(
                audio_data=mono,
                sr=SAMPLE_RATE,
                tau_ms=200.0,
            )

            # ---- Normalize ----
            max_val = np.max(np.abs(input_window))
            if max_val > 0:
                input_window = input_window / max_val
            else:
                print("Silent input (max amplitude = 0). Skipping.")
                continue

            # ---- Zero-out outside active region (mimic training padding behavior) ----
            input_window = zero_out(input_window, threshold=THRESH, sr=SAMPLE_RATE)

            # ---- MFCC extraction ----
            mfccs = feature_extraction.extract_mfccs(audio_data=input_window)
            print("MFCCs shape:", mfccs.shape)

            if mfccs.shape != (N_MFCC, T_FRAMES):
                print(f"Skipping: unexpected MFCC shape {mfccs.shape}")
                continue

            # ---- Prepare batch for TFLite ----
            input_matrix = mfccs.reshape(
                1, N_MFCC, T_FRAMES, 1
            ).astype(model_input_dtype)

            print("input_matrix shape:", input_matrix.shape)

            # Set tensor & run inference
            interpreter.set_tensor(input_details[0]["index"], input_matrix)
            interpreter.invoke()

            # Get output probabilities
            y_pred = interpreter.get_tensor(output_details[0]["index"])[0]

            pred_idx = int(np.argmax(y_pred))
            pred_label = idx_to_label[pred_idx]

            print("\nModel output probabilities:", y_pred)
            print(f"--- PREDICTION: {pred_label} ---\n")

    finally:
        # Cleanup ALSA + audio
        try:
            if asound is not None:
                asound.snd_lib_error_set_handler(None)
        except Exception:
            pass

        if stream is not None and stream.is_active():
            stream.stop_stream()
            stream.close()

        p.terminate()
        print("PyAudio cleaned up.")
