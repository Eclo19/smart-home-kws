import numpy as np
from typing import Optional
import tflite_runtime.interpreter as tflite
import librosa
import sounddevice as sd
import RPi.GPIO as GPIO
import datetime

import warnings

# Hide the “smallest subnormal is zero” warnings from numpy.getlimits
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal",
    category=UserWarning,
    module="numpy.core.getlimits"
)


# Audio Processing / model I/O
SAMPLE_RATE = 22050           # Must match training
DATA_TYPE   = np.float32
LENGTH_S    = 4               # seconds
BUFFER_LEN  = LENGTH_S * SAMPLE_RATE
MODEL_PATH = "cnn_kws_model.tflite"
N_MFCC   = 32
T_FRAMES = 154
INPUT_LENGTH = 39243 # samples
THRESH        = 0.12            # Silence threshold for zero_out()

# --- GPIO pin setup ---
RED_PIN = 10
GREEN_PIN = 9
BLUE_PIN = 11

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

def extract_mfccs(
    audio_data: np.ndarray,
    n_mfcc: int = 32,
    # STFT / framing
    n_fft: int = 512,
    hop_length: int = 256,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    # Mel / magnitude
    power: float = 2.0,
    n_mels: int = 64,
    htk: bool = False,
    fmin: float = 20.0,
    fmax: Optional[float] = 16000.0,
    # dB scaling
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    # DCT / MFCC post-processing
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: int = 0,
    # sampling rate override
    sr: Optional[int] = None,
) -> np.ndarray:
    """Return MFCCs of shape (n_mfcc, T)."""
    sr_eff = SAMPLE_RATE if sr is None else sr
    nyq = 0.5 * sr_eff
    fmax_eff = min(fmax if fmax is not None else nyq, nyq)

    # 1) Mel spectrogram (power)
    S_mel = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr_eff,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        power=power,
        n_mels=n_mels,
        htk=htk,
        fmin=fmin,
        fmax=fmax_eff,
    )

    # 2) Convert to dB
    S_db = librosa.power_to_db(S_mel, ref=ref, amin=amin, top_db=top_db)

    # 3) MFCCs from Mel dB
    mfccs = librosa.feature.mfcc(
        S=S_db,
        n_mfcc=n_mfcc,
        dct_type=dct_type,
        norm=norm,
        lifter=lifter,
    )
    return mfccs

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

# LED 

# Set pin 
GPIO.setmode(GPIO.BCM)

# Setup pins as outputs
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)

# --- Set up PWM on each channel ---
FREQ = 1000  # 1 kHz PWM frequency
red = GPIO.PWM(RED_PIN, FREQ)
green = GPIO.PWM(GREEN_PIN, FREQ)
blue = GPIO.PWM(BLUE_PIN, FREQ)

# Start PWM with 0% duty cycle (off)
red.start(0)
green.start(0)
blue.start(0)

def set_color(r, g, b):
    """Set RGB LED color using 0-100% duty cycle for each channel"""
    red.ChangeDutyCycle(r)
    green.ChangeDutyCycle(g)
    blue.ChangeDutyCycle(b)


if __name__ == "__main__":
        
    print("Starting main routine...\n")

    # Preallocate buffer: 1D mono signal
    buffer = np.zeros(BUFFER_LEN, dtype=DATA_TYPE)

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

    # --- Print devices ---
    devices = sd.query_devices()
    print("Available devices:\n")
    for idx, dev in enumerate(devices):
        print(f"{idx}: {dev['name']}  (max input channels: {dev['max_input_channels']})")

    # Default device info
    default_input, default_output = sd.default.device
    print("\nDefault input device index:", default_input)
    print("Default input device info:", devices[default_input])

    while True:

        # --- Record into buffer using InputStream ---
        print(f"\nRecording {LENGTH_S} seconds of audio at {SAMPLE_RATE} Hz...\n")

        frames_to_read = BUFFER_LEN
        offset = 0
        blocksize = 1024  # you can tweak this

        # Open an InputStream on the default input device
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=DATA_TYPE,
        ) as stream:

            while offset < frames_to_read:
                # How many frames to grab in this iteration
                frames = min(blocksize, frames_to_read - offset)

                data, overflowed = stream.read(frames)
                if overflowed:
                    print("Warning: overflow detected while recording")

                # data.shape = (frames, channels), channels=1 here
                buffer[offset:offset+frames] = data[:, 0]
                offset += frames

        print("Recording done. ")

        # Get window
        input_window = extract_loudest_window_leaky(
            audio_data=buffer,
            sr=SAMPLE_RATE,
            tau_ms=200.0,
        )

        print("Got window.")

        # Normalize 
        print("Normalizing...")
        max_val = np.max(np.abs(input_window))
        if max_val > 0:
            input_window = input_window / max_val
            print("Normalized.")
        else:
            print("Silent input (max amplitude = 0). Skipping.")

        # Zero out 
        print("Zeroing out...")
        input_window = zero_out(input_window, threshold=THRESH, sr=SAMPLE_RATE)
        print("Zerod out.")
        
        print("Getting MFCCs...")
        # ---- MFCC extraction ----
        mfccs = extract_mfccs(audio_data=input_window)
        print("Got MFCCs.")

        if mfccs.shape != (N_MFCC, T_FRAMES):
            print(f"Skipping: unexpected MFCC shape {mfccs.shape}")
            

        print(f"MFCCs shape = {mfccs.shape}")

        # Shape tensor
        input_matrix = mfccs.reshape(
            1, N_MFCC, T_FRAMES, 1
        ).astype(DATA_TYPE)

        print("input_matrix shape:", input_matrix.shape)

        # Set tensor & run inference
        print("Setting tensor...")
        interpreter.set_tensor(input_details[0]["index"], input_matrix)
        interpreter.invoke()
        print("Set.")

    # Get output probabilities
        y_pred = interpreter.get_tensor(output_details[0]["index"])[0]
        print("Ran inference successfully.")

        pred_idx = int(np.argmax(y_pred))
        pred_label = idx_to_label[pred_idx]

        print("\nModel output probabilities:", y_pred)
        print(f"\n--- PREDICTION: {pred_label} ---\n")

        # Big switch to control peripherals

        # Red
        if pred_idx == 0:
            set_color(100, 0, 0)

        # Green
        elif pred_idx == 1:
            set_color(0, 100, 0)
        
        # Blue
        elif pred_idx == 2:
            set_color(0, 0, 100)
        
        # White
        elif pred_idx == 3:
            set_color(100, 100, 100)

        # Off
        elif pred_idx == 4:
            set_color(0, 0, 0)
            set_color(0, 0, 0)
            set_color(0, 0, 0)

        # Time
        elif pred_idx == 5:
            print(f"\nCurrent time is{datetime.datetime.now()}")





