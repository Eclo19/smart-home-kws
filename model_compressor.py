import os
import tensorflow as tf
import numpy as np

# Optional: make TF a bit quieter and avoid some optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---- Paths ----
K_MODEL_PATH = "Models/best_cnn_run4_padrobust-epoch15-valloss0.0025-valacc1.0000.keras"
TFLITE_OUT  = "Models/kws_model.tflite"

# Your CNN input shape (batch, 32, 154, 1)
N_MFCC = 32
T_FRAMES = 154

print("\nLoading Keras model...")
model = tf.keras.models.load_model(K_MODEL_PATH, compile=False)
print(model.summary())

# ---- Build a concrete function for the model ----
print("\nBuilding concrete function for TFLite converter...")
# This wraps the model in a tf.function
@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, N_MFCC, T_FRAMES, 1], dtype=tf.float32)
])
def serving_fn(x):
    return model(x)

concrete_func = serving_fn.get_concrete_function()

# ---- Create converter from the concrete function ----
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# Be explicit about supported ops (only builtins)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# If these attrs exist in your TF build, you can also try:
# converter.experimental_new_converter = True
# converter._experimental_lower_to_saved_model = False

print("\nConverting to TFLite...")
tflite_model = converter.convert()
print("Conversion succeeded!")

# ---- Save .tflite file ----
os.makedirs(os.path.dirname(TFLITE_OUT), exist_ok=True)
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)

print(f"\nTFLite model written to: {TFLITE_OUT}")
print(f"Size on disk: {os.path.getsize(TFLITE_OUT) / 1024:.1f} kB\n")

print("\n\nAll good.\n\n")
