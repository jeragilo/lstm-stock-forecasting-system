import onnxruntime as ort
import numpy as np
import os

# ✅ Define ONNX Model Path
MODEL_PATH = "models/lstm_model.onnx"

# ✅ Check if ONNX Model Exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"\u274c ONNX model not found at {MODEL_PATH}. Train the model first!")

# ✅ Load ONNX Model
print(f"\u2705 Loading ONNX model from {MODEL_PATH}...")
session = ort.InferenceSession(MODEL_PATH)

# ✅ Generate a Test Input (Random Data)
dummy_input = np.random.rand(1, 1, 12).astype(np.float32)
h0 = np.zeros((3, 1, 1024), dtype=np.float32)
c0 = np.zeros((3, 1, 1024), dtype=np.float32)

# ✅ Run ONNX Model Inference
print("\u2705 Running ONNX Inference...")
output = session.run(None, {"input": dummy_input, "h0": h0, "c0": c0})

# ✅ Display Output
print(f"\u2705 ONNX Inference Output: {output}")

