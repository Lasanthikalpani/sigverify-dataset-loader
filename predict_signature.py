"""
Predict Signature - Use Trained Model
SigVerify - Inference Script
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the custom function (must match training)
def absolute_difference(x):
    return tf.abs(x[0] - x[1])

print("=" * 70)
print("🔍 SIGVERIFY - SIGNATURE VERIFICATION")
print("=" * 70)

# Load model
print("\n🧠 Loading model...")
model = load_model(
    'models/model_v3.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

# Load and preprocess two signatures
def load_signature(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"   ❌ Could not load: {path}")
        return None
    img = cv2.resize(img, (128, 128)) / 255.0
    return img.reshape(1, 128, 128, 1)

# Example: Compare two signatures from CEDAR
print("\n📂 Loading signatures...")

sig1 = load_signature('data/raw/cedar/001/001_01.png')
sig2 = load_signature('data/raw/cedar/001/001_02.png')  # Same person
sig3 = load_signature('data/raw/cedar/001/001_01F.png')  # Forged

if sig1 is not None and sig2 is not None:
    # Test 1: Genuine vs Genuine
    pred1 = model.predict([sig1, sig2], verbose=0)[0][0]
    print(f"\n✅ Test 1: Genuine vs Genuine")
    print(f"   Prediction: {pred1:.4f}")
    print(f"   Decision: {'✅ GENUINE' if pred1 > 0.5 else '❌ FORGED'}")

if sig1 is not None and sig3 is not None:
    # Test 2: Genuine vs Forged
    pred2 = model.predict([sig1, sig3], verbose=0)[0][0]
    print(f"\n✅ Test 2: Genuine vs Forged")
    print(f"   Prediction: {pred2:.4f}")
    print(f"   Decision: {'✅ GENUINE' if pred2 > 0.5 else '❌ FORGED'}")

print("\n" + "=" * 70)
print("🎉 SIGNATURE VERIFICATION COMPLETE!")
print("=" * 70)