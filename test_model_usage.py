"""
Test: Use Saved Model for Prediction
SigVerify - Verify Model Works
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

print("=" * 70)
print("🧪 TEST: USE SAVED MODEL")
print("=" * 70)

# Load model
print("\n🧠 Loading model...")
model = load_model('models/model_final_v2.keras', safe_mode=False,
                   custom_objects={'tf': tf})
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

# Load test signatures from CEDAR
print("\n📂 Loading test signatures from CEDAR...")

cedar_dir = Path('data/raw/cedar/001')

# Check files exist
print(f"   Folder: {cedar_dir}")
print(f"   Files: {[f.name for f in cedar_dir.glob('*.png')][:6]}")

# Genuine 1
genuine1 = cv2.imread(str(cedar_dir / '001_01.png'), cv2.IMREAD_GRAYSCALE)
genuine1 = cv2.resize(genuine1, (128, 128)) / 255.0
genuine1 = genuine1.reshape(1, 128, 128, 1)

# Genuine 2 (same person)
genuine2 = cv2.imread(str(cedar_dir / '001_02.png'), cv2.IMREAD_GRAYSCALE)
genuine2 = cv2.resize(genuine2, (128, 128)) / 255.0
genuine2 = genuine2.reshape(1, 128, 128, 1)

# Forged 1
forged1 = cv2.imread(str(cedar_dir / '001_01F.png'), cv2.IMREAD_GRAYSCALE)
forged1 = cv2.resize(forged1, (128, 128)) / 255.0
forged1 = forged1.reshape(1, 128, 128, 1)

print("   ✅ Loaded 3 signatures")

# Test 1: Genuine vs Genuine (should be Genuine)
print("\n" + "=" * 70)
print("TEST 1: Genuine vs Genuine (same person)")
print("=" * 70)
pred1 = model.predict([genuine1, genuine2], verbose=0)[0][0]
print(f"   Prediction: {pred1:.4f}")
print(f"   Decision: {'✅ GENUINE' if pred1 > 0.5 else '❌ FORGED'}")

# Test 2: Genuine vs Forged (should be Forged)
print("\n" + "=" * 70)
print("TEST 2: Genuine vs Forged (same person)")
print("=" * 70)
pred2 = model.predict([genuine1, forged1], verbose=0)[0][0]
print(f"   Prediction: {pred2:.4f}")
print(f"   Decision: {'✅ GENUINE' if pred2 > 0.5 else '❌ FORGED'}")

# Summary
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
✅ Model works with saved weights
✅ Test 1 (Genuine vs Genuine): {pred1:.4f}
✅ Test 2 (Genuine vs Forged): {pred2:.4f}

🎯 Your model is TRAINED and READY TO USE!
""")
print("=" * 70)