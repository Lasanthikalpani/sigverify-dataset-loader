"""
Predict New Signature - Use Trained Model
SigVerify - Deployment Script
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================
# Custom function (required for loading)
# ============================================================
def absolute_difference(x):
    return tf.abs(x[0] - x[1])

print("=" * 70)
print("🔍 SIGVERIFY - PREDICT NEW SIGNATURE")
print("=" * 70)

# ============================================================
# STEP 1: Load trained model
# ============================================================
print("\n🧠 Loading trained model...")
model = load_model(
    'models/model_transfer_aug.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

# ============================================================
# STEP 2: Define signature loading function
# ============================================================
def load_signature(path):
    """Load and preprocess a signature image"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"   ❌ Could not load: {path}")
        return None
    img = cv2.resize(img, (128, 128)) / 255.0
    # Convert to RGB (3 channels for MobileNetV2)
    img = np.repeat(img.reshape(128, 128, 1), 3, axis=-1)
    return img.reshape(1, 128, 128, 3)

# ============================================================
# STEP 3: Predict a pair
# ============================================================
def predict_pair(path1, path2, threshold=0.3):
    """Compare two signatures"""
    sig1 = load_signature(path1)
    sig2 = load_signature(path2)
    
    if sig1 is None or sig2 is None:
        return None
    
    # Predict
    pred = float(model.predict([sig1, sig2], verbose=0)[0][0])
    decision = "GENUINE" if pred > threshold else "FORGED"
    
    return {
        'prediction': pred,
        'decision': decision,
        'confidence': abs(pred - 0.5) * 2
    }

# ============================================================
# STEP 4: Test with sample signatures
# ============================================================
print("\n" + "=" * 70)
print("📋 TESTING WITH NEW DATA")
print("=" * 70)

# Test 1: Genuine vs Genuine (should be GENUINE)
print("\n--- Test 1: Genuine vs Genuine (same person) ---")
result = predict_pair(
    'data/raw/signatures/full_org/original_1_1.png',
    'data/raw/signatures/full_org/original_1_2.png'
)

if result:
    print(f"   Prediction: {result['prediction']:.4f}")
    print(f"   Decision: {'✅ ' + result['decision'] if result['decision'] == 'GENUINE' else '❌ ' + result['decision']}")
    print(f"   Confidence: {result['confidence']:.2f}")

# Test 2: Genuine vs Forged (should be FORGED)
print("\n--- Test 2: Genuine vs Forged (same person) ---")
result = predict_pair(
    'data/raw/signatures/full_org/original_1_1.png',
    'data/raw/signatures/full_forg/forgeries_1_1.png'
)

if result:
    print(f"   Prediction: {result['prediction']:.4f}")
    print(f"   Decision: {'✅ ' + result['decision'] if result['decision'] == 'FORGED' else '❌ ' + result['decision']}")
    print(f"   Confidence: {result['confidence']:.2f}")

# Test 3: Genuine vs Different Person (should be FORGED)
print("\n--- Test 3: Genuine person 1 vs Genuine person 2 ---")
result = predict_pair(
    'data/raw/signatures/full_org/original_1_1.png',
    'data/raw/signatures/full_org/original_2_1.png'
)

if result:
    print(f"   Prediction: {result['prediction']:.4f}")
    print(f"   Decision: {'✅ ' + result['decision'] if result['decision'] == 'FORGED' else '❌ ' + result['decision']}")
    print(f"   Confidence: {result['confidence']:.2f}")

# ============================================================
# STEP 5: Predict YOUR OWN signature
# ============================================================
print("\n" + "=" * 70)
print("📋 PREDICT YOUR OWN SIGNATURE")
print("=" * 70)

print("""
To predict your own signature:

1. Save two signature images:
   - Reference: your_known_signature.png
   - Questioned: your_new_signature.png

2. Add to this script:

   result = predict_pair(
       'path/to/reference.png',
       'path/to/questioned.png'
   )

   if result:
       print(f"Prediction: {result['prediction']:.4f}")
       print(f"Decision: {result['decision']}")
""")

print("\n" + "=" * 70)
print("✅ PREDICTION COMPLETE!")
print("=" * 70)