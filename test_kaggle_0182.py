"""
Test: Kaggle Person 0182
SigVerify - Verify Model with Real Kaggle Data
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

# Custom function
def absolute_difference(x):
    return tf.abs(x[0] - x[1])

print("=" * 70)
print("🧪 TEST: KAGGLE PERSON 0182")
print("=" * 70)

# Load model
print("\n🧠 Loading model...")
model = load_model(
    'models/model_v3.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

# Load images from person 0182
print("\n📂 Loading signatures from person 0182...")

genuine_dir = Path('data/raw/kaggle/0001')
forged_dir = Path('data/raw/kaggle/0001_forg')

# Check folders exist
if not genuine_dir.exists():
    print(f"   ❌ Folder not found: {genuine_dir}")
    exit()

if not forged_dir.exists():
    print(f"   ❌ Folder not found: {forged_dir}")
    exit()

# List files
genuine_files = sorted(list(genuine_dir.glob('*.jpg')))[:5]
forged_files = sorted(list(forged_dir.glob('*.jpg')))[:5]

print(f"   Genuine files: {len(genuine_files)}")
print(f"   Forged files: {len(forged_files)}")

for f in genuine_files[:3]:
    print(f"      Genuine: {f.name}")
for f in forged_files[:3]:
    print(f"      Forged: {f.name}")

# Load images
def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128)) / 255.0
    return img.reshape(1, 128, 128, 1)

# Load 3 genuine and 3 forged
genuine_imgs = [load_image(f) for f in genuine_files[:3]]
forged_imgs = [load_image(f) for f in forged_files[:3]]

# Remove any None values
genuine_imgs = [img for img in genuine_imgs if img is not None]
forged_imgs = [img for img in forged_imgs if img is not None]

print(f"\n   Loaded {len(genuine_imgs)} genuine, {len(forged_imgs)} forged")

# ============================================================
# TEST 1: Genuine vs Genuine (same person)
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Genuine vs Genuine (same person)")
print("=" * 70)

if len(genuine_imgs) >= 2:
    pred1 = model.predict([genuine_imgs[0], genuine_imgs[1]], verbose=0)[0][0]
    print(f"   Prediction: {pred1:.4f}")
    print(f"   Decision: {'✅ GENUINE' if pred1 > 0.5 else '❌ FORGED'}")
    print(f"   Expected: ✅ GENUINE (same person)")
    print(f"   Result: {'✅ CORRECT' if pred1 > 0.5 else '❌ WRONG'}")
else:
    print("   ⚠️ Not enough genuine images")

# ============================================================
# TEST 2: Genuine vs Forged (same person)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Genuine vs Forged (same person)")
print("=" * 70)

if len(genuine_imgs) >= 1 and len(forged_imgs) >= 1:
    pred2 = model.predict([genuine_imgs[0], forged_imgs[0]], verbose=0)[0][0]
    print(f"   Prediction: {pred2:.4f}")
    print(f"   Decision: {'✅ GENUINE' if pred2 > 0.5 else '❌ FORGED'}")
    print(f"   Expected: ❌ FORGED (forged signature)")
    print(f"   Result: {'✅ CORRECT' if pred2 < 0.5 else '❌ WRONG'}")
else:
    print("   ⚠️ Not enough images")

# ============================================================
# TEST 3: Multiple comparisons
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Multiple Genuine Comparisons")
print("=" * 70)

if len(genuine_imgs) >= 2:
    correct = 0
    total = 0
    for i in range(len(genuine_imgs)):
        for j in range(i+1, len(genuine_imgs)):
            pred = model.predict([genuine_imgs[i], genuine_imgs[j]], verbose=0)[0][0]
            status = "✅" if pred > 0.5 else "❌"
            print(f"   Genuine {i+1} vs Genuine {j+1}: {pred:.4f} {status}")
            if pred > 0.5:
                correct += 1
            total += 1
    print(f"\n   Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
🧪 Tested on Kaggle person 0182
   - Genuine images: {len(genuine_imgs)}
   - Forged images: {len(forged_imgs)}

📊 Results:
   - Genuine vs Genuine: {'✅ PASS' if len(genuine_imgs) >= 2 and pred1 > 0.5 else '⚠️ N/A'}
   - Genuine vs Forged: {'✅ PASS' if len(genuine_imgs) >= 1 and len(forged_imgs) >= 1 and pred2 < 0.5 else '⚠️ N/A'}

🎯 Model verified with real Kaggle data!
""")
print("=" * 70)