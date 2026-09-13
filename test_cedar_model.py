"""
Test CEDAR Model
SigVerify - RQ1 Core AI Performance
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

print("=" * 70)
print("🧪 TEST CEDAR MODEL")
print("=" * 70)

# Load model
model = load_model(
    'models/model_cedar.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"✅ Model loaded! Parameters: {model.count_params():,}")

# Load CEDAR
print("\n📂 Loading CEDAR...")
cedar_dir = Path('data/raw/cedar')
people = {}

for person_dir in cedar_dir.iterdir():
    if not person_dir.is_dir():
        continue
    pid = person_dir.name
    genuine, forged = [], []
    for img_file in person_dir.glob('*.png'):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            if 'F' in img_file.stem:
                forged.append(img)
            else:
                genuine.append(img)
    if len(genuine) >= 2 and len(forged) >= 2:
        people[pid] = {'genuine': genuine, 'forged': forged}

print(f"✅ Loaded {len(people)} people\n")

# Test each person
correct_genuine = 0
correct_forged = 0
total = 0

for pid in sorted(people.keys()):
    # Test 1: Genuine vs Genuine
    pred1 = float(model.predict([
        people[pid]['genuine'][0].reshape(1, 128, 128, 1),
        people[pid]['genuine'][1].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    # Test 2: Genuine vs Forged
    pred2 = float(model.predict([
        people[pid]['genuine'][0].reshape(1, 128, 128, 1),
        people[pid]['forged'][0].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    genuine_correct = pred1 > 0.5
    forged_correct = pred2 < 0.5
    
    if genuine_correct:
        correct_genuine += 1
    if forged_correct:
        correct_forged += 1
    
    total += 2
    
    g_status = "✅" if genuine_correct else "❌"
    f_status = "✅" if forged_correct else "❌"
    
    print(f"Person {pid}: Genuine={pred1:.4f} {g_status} | Forged={pred2:.4f} {f_status}")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
✅ Genuine tests: {correct_genuine}/{len(people)} correct
✅ Forged tests:  {correct_forged}/{len(people)} correct

📊 Overall Accuracy: {(correct_genuine + correct_forged)/total*100:.1f}%
""")
print("🎉 DONE!")