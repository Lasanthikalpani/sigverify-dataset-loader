"""
Test Full CEDAR Model
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import re

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

print("=" * 70)
print("TEST FULL CEDAR MODEL")
print("=" * 70)

# Load model
model = load_model(
    'models/model_cedar_full.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"Model loaded! Parameters: {model.count_params():,}")

# Load data
org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

def parse_person_id(filename):
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else None

people = {}

for img_file in sorted(list(org_dir.glob('*.png'))):
    pid = parse_person_id(img_file.stem)
    if pid is None:
        continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in people:
            people[pid] = {'genuine': [], 'forged': []}
        people[pid]['genuine'].append(img)

for img_file in sorted(list(forg_dir.glob('*.png'))):
    pid = parse_person_id(img_file.stem)
    if pid is None:
        continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in people:
            people[pid] = {'genuine': [], 'forged': []}
        people[pid]['forged'].append(img)

print(f"Loaded {len(people)} people")

# Test with different thresholds
print("\nTesting with different thresholds:\n")

for threshold in [0.2, 0.3, 0.4, 0.5]:
    correct_g = 0
    correct_f = 0
    total_g = 0
    total_f = 0
    
    for pid in people:
        if len(people[pid]['genuine']) >= 2:
            pred1 = float(model.predict([
                people[pid]['genuine'][0].reshape(1, 128, 128, 1),
                people[pid]['genuine'][1].reshape(1, 128, 128, 1)
            ], verbose=0)[0][0])
            
            if pred1 > threshold:
                correct_g += 1
            total_g += 1
        
        if len(people[pid]['genuine']) >= 1 and len(people[pid]['forged']) >= 1:
            pred2 = float(model.predict([
                people[pid]['genuine'][0].reshape(1, 128, 128, 1),
                people[pid]['forged'][0].reshape(1, 128, 128, 1)
            ], verbose=0)[0][0])
            
            if pred2 < threshold:
                correct_f += 1
            total_f += 1
    
    acc_g = correct_g / total_g * 100 if total_g > 0 else 0
    acc_f = correct_f / total_f * 100 if total_f > 0 else 0
    overall = (correct_g + correct_f) / (total_g + total_f) * 100
    
    print(f"   Threshold {threshold}: Genuine={acc_g:.1f}%, Forged={acc_f:.1f}%, Overall={overall:.1f}%")

print("\nDONE!")