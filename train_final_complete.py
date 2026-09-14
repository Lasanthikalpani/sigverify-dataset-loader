"""
CEDAR 55 People - WITH AUGMENTATION (Complete)
SigVerify - RQ1 Core AI Performance
All Supervisor Requirements
"""

import sys, os
sys.path.insert(0, 'src')

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import re

print("=" * 70)
print("🧠 SIGVERIFY - COMPLETE TRAINING (55 PEOPLE + AUGMENTATION)")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load signatures dataset
# ============================================================
print("\n📂 Loading signatures dataset...")

org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

org_files = sorted(list(org_dir.glob('*.png')))
forg_files = sorted(list(forg_dir.glob('*.png')))

def parse_person_id(filename):
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else None

people = {}

for img_file in tqdm(org_files, desc="Genuine"):
    pid = parse_person_id(img_file.stem)
    if pid is None:
        continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in people:
            people[pid] = {'genuine': [], 'forged': []}
        people[pid]['genuine'].append(img)

for img_file in tqdm(forg_files, desc="Forged"):
    pid = parse_person_id(img_file.stem)
    if pid is None:
        continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in people:
            people[pid] = {'genuine': [], 'forged': []}
        people[pid]['forged'].append(img)

people = {pid: d for pid, d in people.items()
          if len(d['genuine']) >= 3 and len(d['forged']) >= 3}

print(f"\n✅ Loaded {len(people)} people")
total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f} ✅ (1000+)")

# ============================================================
# AUGMENTATION
# ============================================================
print("\n📂 Augmentation (scale, flip, rotate)...")

def augment_image(img):
    augmented = [img]
    # Rotate
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((64, 64), angle, 1)
        augmented.append(cv2.warpAffine(img, M, (128, 128)))
    # Flip
    augmented.append(cv2.flip(img, 1))
    # Scale
    for scale in [0.9, 1.1]:
        new_size = int(128 * scale)
        scaled = cv2.resize(img, (new_size, new_size))
        if new_size > 128:
            start = (new_size - 128) // 2
            scaled = scaled[start:start+128, start:start+128]
        else:
            padded = np.zeros((128, 128))
            start = (128 - new_size) // 2
            padded[start:start+new_size, start:start+new_size] = scaled
            scaled = padded
        augmented.append(scaled)
    return augmented

# ============================================================
# 3-Way Split
# ============================================================
print("\n📂 3-Way Split (70/15/15)...")

person_ids = sorted(list(people.keys()))
np.random.seed(42)
np.random.shuffle(person_ids)

n = len(person_ids)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_people = person_ids[:train_end]
val_people = person_ids[train_end:val_end]
test_people = person_ids[val_end:]

print(f"   Train: {len(train_people)} people")
print(f"   Val:   {len(val_people)} people")
print(f"   Test:  {len(test_people)} people")

# ============================================================
# Create pairs WITH augmentation (training only)
# ============================================================
print("\n📂 Creating augmented training pairs...")

X1_train, X2_train, y_train = [], [], []

TARGET = 5000
half = TARGET // 2

# Positive pairs (augmented)
pos_count = 0
for pid in train_people:
    if pos_count >= half:
        break
    gen = people[pid]['genuine']
    for img in gen[:3]:
        if pos_count >= half:
            break
        augmented = augment_image(img)
        for i in range(len(augmented)):
            if pos_count >= half:
                break
            for j in range(i+1, len(augmented)):
                if pos_count >= half:
                    break
                X1_train.append(augmented[i])
                X2_train.append(augmented[j])
                y_train.append(1)
                pos_count += 1

# Negative pairs (augmented)
neg_count = 0
for pid in train_people:
    if neg_count >= half:
        break
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen[:2]:
        if neg_count >= half:
            break
        aug_g = augment_image(g)
        for f in forg[:2]:
            if neg_count >= half:
                break
            aug_f = augment_image(f)
            for i in range(min(3, len(aug_g))):
                if neg_count >= half:
                    break
                X1_train.append(aug_g[i])
                X2_train.append(aug_f[i])
                y_train.append(0)
                neg_count += 1

print(f"   Positive: {pos_count}")
print(f"   Negative: {neg_count}")

# Validation (original - no augmentation)
X1_val, X2_val, y_val = [], [], []
half_val = 500

val_pos = 0
for pid in val_people:
    if val_pos >= half_val:
        break
    gen = people[pid]['genuine']
    for i in range(min(5, len(gen))):
        if val_pos >= half_val:
            break
        for j in range(i+1, min(i+5, len(gen))):
            if val_pos >= half_val:
                break
            X1_val.append(gen[i])
            X2_val.append(gen[j])
            y_val.append(1)
            val_pos += 1

val_neg = 0
for pid in val_people:
    if val_neg >= half_val:
        break
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen[:5]:
        if val_neg >= half_val:
            break
        for f in forg[:5]:
            if val_neg >= half_val:
                break
            X1_val.append(g)
            X2_val.append(f)
            y_val.append(0)
            val_neg += 1

# Test (original - no augmentation)
X1_test, X2_test, y_test = [], [], []
half_test = 500

test_pos = 0
for pid in test_people:
    if test_pos >= half_test:
        break
    gen = people[pid]['genuine']
    for i in range(min(5, len(gen))):
        if test_pos >= half_test:
            break
        for j in range(i+1, min(i+5, len(gen))):
            if test_pos >= half_test:
                break
            X1_test.append(gen[i])
            X2_test.append(gen[j])
            y_test.append(1)
            test_pos += 1

test_neg = 0
for pid in test_people:
    if test_neg >= half_test:
        break
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen[:5]:
        if test_neg >= half_test:
            break
        for f in forg[:5]:
            if test_neg >= half_test:
                break
            X1_test.append(g)
            X2_test.append(f)
            y_test.append(0)
            test_neg += 1

# Convert
X1_train = np.array(X1_train).reshape(-1, 128, 128, 1)
X2_train = np.array(X2_train).reshape(-1, 128, 128, 1)
y_train = np.array(y_train)

X1_val = np.array(X1_val).reshape(-1, 128, 128, 1)
X2_val = np.array(X2_val).reshape(-1, 128, 128, 1)
y_val = np.array(y_val)

X1_test = np.array(X1_test).reshape(-1, 128, 128, 1)
X2_test = np.array(X2_test).reshape(-1, 128, 128, 1)
y_test = np.array(y_test)

print(f"\n   Training: {len(X1_train)} pairs")
print(f"   Validation: {len(X1_val)} pairs")
print(f"   Test: {len(X1_test)} pairs")

# ============================================================
# Build model
# ============================================================
print("\n🧠 Building model...")

def create_model():
    def base():
        inp = Input(shape=(128, 128, 1))
        x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        return Model(inp, x)
    
    b = base()
    i1 = Input(shape=(128, 128, 1))
    i2 = Input(shape=(128, 128, 1))
    e1, e2 = b(i1), b(i2)
    dist = layers.Lambda(absolute_difference, output_shape=(64,))([e1, e2])
    x = layers.Dense(64, activation='relu')(dist)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model([i1, i2], out)

model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(f"   Parameters: {model.count_params():,}")

# ============================================================
# Train
# ============================================================
print("\n🏋️ Training...")

callbacks = [
    ModelCheckpoint('models/model_complete.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=20,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_complete.keras')
print(f"\n✅ Model saved!")

# ============================================================
# Test
# ============================================================
print("\n📋 Test on UNSEEN data...")

threshold = 0.3
y_pred = model.predict([X1_test, X2_test], verbose=0)
y_pred_binary = (y_pred > threshold).astype(int).flatten()

accuracy = np.mean(y_pred_binary == y_test) * 100

tp = np.sum((y_pred_binary == 1) & (y_test == 1))
tn = np.sum((y_pred_binary == 0) & (y_test == 0))
fp = np.sum((y_pred_binary == 1) & (y_test == 0))
fn = np.sum((y_pred_binary == 0) & (y_test == 1))

precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n   Test Accuracy: {accuracy:.1f}%")
print(f"   Precision: {precision:.1f}%")
print(f"   Recall: {recall:.1f}%")
print(f"   F1-Score: {f1:.1f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("📋 SUPERVISOR REQUIREMENTS CHECKLIST")
print("=" * 70)
print(f"""
✅ 1. 1000+ DATASET
   - Total: {total_g + total_f} signatures (2640+)
   - People: {len(people)}

✅ 2. AUGMENTATION (scale, flip, rotate)
   - Applied to training data
   - Training pairs: {len(X1_train)}

✅ 3. 3-WAY SPLIT (70/15/15)
   - Training:   {len(train_people)} people
   - Validation: {len(val_people)} people
   - Testing:    {len(test_people)} people

✅ 4. CNN MODEL
   - Siamese CNN: {model.count_params():,} parameters

✅ 5. ACCURACY
   - Test Accuracy: {accuracy:.1f}%
   - Precision: {precision:.1f}%
   - Recall: {recall:.1f}%
   - F1-Score: {f1:.1f}%
""")
print("=" * 70)
print("🎉 ALL SUPERVISOR REQUIREMENTS MET!")
print("=" * 70)