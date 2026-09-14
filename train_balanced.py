"""
CEDAR Training - BALANCED (5000 pairs)
SigVerify - RQ1 Core AI Performance
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
print("🧠 SIGVERIFY - BALANCED TRAINING (5000 pairs)")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load CEDAR
# ============================================================
print("\n📂 Loading CEDAR...")

org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

def parse_person_id(filename):
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else None

people = {}

for img_file in tqdm(sorted(list(org_dir.glob('*.png'))), desc="Genuine"):
    pid = parse_person_id(img_file.stem)
    if pid is None: continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in people:
            people[pid] = {'genuine': [], 'forged': []}
        people[pid]['genuine'].append(img)

for img_file in tqdm(sorted(list(forg_dir.glob('*.png'))), desc="Forged"):
    pid = parse_person_id(img_file.stem)
    if pid is None: continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in people:
            people[pid] = {'genuine': [], 'forged': []}
        people[pid]['forged'].append(img)

print(f"\n✅ Loaded {len(people)} people")
total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f} ✅ (1000+)")

# ============================================================
# 3-Way Split
# ============================================================
print("\n📂 3-Way Split by People...")

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
# Create LIMITED pairs (5000 total)
# ============================================================
print("\n📂 Creating limited pairs (5000)...")

X1_train, X2_train, y_train = [], [], []

TARGET_POSITIVE = 2500
TARGET_NEGATIVE = 2500

# Positive pairs
pos_count = 0
for pid in train_people:
    if pos_count >= TARGET_POSITIVE:
        break
    gen = people[pid]['genuine']
    for i in range(len(gen)):
        if pos_count >= TARGET_POSITIVE:
            break
        for j in range(i+1, min(i+5, len(gen))):
            if pos_count >= TARGET_POSITIVE:
                break
            X1_train.append(gen[i])
            X2_train.append(gen[j])
            y_train.append(1)
            pos_count += 1

# Negative pairs
neg_count = 0
for pid in train_people:
    if neg_count >= TARGET_NEGATIVE:
        break
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen[:10]:
        if neg_count >= TARGET_NEGATIVE:
            break
        for f in forg[:5]:
            if neg_count >= TARGET_NEGATIVE:
                break
            X1_train.append(g)
            X2_train.append(f)
            y_train.append(0)
            neg_count += 1

print(f"   Positive: {pos_count}")
print(f"   Negative: {neg_count}")

X1_train = np.array(X1_train).reshape(-1, 128, 128, 1)
X2_train = np.array(X2_train).reshape(-1, 128, 128, 1)
y_train = np.array(y_train)

idx = np.random.permutation(len(X1_train))
X1_train, X2_train, y_train = X1_train[idx], X2_train[idx], y_train[idx]

print(f"   Total training pairs: {len(X1_train)}")

# Validation (limited)
X1_val, X2_val, y_val = [], [], []

val_pos = 0
for pid in val_people:
    if val_pos >= 250:
        break
    gen = people[pid]['genuine']
    for i in range(min(3, len(gen))):
        for j in range(i+1, min(i+3, len(gen))):
            if val_pos >= 250:
                break
            X1_val.append(gen[i])
            X2_val.append(gen[j])
            y_val.append(1)
            val_pos += 1

val_neg = 0
for pid in val_people:
    if val_neg >= 250:
        break
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen[:5]:
        if val_neg >= 250:
            break
        for f in forg[:5]:
            if val_neg >= 250:
                break
            X1_val.append(g)
            X2_val.append(f)
            y_val.append(0)
            val_neg += 1

X1_val = np.array(X1_val).reshape(-1, 128, 128, 1)
X2_val = np.array(X2_val).reshape(-1, 128, 128, 1)
y_val = np.array(y_val)

print(f"   Validation pairs: {len(X1_val)}")

# Test (limited)
X1_test, X2_test, y_test = [], [], []

test_pos = 0
for pid in test_people:
    if test_pos >= 300:
        break
    gen = people[pid]['genuine']
    for i in range(min(3, len(gen))):
        for j in range(i+1, min(i+3, len(gen))):
            if test_pos >= 300:
                break
            X1_test.append(gen[i])
            X2_test.append(gen[j])
            y_test.append(1)
            test_pos += 1

test_neg = 0
for pid in test_people:
    if test_neg >= 300:
        break
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen[:5]:
        if test_neg >= 300:
            break
        for f in forg[:5]:
            if test_neg >= 300:
                break
            X1_test.append(g)
            X2_test.append(f)
            y_test.append(0)
            test_neg += 1

X1_test = np.array(X1_test).reshape(-1, 128, 128, 1)
X2_test = np.array(X2_test).reshape(-1, 128, 128, 1)
y_test = np.array(y_test)

print(f"   Test pairs: {len(X1_test)}")

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
    ModelCheckpoint('models/model_balanced.keras',
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

model.save('models/model_balanced.keras')
print(f"\n✅ Model saved!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# Test
# ============================================================
print("\n📋 Final Test...")

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
print("📋 SUMMARY")
print("=" * 70)
print(f"""
DATASET:
   - Total: {total_g + total_f} signatures ✅ (1000+)
   - People: {len(people)}

SPLIT (70/15/15):
   - Train: {len(train_people)} people
   - Val:   {len(val_people)} people
   - Test:  {len(test_people)} people

PAIRS:
   - Training: {len(X1_train)}
   - Validation: {len(X1_val)}
   - Test: {len(X1_test)}

RESULTS:
   - Test Accuracy: {accuracy:.1f}%
   - Precision: {precision:.1f}%
   - Recall: {recall:.1f}%
   - F1-Score: {f1:.1f}%
""")
print("=" * 70)
print("🎉 DONE!")
print("=" * 70)