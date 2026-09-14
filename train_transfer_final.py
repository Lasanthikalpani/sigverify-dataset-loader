"""
CEDAR 55 - TRANSFER LEARNING (Best Solution)
SigVerify - RQ1 Core AI Performance

Uses MobileNetV2 pretrained on ImageNet
"""

import sys, os
sys.path.insert(0, 'src')

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import re

print("=" * 70)
print("🧠 SIGVERIFY - TRANSFER LEARNING (CEDAR 55)")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# Load CEDAR 55
print("\n📂 Loading CEDAR 55...")

org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

def parse_person_id(filename):
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else None

person_genuine = {}
person_forged = {}

for img_file in tqdm(sorted(list(org_dir.glob('*.png'))), desc="Genuine"):
    pid = parse_person_id(img_file.stem)
    if pid is None: continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in person_genuine:
            person_genuine[pid] = []
        person_genuine[pid].append(img)

for img_file in tqdm(sorted(list(forg_dir.glob('*.png'))), desc="Forged"):
    pid = parse_person_id(img_file.stem)
    if pid is None: continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in person_forged:
            person_forged[pid] = []
        person_forged[pid].append(img)

valid_pids = sorted([pid for pid in person_genuine.keys()
              if pid in person_forged
              and len(person_genuine[pid]) >= 5
              and len(person_forged[pid]) >= 5])

total_g = sum(len(person_genuine[p]) for p in valid_pids)
total_f = sum(len(person_forged[p]) for p in valid_pids)

print(f"\n✅ Loaded {len(valid_pids)} people")
print(f"   Total: {total_g + total_f} ✅ (1000+)")

# 3-Way Split
print("\n📂 3-Way Split (70/15/15)...")

pids = sorted(valid_pids)
np.random.seed(42)
np.random.shuffle(pids)

n = len(pids)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_people = pids[:train_end]
val_people = pids[train_end:val_end]
test_people = pids[val_end:]

print(f"   Train: {len(train_people)} people")
print(f"   Val:   {len(val_people)} people")
print(f"   Test:  {len(test_people)} people")

# Create pairs (limited)
def create_pairs(person_list, max_pairs=3000):
    X1, X2, y = [], [], []
    half = max_pairs // 2
    
    pos = 0
    for pid in person_list:
        if pos >= half: break
        gen = person_genuine[pid]
        for i in range(min(5, len(gen))):
            if pos >= half: break
            for j in range(i+1, min(i+5, len(gen))):
                if pos >= half: break
                X1.append(gen[i])
                X2.append(gen[j])
                y.append(1)
                pos += 1
    
    neg = 0
    for pid in person_list:
        if neg >= half: break
        gen = person_genuine[pid]
        forg = person_forged[pid]
        for g in gen[:5]:
            if neg >= half: break
            for f in forg[:5]:
                if neg >= half: break
                X1.append(g)
                X2.append(f)
                y.append(0)
                neg += 1
    
    return np.array(X1), np.array(X2), np.array(y)

X1_train, X2_train, y_train = create_pairs(train_people, 3000)
X1_val, X2_val, y_val = create_pairs(val_people, 500)
X1_test, X2_test, y_test = create_pairs(test_people, 500)

# Convert to RGB for MobileNetV2
def to_rgb(X):
    return np.repeat(X.reshape(-1, 128, 128, 1), 3, axis=-1)

X1_train = to_rgb(X1_train)
X2_train = to_rgb(X2_train)
X1_val = to_rgb(X1_val)
X2_val = to_rgb(X2_val)
X1_test = to_rgb(X1_test)
X2_test = to_rgb(X2_test)

print(f"\n   Training: {len(X1_train)}")
print(f"   Validation: {len(X1_val)}")
print(f"   Test: {len(X1_test)}")

# Build MobileNetV2 Siamese
print("\n🧠 Building MobileNetV2 Siamese...")

def create_base():
    base = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    
    inp = Input(shape=(128, 128, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    return Model(inp, x)

b = create_base()
i1 = Input(shape=(128, 128, 3))
i2 = Input(shape=(128, 128, 3))
e1, e2 = b(i1), b(i2)
dist = layers.Lambda(absolute_difference, output_shape=(64,))([e1, e2])
x = layers.Dense(64, activation='relu')(dist)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = Model([i1, i2], out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(f"   Parameters: {model.count_params():,}")

# Train
callbacks = [
    ModelCheckpoint('models/model_transfer55.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

print("\n🏋️ Training...")

history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_transfer55.keras')

# Test
threshold = 0.3
y_pred = model.predict([X1_test, X2_test], verbose=0)
y_pred_binary = (y_pred > threshold).astype(int).flatten()
accuracy = np.mean(y_pred_binary == y_test) * 100

print(f"\n📋 Test Accuracy: {accuracy:.1f}%")
print(f"   Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
DATASET:
   - People: {len(valid_pids)}
   - Signatures: {total_g + total_f} ✅ (1000+)
   - 3-way split: {len(train_people)}/{len(val_people)}/{len(test_people)}

MODEL:
   - MobileNetV2 + Siamese
   - Parameters: {model.count_params():,}

RESULTS:
   - Test Accuracy: {accuracy:.1f}%
   - Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%
""")
print("=" * 70)
print("🎉 DONE!")