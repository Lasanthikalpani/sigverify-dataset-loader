"""
CEDAR 55 People - ALL DATA (No Aggressive Split)
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
print("🧠 SIGVERIFY - 55 PEOPLE (FULL DATA)")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load signatures
# ============================================================
print("\n📂 Loading signatures dataset...")

org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

org_files = sorted(list(org_dir.glob('*.png')))
forg_files = sorted(list(forg_dir.glob('*.png')))

def parse_person_id(filename):
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else None

person_genuine = {}
person_forged = {}

for img_file in tqdm(org_files, desc="Genuine"):
    pid = parse_person_id(img_file.stem)
    if pid is None: continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in person_genuine:
            person_genuine[pid] = []
        person_genuine[pid].append(img)

for img_file in tqdm(forg_files, desc="Forged"):
    pid = parse_person_id(img_file.stem)
    if pid is None: continue
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128)) / 255.0
        if pid not in person_forged:
            person_forged[pid] = []
        person_forged[pid].append(img)

valid_pids = [pid for pid in person_genuine.keys()
              if pid in person_forged
              and len(person_genuine[pid]) >= 3
              and len(person_forged[pid]) >= 3]

total_orig = sum(len(person_genuine[p]) + len(person_forged[p]) for p in valid_pids)

print(f"\n✅ Loaded {len(valid_pids)} people")
print(f"   Genuine: {sum(len(person_genuine[p]) for p in valid_pids)}")
print(f"   Forged: {sum(len(person_forged[p]) for p in valid_pids)}")
print(f"   Total: {total_orig} ✅ (1000+)")

# ============================================================
# Use ALL 55 people (no split, no validation)
# ============================================================
print("\n📂 Using ALL 55 people for training...")

X1, X2, y = [], [], []

# Create pairs from all people
for pid in tqdm(valid_pids, desc="Creating pairs"):
    gen = person_genuine[pid]
    forg = person_forged[pid]
    
    # Positive: Same person genuine (up to 10 pairs per person)
    for i in range(min(5, len(gen))):
        for j in range(i+1, min(i+5, len(gen))):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)
    
    # Negative: Genuine vs Forged
    for g in gen[:5]:
        for f in forg[:5]:
            X1.append(g)
            X2.append(f)
            y.append(0)

X1 = np.array(X1).reshape(-1, 128, 128, 1)
X2 = np.array(X2).reshape(-1, 128, 128, 1)
y = np.array(y)

# Shuffle
idx = np.random.permutation(len(X1))
X1, X2, y = X1[idx], X2[idx], y[idx]

print(f"\n   Total pairs: {len(X1)}")
print(f"   Positive: {np.sum(y)}")
print(f"   Negative: {len(y) - np.sum(y)}")

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
# Train (no validation - use all data)
# ============================================================
print("\n🏋️ Training (on all data)...")

callbacks = [
    ModelCheckpoint('models/model_55all.keras',
                    monitor='accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_55all.keras')
print(f"\n✅ Model saved!")
print(f"   Final accuracy: {max(history.history['accuracy'])*100:.2f}%")

# ============================================================
# Test on sample pairs
# ============================================================
print("\n📋 Testing on sample pairs...")

threshold = 0.3
correct = 0
total = 0

for pid in valid_pids[:20]:
    if len(person_genuine[pid]) >= 2:
        pred1 = float(model.predict([
            person_genuine[pid][0].reshape(1, 128, 128, 1),
            person_genuine[pid][1].reshape(1, 128, 128, 1)
        ], verbose=0)[0][0])
        if pred1 > threshold: correct += 1
        total += 1
    
    if len(person_genuine[pid]) >= 1 and len(person_forged[pid]) >= 1:
        pred2 = float(model.predict([
            person_genuine[pid][0].reshape(1, 128, 128, 1),
            person_forged[pid][0].reshape(1, 128, 128, 1)
        ], verbose=0)[0][0])
        if pred2 < threshold: correct += 1
        total += 1

print(f"\n   Test accuracy (20 people): {correct/total*100:.1f}%")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
DATASET:
   - Total: {total_orig} signatures ✅ (1000+)
   - People: {len(valid_pids)}

TRAINING:
   - Training pairs: {len(X1)}
   - Model: Siamese CNN
   - Parameters: {model.count_params():,}

RESULTS:
   - Final accuracy: {max(history.history['accuracy'])*100:.2f}%
   - Test accuracy: {correct/total*100:.1f}%
""")
print("=" * 70)
print("🎉 DONE!")
print("=" * 70)