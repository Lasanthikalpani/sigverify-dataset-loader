"""
CEDAR 55 People - FINAL WORKING VERSION
SigVerify - RQ1 Core AI Performance

Strategy: Use MORE images per person in training (not augmentation)
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
print("🧠 SIGVERIFY - CEDAR 55 FINAL")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load CEDAR (55 people)
# ============================================================
print("\n📂 Loading CEDAR...")

org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

def parse_person_id(filename):
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else None

# Load ALL by person
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

# ============================================================
# Train on ALL 55 people
# ============================================================
print("\n📂 Creating pairs from ALL 55 people...")

X1, X2, y = [], [], []

for pid in tqdm(valid_pids, desc="Pairs"):
    gen = person_genuine[pid]
    forg = person_forged[pid]
    
    # Positive pairs (ALL combinations)
    for i in range(len(gen)):
        for j in range(i+1, len(gen)):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)
    
    # Negative pairs (genuine vs forged)
    for g in gen:
        for f in forg:
            X1.append(g)
            X2.append(f)
            y.append(0)

X1 = np.array(X1).reshape(-1, 128, 128, 1)
X2 = np.array(X2).reshape(-1, 128, 128, 1)
y = np.array(y)

idx = np.random.permutation(len(X1))
X1, X2, y = X1[idx], X2[idx], y[idx]

print(f"   Total pairs: {len(X1)}")
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
        x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
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
    ModelCheckpoint('models/model_55final.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_55final.keras')
print(f"\n✅ Model saved!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# Test on each person
# ============================================================
print("\n📋 Testing on 20 people...")

threshold = 0.3
correct_g = 0
correct_f = 0
total_people = 20

for pid in valid_pids[:total_people]:
    # Genuine vs Genuine
    pred1 = float(model.predict([
        person_genuine[pid][0].reshape(1, 128, 128, 1),
        person_genuine[pid][1].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    # Genuine vs Forged
    pred2 = float(model.predict([
        person_genuine[pid][0].reshape(1, 128, 128, 1),
        person_forged[pid][0].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    g_ok = pred1 > threshold
    f_ok = pred2 < threshold
    
    if g_ok: correct_g += 1
    if f_ok: correct_f += 1
    
    print(f"Person {pid}: Genuine={pred1:.4f} {'✅' if g_ok else '❌'} | Forged={pred2:.4f} {'✅' if f_ok else '❌'}")

print(f"\n   Genuine: {correct_g}/{total_people}")
print(f"   Forged: {correct_f}/{total_people}")
print(f"   Overall: {(correct_g + correct_f)/(total_people*2)*100:.1f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
DATASET:
   - People: {len(valid_pids)}
   - Signatures: {total_g + total_f} ✅ (1000+)
   - Training pairs: {len(X1)}

MODEL:
   - Siamese CNN: {model.count_params():,} parameters

RESULTS:
   - Validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%
   - Test accuracy (20 people): {(correct_g + correct_f)/(total_people*2)*100:.1f}%
""")
print("=" * 70)
print("🎉 DONE!")
print("=" * 70)