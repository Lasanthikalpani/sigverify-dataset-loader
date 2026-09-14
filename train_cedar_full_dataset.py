"""
Train with FULL CEDAR Dataset (1,321 + 1,321 = 2,642 signatures)
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
print("🧠 SIGVERIFY - FULL CEDAR TRAINING")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load dataset
# ============================================================
print("\n📂 Loading signatures...")

org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

org_files = sorted(list(org_dir.glob('*.png')) + list(org_dir.glob('*.jpg')))
forg_files = sorted(list(forg_dir.glob('*.png')) + list(forg_dir.glob('*.jpg')))

print(f"   Genuine files: {len(org_files)}")
print(f"   Forged files: {len(forg_files)}")

# ============================================================
# Parse person ID from filename
# ============================================================
def parse_person_id(filename):
    """Extract person ID from filename like original_10_1 or forgeries_10_1"""
    match = re.search(r'_(\d+)_', filename)
    if match:
        return match.group(1)
    return None

# Show sample
print(f"\n   Sample genuine: {[f.name for f in org_files[:3]]}")
print(f"   Sample forged: {[f.name for f in forg_files[:3]]}")

# Test parsing
sample = org_files[0].stem
print(f"   Parsing '{sample}' -> Person ID: {parse_person_id(sample)}")

# ============================================================
# Group by person
# ============================================================
print("\n📂 Grouping by person...")

people = {}

# Load genuine
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

# Load forged
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

# Filter people with both
people = {pid: data for pid, data in people.items()
          if len(data['genuine']) >= 2 and len(data['forged']) >= 2}

print(f"\n✅ Loaded {len(people)} people")

total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f}")

# ============================================================
# Create pairs
# ============================================================
print("\n🔀 Creating pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

# 1. Positive: Same person genuine
for pid in person_ids:
    gen = people[pid]['genuine']
    for i in range(min(5, len(gen))):
        for j in range(i+1, min(i+3, len(gen))):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)

pos_count = len(X1)
print(f"   Positive: {pos_count}")

# 2. Negative: Genuine vs Forged (same person)
for pid in person_ids:
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen[:3]:
        for f in forg[:3]:
            X1.append(g)
            X2.append(f)
            y.append(0)

neg_count = len([l for l in y if l == 0])
print(f"   Negative (genuine vs forged): {neg_count}")

# 3. Negative: Different person
for _ in range(pos_count):
    p1, p2 = np.random.choice(person_ids, 2, replace=False)
    g1 = np.random.choice(len(people[p1]['genuine']))
    g2 = np.random.choice(len(people[p2]['genuine']))
    X1.append(people[p1]['genuine'][g1])
    X2.append(people[p2]['genuine'][g2])
    y.append(0)

X1 = np.array(X1).reshape(-1, 128, 128, 1)
X2 = np.array(X2).reshape(-1, 128, 128, 1)
y = np.array(y)

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
# Train
# ============================================================
print("\n🏋️ Training...")

callbacks = [
    ModelCheckpoint('models/model_cedar_full.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_cedar_full.keras')
print(f"\n✅ Model saved as 'models/model_cedar_full.keras'")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# Test
# ============================================================
print("\n📋 Testing...")

threshold = 0.3
correct_g = 0
correct_f = 0

for pid in person_ids[:50]:
    if len(people[pid]['genuine']) >= 2:
        pred1 = float(model.predict([
            people[pid]['genuine'][0].reshape(1, 128, 128, 1),
            people[pid]['genuine'][1].reshape(1, 128, 128, 1)
        ], verbose=0)[0][0])
        
        if pred1 > threshold:
            correct_g += 1
    
    if len(people[pid]['genuine']) >= 1 and len(people[pid]['forged']) >= 1:
        pred2 = float(model.predict([
            people[pid]['genuine'][0].reshape(1, 128, 128, 1),
            people[pid]['forged'][0].reshape(1, 128, 128, 1)
        ], verbose=0)[0][0])
        
        if pred2 < threshold:
            correct_f += 1

print(f"\n   Genuine correct: {correct_g}/50")
print(f"   Forged correct: {correct_f}/50")
print(f"   Overall: {(correct_g + correct_f)/100*100:.1f}%")
print("\n🎉 DONE!")