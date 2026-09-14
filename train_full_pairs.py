"""
CEDAR Training - FULL Pairs (From All Images)
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
print("🧠 SIGVERIFY - FULL PAIRS TRAINING")
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

# Filter people with enough images
people = {pid: d for pid, d in people.items()
          if len(d['genuine']) >= 5 and len(d['forged']) >= 5}

print(f"\n✅ Loaded {len(people)} people (with >=5 images each)")

total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f}")

# ============================================================
# Split by people (80/20)
# ============================================================
print("\n📂 Splitting by people (80/20)...")

person_ids = sorted(list(people.keys()))
np.random.seed(42)
np.random.shuffle(person_ids)

split_idx = int(len(person_ids) * 0.8)
train_people = person_ids[:split_idx]
val_people = person_ids[split_idx:]

print(f"   Train people: {len(train_people)}")
print(f"   Val people: {len(val_people)}")

# ============================================================
# Create pairs (USE ALL IMAGES)
# ============================================================
print("\n📂 Creating pairs (all images)...")

X1_train, X2_train, y_train = [], [], []

for pid in train_people:
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    
    # Positive: All combinations of genuine (limit to 10 per person for speed)
    for i in range(min(len(gen), 10)):
        for j in range(i+1, min(i+3, len(gen))):
            X1_train.append(gen[i])
            X2_train.append(gen[j])
            y_train.append(1)
    
    # Negative: All combinations of genuine vs forged (limit to 10 per person)
    for g in gen[:5]:
        for f in forg[:5]:
            X1_train.append(g)
            X2_train.append(f)
            y_train.append(0)

print(f"   Before balance: Positive={sum(y_train)}, Negative={len(y_train)-sum(y_train)}")

# Balance
pos_idx = [i for i, l in enumerate(y_train) if l == 1]
neg_idx = [i for i, l in enumerate(y_train) if l == 0]
min_count = min(len(pos_idx), len(neg_idx))

keep = list(np.random.choice(pos_idx, min_count, replace=False)) + \
       list(np.random.choice(neg_idx, min_count, replace=False))

X1_train = np.array([X1_train[i] for i in keep]).reshape(-1, 128, 128, 1)
X2_train = np.array([X2_train[i] for i in keep]).reshape(-1, 128, 128, 1)
y_train = np.array([y_train[i] for i in keep])

print(f"   After balance: Positive={sum(y_train)}, Negative={len(y_train)-sum(y_train)}")
print(f"   Training pairs: {len(X1_train)}")

# ============================================================
# Validation pairs
# ============================================================
print("\n📂 Creating validation pairs...")

X1_val, X2_val, y_val = [], [], []

for pid in val_people:
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    
    # Positive (use different images)
    for i in range(min(3, len(gen))):
        for j in range(i+1, min(i+3, len(gen))):
            X1_val.append(gen[i])
            X2_val.append(gen[j])
            y_val.append(1)
    
    # Negative
    for g in gen[:2]:
        for f in forg[:2]:
            X1_val.append(g)
            X2_val.append(f)
            y_val.append(0)

X1_val = np.array(X1_val).reshape(-1, 128, 128, 1)
X2_val = np.array(X2_val).reshape(-1, 128, 128, 1)
y_val = np.array(y_val)

print(f"   Validation pairs: {len(X1_val)}")

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
    ModelCheckpoint('models/model_full.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_full.keras')
print(f"\n✅ Model saved!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# Test
# ============================================================
print("\n📋 Testing...")

threshold = 0.3
correct_g = 0
correct_f = 0
total_g = 0
total_f = 0

for pid in val_people:
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

print(f"\n   Genuine: {correct_g}/{total_g}")
print(f"   Forged: {correct_f}/{total_f}")
print(f"   Overall: {(correct_g + correct_f)/(total_g + total_f)*100:.1f}%")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
DATASET:
   - People: {len(people)}
   - Total signatures: {total_g + total_f} ✅ (1000+)

TRAINING:
   - Train people: {len(train_people)}
   - Val people: {len(val_people)}
   - Training pairs: {len(X1_train)}
   - Validation pairs: {len(X1_val)}

RESULTS:
   - Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%
   - Genuine: {correct_g}/{total_g}
   - Forged: {correct_f}/{total_f}
   - Overall: {(correct_g + correct_f)/(total_g + total_f)*100:.1f}%
""")
print("=" * 70)
print("🎉 DONE!")
print("=" * 70)