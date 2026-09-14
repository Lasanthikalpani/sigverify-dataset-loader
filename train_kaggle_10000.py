"""
Kaggle Training - 10,000 Pairs
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

print("=" * 70)
print("🧠 SIGVERIFY - KAGGLE 10,000 PAIRS")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load 500 people
# ============================================================
print("\n📂 Loading 500 people...")

kaggle_dir = Path('data/raw/kaggle')
genuine_folders = sorted([d for d in kaggle_dir.iterdir()
                          if d.is_dir() and not d.name.endswith('_forg')])

people = {}
MAX_PEOPLE = 500

for person_dir in tqdm(genuine_folders[:MAX_PEOPLE], desc="Loading"):
    pid = person_dir.name
    forged_dir = kaggle_dir / f"{pid}_forg"
    
    if not forged_dir.exists():
        continue
    
    genuine = []
    for img_file in sorted(person_dir.glob('*.jpg'))[:5]:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = cv2.equalizeHist(img)
            img = img.astype(np.float32) / 255.0
            genuine.append(img)
    
    forged = []
    for img_file in sorted(forged_dir.glob('*.jpg'))[:5]:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = cv2.equalizeHist(img)
            img = img.astype(np.float32) / 255.0
            forged.append(img)
    
    if len(genuine) >= 2 and len(forged) >= 2:
        people[pid] = {'genuine': genuine, 'forged': forged}

print(f"\n✅ Loaded {len(people)} people")

total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f}")

# ============================================================
# Create exactly 10,000 pairs
# ============================================================
print("\n🔀 Creating 10,000 pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

TARGET_PAIRS = 10000
POSITIVE_TARGET = TARGET_PAIRS // 2  # 5,000
NEGATIVE_TARGET = TARGET_PAIRS // 2  # 5,000

# Positive: Same person genuine
pos_count = 0
while pos_count < POSITIVE_TARGET:
    pid = np.random.choice(person_ids)
    gen = people[pid]['genuine']
    if len(gen) >= 2:
        i, j = np.random.choice(len(gen), 2, replace=False)
        X1.append(gen[i])
        X2.append(gen[j])
        y.append(1)
        pos_count += 1

print(f"   Positive: {pos_count}")

# Negative 1: Genuine vs Forged (same person)
neg_count = 0
while neg_count < NEGATIVE_TARGET // 2:
    pid = np.random.choice(person_ids)
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    if len(gen) >= 1 and len(forg) >= 1:
        g = np.random.choice(len(gen))
        f = np.random.choice(len(forg))
        X1.append(gen[g])
        X2.append(forg[f])
        y.append(0)
        neg_count += 1

print(f"   Negative (genuine vs forged): {neg_count}")

# Negative 2: Different person genuine
while neg_count < NEGATIVE_TARGET:
    p1, p2 = np.random.choice(person_ids, 2, replace=False)
    g1 = np.random.choice(len(people[p1]['genuine']))
    g2 = np.random.choice(len(people[p2]['genuine']))
    X1.append(people[p1]['genuine'][g1])
    X2.append(people[p2]['genuine'][g2])
    y.append(0)
    neg_count += 1

print(f"   Negative (different person): {neg_count - NEGATIVE_TARGET // 2}")

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
    ModelCheckpoint('models/model_kaggle_10k.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_kaggle_10k.keras')
print(f"\n✅ Model saved as 'models/model_kaggle_10k.keras'")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# Test
# ============================================================
print("\n📋 Testing...")

threshold = 0.3
correct = 0
total = 0

for pid in person_ids[:100]:
    pred1 = float(model.predict([
        people[pid]['genuine'][0].reshape(1, 128, 128, 1),
        people[pid]['genuine'][1].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    pred2 = float(model.predict([
        people[pid]['genuine'][0].reshape(1, 128, 128, 1),
        people[pid]['forged'][0].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    if pred1 > threshold: correct += 1
    if pred2 < threshold: correct += 1
    total += 2

print(f"\n   Test Accuracy (100 people): {correct/total*100:.1f}%")
print("\n🎉 DONE!")