"""
Final CEDAR Training - Reliable Data
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

print("=" * 70)
print("🧠 SIGVERIFY - CEDAR TRAINING (RELIABLE)")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load CEDAR
# ============================================================
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

print(f"   ✅ Loaded {len(people)} people")

for pid in list(people.keys())[:3]:
    print(f"   {pid}: {len(people[pid]['genuine'])} genuine, {len(people[pid]['forged'])} forged")

# ============================================================
# Create pairs CORRECTLY
# ============================================================
print("\n🔀 Creating pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

# 1. POSITIVE: Genuine vs Genuine (SAME person)
for pid in person_ids:
    gen = people[pid]['genuine']
    for i in range(len(gen)):
        for j in range(i+1, len(gen)):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)

pos_count = len(X1)
print(f"   Positive (same person): {pos_count}")

# 2. NEGATIVE: Genuine vs Forged (SAME person)
for pid in person_ids:
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen:
        for f in forg:
            X1.append(g)
            X2.append(f)
            y.append(0)

neg_count = len([l for l in y if l == 0])
print(f"   Negative (genuine vs forged): {neg_count}")

# 3. NEGATIVE: Genuine vs Genuine (DIFFERENT person)
neg2_count = 0
for _ in range(pos_count):
    p1, p2 = np.random.choice(person_ids, 2, replace=False)
    g1 = np.random.choice(len(people[p1]['genuine']))
    g2 = np.random.choice(len(people[p2]['genuine']))
    X1.append(people[p1]['genuine'][g1])
    X2.append(people[p2]['genuine'][g2])
    y.append(0)
    neg2_count += 1

print(f"   Negative (different person): {neg2_count}")

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
    ModelCheckpoint('models/model_cedar_final.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_cedar_final.keras')
print(f"\n✅ Model saved as 'models/model_cedar_final.keras'")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# Test
# ============================================================
print("\n📋 Testing...")

threshold = 0.3
correct = 0
total = 0

for pid in person_ids:
    # Genuine vs Genuine
    pred1 = float(model.predict([
        people[pid]['genuine'][0].reshape(1, 128, 128, 1),
        people[pid]['genuine'][1].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    # Genuine vs Forged
    pred2 = float(model.predict([
        people[pid]['genuine'][0].reshape(1, 128, 128, 1),
        people[pid]['forged'][0].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    g_ok = pred1 > threshold
    f_ok = pred2 < threshold
    
    if g_ok: correct += 1
    if f_ok: correct += 1
    total += 2

accuracy = correct / total * 100
print(f"\n   Test Accuracy: {accuracy:.1f}%")
print(f"   Genuine correct: {sum(1 for pid in person_ids if float(model.predict([people[pid]['genuine'][0].reshape(1,128,128,1), people[pid]['genuine'][1].reshape(1,128,128,1)], verbose=0)[0][0]) > threshold)}/{len(person_ids)}")
print(f"   Forged correct: {sum(1 for pid in person_ids if float(model.predict([people[pid]['genuine'][0].reshape(1,128,128,1), people[pid]['forged'][0].reshape(1,128,128,1)], verbose=0)[0][0]) < threshold)}/{len(person_ids)}")

print("\n🎉 DONE!")