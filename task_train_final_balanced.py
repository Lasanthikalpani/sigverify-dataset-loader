"""
Task: Final Balanced Training
SigVerify - RQ1 Core AI Performance
"""

import sys, os
sys.path.insert(0, 'src')

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("=" * 70)
print("🧠 SIGVERIFY - FINAL BALANCED TRAINING")
print("=" * 70)

kaggle_dir = Path('data/raw/kaggle')

genuine_folders = sorted([d for d in kaggle_dir.iterdir() 
                          if d.is_dir() and '_forg' not in d.name.lower()])

print(f"   Total people: {len(genuine_folders)}")

# Load 500 people
MAX_PEOPLE = 500
MAX_PER_PERSON = 5

people = {}

for person_dir in genuine_folders[:MAX_PEOPLE]:
    person_id = person_dir.name
    forged_dir = kaggle_dir / f"{person_id}_forg"
    
    if not forged_dir.exists():
        continue
    
    genuine = []
    for img_file in sorted(person_dir.glob('*.jpg'))[:MAX_PER_PERSON]:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            genuine.append(img)
    
    forged = []
    for img_file in sorted(forged_dir.glob('*.jpg'))[:MAX_PER_PERSON]:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            forged.append(img)
    
    if len(genuine) >= 2 and len(forged) >= 2:
        people[person_id] = {'genuine': genuine, 'forged': forged}
    
    if len(people) % 100 == 0 and len(people) > 0:
        print(f"   Loaded {len(people)} people...")

print(f"\n   ✅ Loaded {len(people)} people")

total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")

# ============================================================
# Create BALANCED pairs
# ============================================================
print("\n🔀 Creating balanced pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

# POSITIVE: Same person genuine
for pid in person_ids:
    gen = people[pid]['genuine']
    for i in range(len(gen)):
        for j in range(i+1, min(i+3, len(gen))):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)

pos_count = len(X1)
print(f"   Positive: {pos_count}")

# NEGATIVE 1: Genuine vs Forged (same person) — SAME count as positive
for pid in person_ids:
    if len(X1) >= pos_count * 2:
        break
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen[:1]:
        for f in forg[:1]:
            X1.append(g)
            X2.append(f)
            y.append(0)

print(f"   Negative (genuine vs forged): {len([l for l in y if l == 0])}")

# NEGATIVE 2: Different person — balance to equal
while len([l for l in y if l == 0]) < pos_count:
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
# Build BETTER model
# ============================================================
print("\n🧠 Building improved model...")

def create_model():
    def base():
        inp = Input(shape=(128, 128, 1))
        
        # Block 1
        x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
        
        # Block 2
        x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Block 3
        x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Block 4
        x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        
        return Model(inp, x)
    
    b = base()
    i1 = Input(shape=(128, 128, 1))
    i2 = Input(shape=(128, 128, 1))
    e1, e2 = b(i1), b(i2)
    
    # L1 distance
    dist = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([e1, e2])
    
    x = layers.Dense(64, activation='relu')(dist)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
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
    ModelCheckpoint('models/simple_siamese_model.keras', 
                    monitor='val_accuracy', 
                    save_best_only=True, 
                    verbose=1),
    EarlyStopping(monitor='val_loss', 
                  patience=10, 
                  restore_best_weights=True, 
                  verbose=1),
    ReduceLROnPlateau(monitor='val_loss', 
                      factor=0.5, 
                      patience=3, 
                      min_lr=1e-7, 
                      verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/simple_siamese_model.keras')
print("\n✅ Model saved!")
print("\n🎉 DONE!")