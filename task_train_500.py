"""
Task: Train with 500 People
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
print("🧠 SIGVERIFY - TRAIN WITH 500 PEOPLE")
print("=" * 70)

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
    genuine = []
    forged = []
    
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

print(f"   CEDAR: {len(people)} people")

# ============================================================
# Load Kaggle (500 people)
# ============================================================
print("\n📂 Loading Kaggle...")

kaggle_dir = Path('data/raw/kaggle')
genuine_folders = sorted([d for d in kaggle_dir.iterdir() 
                          if d.is_dir() and not d.name.endswith('_forg')])

kaggle_loaded = 0
MAX_PEOPLE = 500  # ← Changed from 200 to 500

for person_dir in genuine_folders:
    pid = f"kaggle_{person_dir.name}"
    forged_dir = kaggle_dir / f"{person_dir.name}_forg"
    
    if not forged_dir.exists():
        continue
    
    genuine = []
    for img_file in person_dir.glob('*.jpg'):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            genuine.append(img)
            if len(genuine) >= 5:
                break
    
    forged = []
    for img_file in forged_dir.glob('*.jpg'):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            forged.append(img)
            if len(forged) >= 5:
                break
    
    if len(genuine) >= 2 and len(forged) >= 2:
        people[pid] = {'genuine': genuine, 'forged': forged}
        kaggle_loaded += 1
    
    if kaggle_loaded % 50 == 0 and kaggle_loaded > 0:
        print(f"   Loaded {kaggle_loaded} Kaggle people...")
    
    if kaggle_loaded >= MAX_PEOPLE:
        break

print(f"   Kaggle: {kaggle_loaded} people")
print(f"   Total: {len(people)} people")

# Count
total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"\n   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total signatures: {total_g + total_f}")

# ============================================================
# Create pairs
# ============================================================
print("\n🔀 Creating pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

# Positive
for pid in person_ids:
    gen = people[pid]['genuine']
    for i in range(len(gen)):
        for j in range(i+1, len(gen)):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)

pos_count = len(X1)
print(f"   Positive: {pos_count}")

# Negative (genuine vs forged)
for pid in person_ids:
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen:
        for f in forg:
            X1.append(g)
            X2.append(f)
            y.append(0)

# Balance with different-person pairs
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
# Model
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
    dist = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([e1, e2])
    x = layers.Dense(64, activation='relu')(dist)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model([i1, i2], out)

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(f"   Parameters: {model.count_params():,}")

# ============================================================
# Train
# ============================================================
print("\n🏋️ Training...")

callbacks = [
    ModelCheckpoint('models/simple_siamese_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=20,
    batch_size=64,  # Larger for faster training
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/simple_siamese_model.keras')
print("\n✅ Model saved!")
print("\n🎉 DONE!")