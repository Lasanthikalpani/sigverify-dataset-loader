"""
CEDAR 55 People - FAST Training (5000 pairs)
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
print("🧠 SIGVERIFY - 55 PEOPLE FAST TRAINING")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load signatures dataset (55 people)
# ============================================================
print("\n📂 Loading signatures dataset...")

org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

org_files = sorted(list(org_dir.glob('*.png')))
forg_files = sorted(list(forg_dir.glob('*.png')))

print(f"   Genuine files: {len(org_files)}")
print(f"   Forged files: {len(forg_files)}")

def parse_person_id(filename):
    match = re.search(r'_(\d+)_', filename)
    return match.group(1) if match else None

# Load by person
people = {}

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

people = {pid: d for pid, d in people.items()
          if len(d['genuine']) >= 3 and len(d['forged']) >= 3}

print(f"\n✅ Loaded {len(people)} people")
total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f} ✅ (1000+)")

# ============================================================
# 3-Way Split
# ============================================================
print("\n📂 3-Way Split (70/15/15)...")

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
# Create LIMITED pairs (5000 training, 500 val, 500 test)
# ============================================================
print("\n📂 Creating limited pairs...")

def create_limited_pairs(person_list, max_pairs=5000):
    X1, X2, y = [], [], []
    half = max_pairs // 2
    
    # Positive pairs
    pos_count = 0
    for pid in person_list:
        if pos_count >= half:
            break
        gen = people[pid]['genuine']
        for i in range(min(5, len(gen))):
            if pos_count >= half:
                break
            for j in range(i+1, min(i+5, len(gen))):
                if pos_count >= half:
                    break
                X1.append(gen[i])
                X2.append(gen[j])
                y.append(1)
                pos_count += 1
    
    # Negative pairs
    neg_count = 0
    for pid in person_list:
        if neg_count >= half:
            break
        gen = people[pid]['genuine']
        forg = people[pid]['forged']
        for g in gen[:5]:
            if neg_count >= half:
                break
            for f in forg[:5]:
                if neg_count >= half:
                    break
                X1.append(g)
                X2.append(f)
                y.append(0)
                neg_count += 1
    
    return np.array(X1), np.array(X2), np.array(y)

X1_train, X2_train, y_train = create_limited_pairs(train_people, 5000)
X1_val, X2_val, y_val = create_limited_pairs(val_people, 1000)
X1_test, X2_test, y_test = create_limited_pairs(test_people, 1000)

X1_train = X1_train.reshape(-1, 128, 128, 1)
X2_train = X2_train.reshape(-1, 128, 128, 1)
X1_val = X1_val.reshape(-1, 128, 128, 1)
X2_val = X2_val.reshape(-1, 128, 128, 1)
X1_test = X1_test.reshape(-1, 128, 128, 1)
X2_test = X2_test.reshape(-1, 128, 128, 1)

print(f"   Training: {len(X1_train)}")
print(f"   Validation: {len(X1_val)}")
print(f"   Test: {len(X1_test)}")

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
    ModelCheckpoint('models/model_55fast.keras',
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

model.save('models/model_55fast.keras')
print(f"\n✅ Model saved!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# Test
# ============================================================
print("\n📋 Test on UNSEEN data...")

threshold = 0.3
y_pred = model.predict([X1_test, X2_test], verbose=0)
y_pred_binary = (y_pred > threshold).astype(int).flatten()

accuracy = np.mean(y_pred_binary == y_test) * 100

print(f"\n   Test Accuracy: {accuracy:.1f}%")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
DATASET: {total_g + total_f} signatures, {len(people)} people ✅ (1000+)
SPLIT: Train {len(train_people)}, Val {len(val_people)}, Test {len(test_people)}

PAIRS:
   Training:   {len(X1_train)}
   Validation: {len(X1_val)}
   Testing:    {len(X1_test)}

RESULTS:
   Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%
   Test accuracy: {accuracy:.1f}%
""")
print("=" * 70)
print("🎉 DONE!")