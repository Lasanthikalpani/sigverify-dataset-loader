"""
CEDAR 20 People - Best Balance (More data + Better accuracy)
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
print("🧠 SIGVERIFY - CEDAR 20 PEOPLE")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load CEDAR 55 and use first 20 people
# ============================================================
print("\n📂 Loading CEDAR 20 people...")

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

# USE ONLY 20 PEOPLE
MAX_PEOPLE = 20
valid_pids = valid_pids[:MAX_PEOPLE]

total_g = sum(len(person_genuine[p]) for p in valid_pids)
total_f = sum(len(person_forged[p]) for p in valid_pids)

print(f"\n✅ Loaded {len(valid_pids)} people")
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f}")

# ============================================================
# Augment to reach 5000+
# ============================================================
print("\n📂 Augmenting to 5000+...")

def augment_image(img):
    augmented = [img]
    # Rotate +10
    M = cv2.getRotationMatrix2D((64, 64), 10, 1)
    augmented.append(cv2.warpAffine(img, M, (128, 128)))
    # Rotate -10
    M = cv2.getRotationMatrix2D((64, 64), -10, 1)
    augmented.append(cv2.warpAffine(img, M, (128, 128)))
    # Flip
    augmented.append(cv2.flip(img, 1))
    # Scale
    new_size = int(128 * 0.9)
    scaled = cv2.resize(img, (new_size, new_size))
    padded = np.zeros((128, 128))
    start = (128 - new_size) // 2
    padded[start:start+new_size, start:start+new_size] = scaled
    augmented.append(padded)
    return augmented

aug_genuine = {}
aug_forged = {}

for pid in tqdm(valid_pids, desc="Augmenting"):
    aug_genuine[pid] = []
    aug_forged[pid] = []
    for img in person_genuine[pid]:
        aug_genuine[pid].extend(augment_image(img))
    for img in person_forged[pid]:
        aug_forged[pid].extend(augment_image(img))

total_aug_g = sum(len(aug_genuine[p]) for p in valid_pids)
total_aug_f = sum(len(aug_forged[p]) for p in valid_pids)

print(f"\n   ✅ Augmented Genuine: {total_aug_g}")
print(f"   ✅ Augmented Forged: {total_aug_f}")
print(f"   ✅ Total Augmented: {total_aug_g + total_aug_f}")

# ============================================================
# 3-Way Split by People
# ============================================================
print("\n📂 3-Way Split by People (70/15/15)...")

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

# ============================================================
# Create pairs (limited to 3000)
# ============================================================
print("\n📂 Creating pairs...")

def create_pairs(aug_g, aug_f, person_list, max_pairs=3000):
    X1, X2, y = [], [], []
    half = max_pairs // 2
    
    pos = 0
    for pid in person_list:
        if pos >= half: break
        gen = aug_g[pid]
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
        gen = aug_g[pid]
        forg = aug_f[pid]
        for g in gen[:5]:
            if neg >= half: break
            for f in forg[:5]:
                if neg >= half: break
                X1.append(g)
                X2.append(f)
                y.append(0)
                neg += 1
    
    return np.array(X1), np.array(X2), np.array(y)

X1_train, X2_train, y_train = create_pairs(aug_genuine, aug_forged, train_people, 3000)
X1_val, X2_val, y_val = create_pairs(aug_genuine, aug_forged, val_people, 500)
X1_test, X2_test, y_test = create_pairs(aug_genuine, aug_forged, test_people, 500)

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
    ModelCheckpoint('models/model_cedar20.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_cedar20.keras')

# ============================================================
# Test
# ============================================================
print("\n📋 Test on UNSEEN data...")

threshold = 0.3
y_pred = model.predict([X1_test, X2_test], verbose=0)
y_pred_binary = (y_pred > threshold).astype(int).flatten()
accuracy = np.mean(y_pred_binary == y_test) * 100

print(f"\n   Test Accuracy: {accuracy:.1f}%")
print(f"   Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
DATASET:
   - People: {len(valid_pids)}
   - Original: {total_g + total_f}
   - Augmented: {total_aug_g + total_aug_f}

SPLIT:
   - Train: {len(train_people)} people
   - Val:   {len(val_people)} people
   - Test:  {len(test_people)} people

RESULTS:
   - Test Accuracy: {accuracy:.1f}%
   - Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%
""")
print("=" * 70)
print("🎉 DONE!")