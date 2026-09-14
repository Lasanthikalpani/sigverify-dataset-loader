"""
CEDAR Training - 3-Way Split (Train/Val/Test)
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
print("🧠 SIGVERIFY - 3-WAY SPLIT TRAINING")
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

print(f"\n✅ Loaded {len(people)} people")

total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f}")

# ============================================================
# 3-WAY SPLIT BY PEOPLE (70/15/15)
# ============================================================
print("\n📂 3-Way Split by People (70/15/15)...")

person_ids = sorted(list(people.keys()))
np.random.seed(42)
np.random.shuffle(person_ids)

n = len(person_ids)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_people = person_ids[:train_end]
val_people = person_ids[train_end:val_end]
test_people = person_ids[val_end:]

print(f"   Train people: {len(train_people)} ({len(train_people)/n*100:.1f}%)")
print(f"   Val people: {len(val_people)} ({len(val_people)/n*100:.1f}%)")
print(f"   Test people: {len(test_people)} ({len(test_people)/n*100:.1f}%)")

# ============================================================
# Create pairs
# ============================================================
def create_pairs(person_list):
    """Create pairs from given people"""
    X1, X2, y = [], [], []
    
    for pid in person_list:
        gen = people[pid]['genuine']
        forg = people[pid]['forged']
        
        # Positive: same person genuine
        for i in range(min(5, len(gen))):
            for j in range(i+1, min(i+3, len(gen))):
                X1.append(gen[i])
                X2.append(gen[j])
                y.append(1)
        
        # Negative: genuine vs forged
        for g in gen[:3]:
            for f in forg[:3]:
                X1.append(g)
                X2.append(f)
                y.append(0)
    
    return np.array(X1), np.array(X2), np.array(y)

# Training pairs
print("\n📂 Creating training pairs...")
X1_train, X2_train, y_train = create_pairs(train_people)

# Balance training
pos_idx = [i for i, l in enumerate(y_train) if l == 1]
neg_idx = [i for i, l in enumerate(y_train) if l == 0]
min_count = min(len(pos_idx), len(neg_idx))

keep = list(np.random.choice(pos_idx, min_count, replace=False)) + \
       list(np.random.choice(neg_idx, min_count, replace=False))

X1_train = X1_train[keep]
X2_train = X2_train[keep]
y_train = y_train[keep]

# Shuffle
idx = np.random.permutation(len(X1_train))
X1_train, X2_train, y_train = X1_train[idx], X2_train[idx], y_train[idx]

# Reshape
X1_train = X1_train.reshape(-1, 128, 128, 1)
X2_train = X2_train.reshape(-1, 128, 128, 1)

print(f"   Training pairs: {len(X1_train)}")
print(f"   Positive: {np.sum(y_train)}")
print(f"   Negative: {len(y_train) - np.sum(y_train)}")

# Validation pairs
print("\n📂 Creating validation pairs...")
X1_val, X2_val, y_val = create_pairs(val_people)

X1_val = X1_val.reshape(-1, 128, 128, 1)
X2_val = X2_val.reshape(-1, 128, 128, 1)

print(f"   Validation pairs: {len(X1_val)}")

# Test pairs
print("\n📂 Creating test pairs...")
X1_test, X2_test, y_test = create_pairs(test_people)

X1_test = X1_test.reshape(-1, 128, 128, 1)
X2_test = X2_test.reshape(-1, 128, 128, 1)

print(f"   Test pairs: {len(X1_test)}")

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
    ModelCheckpoint('models/model_3way.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=25,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_3way.keras')
print(f"\n✅ Model saved!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# TEST on unseen test set
# ============================================================
print("\n📋 TESTING on unseen test set...")

threshold = 0.3
y_pred = model.predict([X1_test, X2_test], verbose=0)
y_pred_binary = (y_pred > threshold).astype(int).flatten()

accuracy = np.mean(y_pred_binary == y_test) * 100
print(f"\n   Test Accuracy: {accuracy:.1f}%")

# Confusion matrix
tp = np.sum((y_pred_binary == 1) & (y_test == 1))
tn = np.sum((y_pred_binary == 0) & (y_test == 0))
fp = np.sum((y_pred_binary == 1) & (y_test == 0))
fn = np.sum((y_pred_binary == 0) & (y_test == 1))

print(f"\n   Confusion Matrix:")
print(f"   True Positive:  {tp}")
print(f"   True Negative:  {tn}")
print(f"   False Positive: {fp}")
print(f"   False Negative: {fn}")

precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n   Precision: {precision:.1f}%")
print(f"   Recall: {recall:.1f}%")
print(f"   F1-Score: {f1:.1f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
DATASET:
   - Total people: {len(people)}
   - Total signatures: {total_g + total_f} ✅ (1000+)

3-WAY SPLIT:
   - Training:   {len(train_people)} people ({len(train_people)/n*100:.0f}%)
   - Validation: {len(val_people)} people ({len(val_people)/n*100:.0f}%)
   - Testing:    {len(test_people)} people ({len(test_people)/n*100:.0f}%)

PAIRS:
   - Training:   {len(X1_train)}
   - Validation: {len(X1_val)}
   - Testing:    {len(X1_test)}

RESULTS:
   - Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%
   - Test accuracy: {accuracy:.1f}%
   - Precision: {precision:.1f}%
   - Recall: {recall:.1f}%
   - F1-Score: {f1:.1f}%
""")
print("=" * 70)
print("🎉 DONE!")
print("=" * 70)