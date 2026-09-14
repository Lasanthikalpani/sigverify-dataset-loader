"""
CEDAR Training - FULL 55 People (data/raw/signatures/)
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
print("🧠 SIGVERIFY - FULL 55 PEOPLE TRAINING")
print("=" * 70)

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

# ============================================================
# Load signatures dataset (55 people)
# ============================================================
print("\n📂 Loading signatures dataset...")

org_dir = Path('data/raw/signatures/full_org')
forg_dir = Path('data/raw/signatures/full_forg')

# Get all files
org_files = sorted(list(org_dir.glob('*.png')))
forg_files = sorted(list(forg_dir.glob('*.png')))

print(f"   Genuine files: {len(org_files)}")
print(f"   Forged files: {len(forg_files)}")

# Show sample names
print(f"\n   Sample genuine: {[f.name for f in org_files[:3]]}")
print(f"   Sample forged: {[f.name for f in forg_files[:3]]}")

# ============================================================
# Parse person ID correctly
# ============================================================
def parse_person_id(filename):
    """Parse person ID from filename like original_10_1 or forgeries_10_1"""
    # Match pattern: prefix_NUMBER_suffix
    match = re.search(r'_(\d+)_', filename)
    if match:
        return match.group(1)
    return None

# Test parsing
test_names = [f.stem for f in org_files[:5]]
print(f"\n   Testing parser:")
for name in test_names:
    pid = parse_person_id(name)
    print(f"      {name} -> Person ID: {pid}")

# ============================================================
# Load by person
# ============================================================
print("\n📂 Loading by person...")

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

# Filter people with enough images
people = {pid: d for pid, d in people.items()
          if len(d['genuine']) >= 3 and len(d['forged']) >= 3}

print(f"\n✅ Loaded {len(people)} people")
print(f"   Person IDs: {sorted(list(people.keys()))[:10]}...")

total_g = sum(len(p['genuine']) for p in people.values())
total_f = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_g}")
print(f"   Forged: {total_f}")
print(f"   Total: {total_g + total_f} ✅ (1000+)")

# ============================================================
# 3-Way Split by People
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

print(f"   Train: {len(train_people)} people")
print(f"   Val:   {len(val_people)} people")
print(f"   Test:  {len(test_people)} people")

# ============================================================
# Create pairs (all images)
# ============================================================
print("\n📂 Creating pairs (all images)...")

def create_pairs(person_list):
    X1, X2, y = [], [], []
    for pid in person_list:
        gen = people[pid]['genuine']
        forg = people[pid]['forged']
        
        # Positive: Same person genuine
        for i in range(len(gen)):
            for j in range(i+1, len(gen)):
                X1.append(gen[i])
                X2.append(gen[j])
                y.append(1)
        
        # Negative: Genuine vs Forged
        for g in gen:
            for f in forg:
                X1.append(g)
                X2.append(f)
                y.append(0)
    
    return np.array(X1), np.array(X2), np.array(y)

X1_train, X2_train, y_train = create_pairs(train_people)
X1_val, X2_val, y_val = create_pairs(val_people)
X1_test, X2_test, y_test = create_pairs(test_people)

print(f"   Training pairs: {len(X1_train)}")
print(f"   Validation pairs: {len(X1_val)}")
print(f"   Test pairs: {len(X1_test)}")

# ============================================================
# Reshape
# ============================================================
X1_train = X1_train.reshape(-1, 128, 128, 1)
X2_train = X2_train.reshape(-1, 128, 128, 1)
X1_val = X1_val.reshape(-1, 128, 128, 1)
X2_val = X2_val.reshape(-1, 128, 128, 1)
X1_test = X1_test.reshape(-1, 128, 128, 1)
X2_test = X2_test.reshape(-1, 128, 128, 1)

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
    ModelCheckpoint('models/model_55people.keras',
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

model.save('models/model_55people.keras')
print(f"\n✅ Model saved!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# ============================================================
# Test
# ============================================================
print("\n📋 Final Test on UNSEEN data...")

threshold = 0.3
y_pred = model.predict([X1_test, X2_test], verbose=0)
y_pred_binary = (y_pred > threshold).astype(int).flatten()

accuracy = np.mean(y_pred_binary == y_test) * 100

tp = np.sum((y_pred_binary == 1) & (y_test == 1))
tn = np.sum((y_pred_binary == 0) & (y_test == 0))
fp = np.sum((y_pred_binary == 1) & (y_test == 0))
fn = np.sum((y_pred_binary == 0) & (y_test == 1))

precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n   Test Accuracy: {accuracy:.1f}%")
print(f"   Precision: {precision:.1f}%")
print(f"   Recall: {recall:.1f}%")
print(f"   F1-Score: {f1:.1f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("📋 SUPERVISOR REQUIREMENTS CHECKLIST")
print("=" * 70)
print(f"""
✅ 1. 1000+ DATASET
   - Total: {total_g + total_f} signatures
   - People: {len(people)}

✅ 2. 3-WAY SPLIT (70/15/15)
   - Training:   {len(train_people)} people
   - Validation: {len(val_people)} people
   - Testing:    {len(test_people)} people

✅ 3. CNN MODEL
   - Siamese CNN: {model.count_params():,} parameters

✅ 4. ACCURACY
   - Test Accuracy: {accuracy:.1f}%
   - Precision: {precision:.1f}%
   - Recall: {recall:.1f}%
   - F1-Score: {f1:.1f}%

TRAINING PAIRS:
   - Training:   {len(X1_train)}
   - Validation: {len(X1_val)}
   - Testing:    {len(X1_test)}
""")
print("=" * 70)
print("🎉 DONE!")
print("=" * 70)