"""
Task: Fast Training with 764 People
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
import matplotlib.pyplot as plt

print("=" * 70)
print("🧠 SIGVERIFY - FAST TRAINING")
print("=" * 70)

# ============================================================
# Load data
# ============================================================
print("\n📂 Loading Kaggle dataset...")

kaggle_dir = Path('data/raw/kaggle')

genuine_folders = sorted([d for d in kaggle_dir.iterdir() 
                          if d.is_dir() and '_forg' not in d.name.lower()])

print(f"   Total people available: {len(genuine_folders)}")

# Use 200 people only for speed
MAX_PEOPLE = 200
MAX_PER_PERSON = 3

people = {}

for person_dir in genuine_folders[:MAX_PEOPLE]:
    person_id = person_dir.name
    forged_dir = kaggle_dir / f"{person_id}_forg"
    
    genuine = []
    for img_file in sorted(person_dir.glob('*.jpg'))[:MAX_PER_PERSON]:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            genuine.append(img)
    
    forged = []
    if forged_dir.exists():
        for img_file in sorted(forged_dir.glob('*.jpg'))[:MAX_PER_PERSON]:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128)) / 255.0
                forged.append(img)
    
    if len(genuine) >= 2 and len(forged) >= 2:
        people[person_id] = {'genuine': genuine, 'forged': forged}

print(f"   ✅ Loaded {len(people)} people")

# ============================================================
# Create pairs (LIMITED)
# ============================================================
print("\n🔀 Creating pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

# POSITIVE: 2 pairs per person
for pid in person_ids:
    gen = people[pid]['genuine']
    X1.append(gen[0])
    X2.append(gen[1])
    y.append(1)

print(f"   Positive: {len(X1)}")

# NEGATIVE: Genuine vs Forged (same person)
for pid in person_ids:
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    X1.append(gen[0])
    X2.append(forg[0])
    y.append(0)

print(f"   Negative (genuine vs forged): {len(X1) - len(person_ids)}")

# NEGATIVE: Genuine vs Genuine (different person)
for _ in range(len(person_ids)):
    p1, p2 = np.random.choice(person_ids, 2, replace=False)
    X1.append(people[p1]['genuine'][0])
    X2.append(people[p2]['genuine'][0])
    y.append(0)

X1 = np.array(X1).reshape(-1, 128, 128, 1)
X2 = np.array(X2).reshape(-1, 128, 128, 1)
y = np.array(y)

idx = np.random.permutation(len(X1))
X1, X2, y = X1[idx], X2[idx], y[idx]

print(f"   Total pairs: {len(X1)}")
print(f"   Positive: {np.sum(y)}")
print(f"   Negative: {len(y) - np.sum(y)}")

# ============================================================
# Build model
# ============================================================
print("\n🧠 Building model...")

def create_model():
    def base():
        inp = Input(shape=(128, 128, 1))
        x = layers.Conv2D(32, (3,3), activation='relu')(inp)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, (3,3), activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, (3,3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        return Model(inp, x)
    
    b = base()
    i1 = Input(shape=(128, 128, 1))
    i2 = Input(shape=(128, 128, 1))
    e1, e2 = b(i1), b(i2)
    concat = layers.Concatenate()([e1, e2])
    x = layers.Dense(64, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model([i1, i2], out)

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(f"   Parameters: {model.count_params():,}")

# ============================================================
# Train - FAST
# ============================================================
print("\n🏋️ Training (fast)...")

callbacks = [
    ModelCheckpoint('models/simple_siamese_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=15,
    batch_size=64,  # Larger batch = faster
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Save
model.save('models/simple_siamese_model.keras')
print("\n✅ Model saved!")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('Accuracy')
ax1.legend()
ax1.grid(True)
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('Loss')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/training_fast.png')
plt.show()

print("\n🎉 DONE!")