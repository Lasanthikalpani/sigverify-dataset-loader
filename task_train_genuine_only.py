"""
Task: Train with Genuine Signatures Only
SigVerify - RQ1 Core AI Performance

Since most folders don't have forged versions,
we train on genuine-vs-genuine pairs.
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
print("🧠 SIGVERIFY - GENUINE-ONLY TRAINING")
print("=" * 70)

# ============================================================
# Load genuine signatures from ALL people
# ============================================================
print("\n📂 Loading Kaggle genuine signatures...")

kaggle_dir = Path('data/raw/kaggle')

genuine_folders = sorted([d for d in kaggle_dir.iterdir() 
                          if d.is_dir() and '_forg' not in d.name.lower()])

print(f"   Total people: {len(genuine_folders)}")

MAX_PEOPLE = 500
MAX_PER_PERSON = 5

people = {}

for person_dir in genuine_folders[:MAX_PEOPLE]:
    person_id = person_dir.name
    genuine = []
    
    for img_file in sorted(person_dir.glob('*.jpg'))[:MAX_PER_PERSON]:
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            genuine.append(img)
    
    if len(genuine) >= 2:
        people[person_id] = genuine
    
    if len(people) % 100 == 0 and len(people) > 0:
        print(f"   Loaded {len(people)} people...")

print(f"\n   ✅ Loaded {len(people)} people")

# ============================================================
# Create pairs
# ============================================================
print("\n🔀 Creating pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

# POSITIVE: Same person
for pid in person_ids:
    gen = people[pid]
    for i in range(len(gen)):
        for j in range(i+1, len(gen)):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)

print(f"   Positive (same person): {len(X1)}")

# NEGATIVE: Different people
num_neg = len(X1)

for _ in range(num_neg):
    p1, p2 = np.random.choice(person_ids, 2, replace=False)
    g1 = np.random.choice(len(people[p1]))
    g2 = np.random.choice(len(people[p2]))
    X1.append(people[p1][g1])
    X2.append(people[p2][g2])
    y.append(0)

print(f"   Negative (different person): {num_neg}")

X1 = np.array(X1).reshape(-1, 128, 128, 1)
X2 = np.array(X2).reshape(-1, 128, 128, 1)
y = np.array(y)

idx = np.random.permutation(len(X1))
X1, X2, y = X1[idx], X2[idx], y[idx]

print(f"\n   Total pairs: {len(X1)}")

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
# Train
# ============================================================
print("\n🏋️ Training...")

callbacks = [
    ModelCheckpoint('models/simple_siamese_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/simple_siamese_model.keras')
print("\n✅ Model saved!")

# Plot without plt.show() to avoid blocking
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
plt.savefig('results/training_genuine_only.png')
print("📊 Plot saved to results/training_genuine_only.png")
# No plt.show() - avoids blocking

print("\n🎉 DONE!")