"""
Train with CEDAR + Augmentation
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
print("🧠 SIGVERIFY - CEDAR + AUGMENTATION")
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

# ============================================================
# Augment function
# ============================================================
def augment_image(img):
    """Create augmented versions"""
    augmented = [img]
    
    # Rotation
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((64, 64), angle, 1)
        rotated = cv2.warpAffine(img, M, (128, 128))
        augmented.append(rotated)
    
    # Flip
    augmented.append(cv2.flip(img, 1))
    
    # Scale
    for scale in [0.9, 1.1]:
        new_size = int(128 * scale)
        scaled = cv2.resize(img, (new_size, new_size))
        if new_size > 128:
            start = (new_size - 128) // 2
            scaled = scaled[start:start+128, start:start+128]
        else:
            padded = np.zeros((128, 128))
            start = (128 - new_size) // 2
            padded[start:start+new_size, start:start+new_size] = scaled
            scaled = padded
        augmented.append(scaled)
    
    # Noise
    noise = np.random.normal(0, 0.05, img.shape)
    augmented.append(np.clip(img + noise, 0, 1))
    
    return augmented

# ============================================================
# Create augmented pairs
# ============================================================
print("\n🔀 Creating augmented pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

# Positive: Genuine vs Genuine (augmented)
for pid in person_ids:
    gen = people[pid]['genuine']
    for i in range(len(gen)):
        for j in range(i+1, len(gen)):
            # Original
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)
            
            # Augmented versions
            for aug in augment_image(gen[i])[1:3]:
                X1.append(aug)
                X2.append(gen[j])
                y.append(1)

# Negative: Genuine vs Forged
for pid in person_ids:
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen:
        for f in forg:
            X1.append(g)
            X2.append(f)
            y.append(0)

# Balance
pos_count = len([l for l in y if l == 1])
neg_count = len([l for l in y if l == 0])

print(f"   Before balance - Positive: {pos_count}, Negative: {neg_count}")

min_count = min(pos_count, neg_count)
pos_idx = [i for i, l in enumerate(y) if l == 1]
neg_idx = [i for i, l in enumerate(y) if l == 0]

keep = list(np.random.choice(pos_idx, min_count, replace=False)) + \
       list(np.random.choice(neg_idx, min_count, replace=False))

X1 = [X1[i] for i in keep]
X2 = [X2[i] for i in keep]
y = [y[i] for i in keep]

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
    dist = layers.Lambda(absolute_difference, output_shape=(64,))([e1, e2])
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
    ModelCheckpoint('models/model_cedar_augmented.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

model.save('models/model_cedar_augmented.keras')
print(f"\n✅ Model saved!")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print("\n🎉 DONE!")