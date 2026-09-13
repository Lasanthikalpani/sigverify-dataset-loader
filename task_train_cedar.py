"""
Task: Train Model with CEDAR Data (Person-Aware Pairs)
SigVerify - RQ1 Core AI Performance
"""

import sys
sys.path.insert(0, 'src')

import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

print("=" * 70)
print("🧠 SIGVERIFY - TRAIN WITH PERSON-AWARE PAIRS")
print("   RQ1: Core AI Performance")
print("=" * 70)

# ============================================================
# Step 1: Load CEDAR preserving person identity
# ============================================================
print("\n📂 Loading CEDAR (person-aware)...")

cedar_dir = Path('data/raw/cedar')

# Store by person: {person_id: {'genuine': [...], 'forged': [...]}}
people = {}

for person_dir in cedar_dir.iterdir():
    if not person_dir.is_dir():
        continue
    
    person_id = person_dir.name
    people[person_id] = {'genuine': [], 'forged': []}
    
    for img_file in person_dir.iterdir():
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128)) / 255.0
                if 'F' in img_file.stem:
                    people[person_id]['forged'].append(img)
                else:
                    people[person_id]['genuine'].append(img)

print(f"   Loaded {len(people)} people")

# Count
total_genuine = sum(len(p['genuine']) for p in people.values())
total_forged = sum(len(p['forged']) for p in people.values())
print(f"   Genuine: {total_genuine}")
print(f"   Forged: {total_forged}")

# ============================================================
# Step 2: Create CORRECT pairs (person-aware)
# ============================================================
print("\n🔀 Creating CORRECT training pairs...")

X1, X2, y = [], [], []

person_ids = list(people.keys())

# Positive pairs: two genuine from SAME person
for person_id in person_ids:
    gen = people[person_id]['genuine']
    if len(gen) >= 2:
        for i in range(len(gen) - 1):
            X1.append(gen[i])
            X2.append(gen[i + 1])
            y.append(1)  # Same person

# Negative pairs: genuine from person A + genuine from person B
num_negative = len(X1)  # Balance

for _ in range(num_negative):
    p1 = np.random.choice(person_ids)
    p2 = np.random.choice([p for p in person_ids if p != p1])
    
    if len(people[p1]['genuine']) > 0 and len(people[p2]['genuine']) > 0:
        g1 = np.random.choice(len(people[p1]['genuine']))
        g2 = np.random.choice(len(people[p2]['genuine']))
        
        X1.append(people[p1]['genuine'][g1])
        X2.append(people[p2]['genuine'][g2])
        y.append(0)  # Different person

X1 = np.array(X1).reshape(-1, 128, 128, 1)
X2 = np.array(X2).reshape(-1, 128, 128, 1)
y = np.array(y)

# Shuffle
indices = np.random.permutation(len(X1))
X1 = X1[indices]
X2 = X2[indices]
y = y[indices]

print(f"   Total pairs: {len(X1)}")
print(f"   Positive (same person): {np.sum(y)}")
print(f"   Negative (different person): {len(y) - np.sum(y)}")

# ============================================================
# Step 3: Build model
# ============================================================
print("\n🧠 Building Siamese CNN model...")

def create_simple_siamese():
    def create_base():
        inputs = Input(shape=(128, 128, 1))
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        return Model(inputs, x)
    
    base = create_base()
    input1 = Input(shape=(128, 128, 1))
    input2 = Input(shape=(128, 128, 1))
    
    emb1 = base(input1)
    emb2 = base(input2)
    
    concat = layers.Concatenate()([emb1, emb2])
    x = layers.Dense(64, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return Model([input1, input2], output)

model = create_simple_siamese()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(f"   Model parameters: {model.count_params():,}")

# ============================================================
# Step 4: Train
# ============================================================
print("\n🏋️ Training model...")

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

# ============================================================
# Step 5: Save & Plot
# ============================================================
model.save('models/simple_siamese_model.keras')
print("\n✅ Model saved to models/simple_siamese_model.keras")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/training_history_cedar.png')
plt.show()

print("\n🎉 TRAINING COMPLETE!")