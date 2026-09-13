"""
Task: Train Model with Kaggle Dataset (Person-Aware)
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
print("🧠 SIGVERIFY - TRAIN WITH KAGGLE (PERSON-AWARE)")
print("   RQ1: Core AI Performance")
print("=" * 70)

# ============================================================
# Step 1: Load Kaggle preserving person identity
# ============================================================
print("\n📂 Loading Kaggle dataset (person-aware)...")

kaggle_dir = Path('data/raw/kaggle')

# Check what's in the folder
all_folders = [d for d in kaggle_dir.iterdir() if d.is_dir()]
print(f"   Total folders: {len(all_folders)}")

# Get genuine folders (no _forg)
genuine_folders = [d for d in all_folders if '_forg' not in d.name.lower()]
print(f"   Genuine folders: {len(genuine_folders)}")

# Limit to 100 people for speed
MAX_PEOPLE = 100
genuine_folders = genuine_folders[:MAX_PEOPLE]

# Load people
people = {}
for person_dir in genuine_folders:
    person_id = person_dir.name
    forged_dir = kaggle_dir / f"{person_id}_forg"
    
    # Load genuine
    genuine = []
    for img_file in sorted(person_dir.glob('*.jpg'))[:10]:  # Max 10 per person
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            genuine.append(img)
    
    # Load forged
    forged = []
    if forged_dir.exists():
        for img_file in sorted(forged_dir.glob('*.jpg'))[:10]:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128)) / 255.0
                forged.append(img)
    
    if len(genuine) >= 2 and len(forged) >= 2:
        people[person_id] = {'genuine': genuine, 'forged': forged}
    
    if len(people) % 20 == 0 and len(people) > 0:
        print(f"   Loaded {len(people)} people...")

print(f"\n   ✅ Loaded {len(people)} people")
print(f"   Genuine per person: ~{np.mean([len(p['genuine']) for p in people.values()]):.1f}")
print(f"   Forged per person: ~{np.mean([len(p['forged']) for p in people.values()]):.1f}")

# ============================================================
# Step 2: Create CORRECT person-aware pairs
# ============================================================
print("\n🔀 Creating CORRECT person-aware pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

# Positive pairs: two genuine from SAME person
for person_id in person_ids:
    gen = people[person_id]['genuine']
    for i in range(len(gen)):
        for j in range(i + 1, len(gen)):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)  # Same person

# Negative pairs: genuine from person A + forged from person A
# (This teaches the model to distinguish genuine from forged)
for person_id in person_ids:
    gen = people[person_id]['genuine']
    forg = people[person_id]['forged']
    for g in gen[:5]:
        for f in forg[:5]:
            X1.append(g)
            X2.append(f)
            y.append(0)  # Different (forged)

# Balance positive and negative
pos_count = sum(y)
neg_count = len(y) - pos_count

if pos_count > neg_count:
    # Remove extra positives
    pos_indices = [i for i, label in enumerate(y) if label == 1]
    remove_indices = np.random.choice(pos_indices, pos_count - neg_count, replace=False)
    remove_set = set(remove_indices)
    X1 = [x for i, x in enumerate(X1) if i not in remove_set]
    X2 = [x for i, x in enumerate(X2) if i not in remove_set]
    y = [label for i, label in enumerate(y) if i not in remove_set]
elif neg_count > pos_count:
    # Remove extra negatives
    neg_indices = [i for i, label in enumerate(y) if label == 0]
    remove_indices = np.random.choice(neg_indices, neg_count - pos_count, replace=False)
    remove_set = set(remove_indices)
    X1 = [x for i, x in enumerate(X1) if i not in remove_set]
    X2 = [x for i, x in enumerate(X2) if i not in remove_set]
    y = [label for i, label in enumerate(y) if i not in remove_set]

X1 = np.array(X1).reshape(-1, 128, 128, 1)
X2 = np.array(X2).reshape(-1, 128, 128, 1)
y = np.array(y)

# Shuffle
indices = np.random.permutation(len(X1))
X1, X2, y = X1[indices], X2[indices], y[indices]

print(f"   Total pairs: {len(X1)}")
print(f"   Positive (same person): {np.sum(y)}")
print(f"   Negative (genuine vs forged): {len(y) - np.sum(y)}")

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
plt.savefig('results/training_history_kaggle.png')
plt.show()

print("\n🎉 TRAINING COMPLETE!")