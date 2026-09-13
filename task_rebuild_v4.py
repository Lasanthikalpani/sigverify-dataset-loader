"""
Task: Rebuild Model with Grad-CAM Compatible Architecture
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
print("🧠 SIGVERIFY - REBUILD MODEL v4 (Grad-CAM Compatible)")
print("=" * 70)

# ============================================================
# Load data
# ============================================================
print("\n📂 Loading data...")

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

print(f"   CEDAR: {len(people)} people")

kaggle_dir = Path('data/raw/kaggle')
genuine_folders = sorted([d for d in kaggle_dir.iterdir()
                          if d.is_dir() and not d.name.endswith('_forg')])

kaggle_loaded = 0
MAX_PEOPLE = 300

for person_dir in genuine_folders:
    pid = f"kaggle_{person_dir.name}"
    forged_dir = kaggle_dir / f"{person_dir.name}_forg"
    if not forged_dir.exists():
        continue

    genuine, forged = [], []
    for img_file in person_dir.glob('*.jpg'):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            genuine.append(img)
            if len(genuine) >= 3:
                break
    for img_file in forged_dir.glob('*.jpg'):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128)) / 255.0
            forged.append(img)
            if len(forged) >= 3:
                break

    if len(genuine) >= 2 and len(forged) >= 2:
        people[pid] = {'genuine': genuine, 'forged': forged}
        kaggle_loaded += 1

    if kaggle_loaded >= MAX_PEOPLE:
        break

print(f"   Kaggle: {kaggle_loaded} people")
print(f"   Total: {len(people)} people")

# ============================================================
# Create pairs
# ============================================================
print("\n🔀 Creating pairs...")

X1, X2, y = [], [], []
person_ids = list(people.keys())

for pid in person_ids:
    gen = people[pid]['genuine']
    for i in range(len(gen)):
        for j in range(i+1, len(gen)):
            X1.append(gen[i])
            X2.append(gen[j])
            y.append(1)

pos_count = len(X1)

for pid in person_ids:
    gen = people[pid]['genuine']
    forg = people[pid]['forged']
    for g in gen:
        for f in forg:
            X1.append(g)
            X2.append(f)
            y.append(0)

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

print(f"   Total pairs: {len(X1)}")

# ============================================================
# Build model with Grad-CAM compatible architecture
# ============================================================
print("\n🧠 Building Grad-CAM compatible model...")

def create_model():
    """Model with conv layers directly accessible"""
    
    # Input layers
    input1 = Input(shape=(128, 128, 1), name='input1')
    input2 = Input(shape=(128, 128, 1), name='input2')
    
    # Shared CNN (using Sequential for cleaner access)
    base = tf.keras.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu', 
                      input_shape=(128, 128, 1), name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu', name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu', name='conv3'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu')
    ], name='base_network')
    
    # Process both inputs
    emb1 = base(input1)
    emb2 = base(input2)
    
    # Absolute difference (using Subtract and Abs layers)
    diff = layers.Subtract()([emb1, emb2])
    abs_diff = layers.Lambda(
        lambda x: tf.abs(x),
        output_shape=(64,),
        name='abs_diff'
    )(diff)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(abs_diff)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=[input1, input2], outputs=output)
    return model

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(f"   Parameters: {model.count_params():,}")

# Print model structure
print("\n   Model layers:")
for i, layer in enumerate(model.layers):
    print(f"      {i}: {layer.name} ({layer.__class__.__name__})")

# ============================================================
# Train
# ============================================================
print("\n🏋️ Training...")

callbacks = [
    ModelCheckpoint('models/model_v4.keras',
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

history = model.fit(
    [X1, X2], y,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Save model
model.save('models/model_v4.keras')
print("\n✅ Model saved as 'models/model_v4.keras'")
print("\n🎉 DONE!")