"""
Task: Train Model with 5,000 Augmented Images
SigVerify - RQ1 Core AI Performance
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

print("=" * 70)
print("🧠 SIGVERIFY - TRAIN WITH EXPANDED DATASET")
print("   RQ1: Core AI Performance")
print("=" * 70)

# Load augmented data
print("\n📂 Loading augmented data...")
genuine = np.load('data/processed/genuine_augmented.npy')
forged = np.load('data/processed/forged_augmented.npy')

print(f"   Genuine: {genuine.shape}")
print(f"   Forged: {forged.shape}")

# Add channel dimension
genuine = genuine.reshape(-1, 128, 128, 1)
forged = forged.reshape(-1, 128, 128, 1)

# Create pairs
print("\n🔀 Creating training pairs...")
X1, X2, y = [], [], []

# Positive pairs (genuine vs genuine)
for i in range(len(genuine) - 1):
    X1.append(genuine[i])
    X2.append(genuine[i + 1])
    y.append(1)

# Negative pairs (genuine vs forged)
for i in range(len(forged)):
    X1.append(genuine[i])
    X2.append(forged[i])
    y.append(0)

X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)

# Shuffle
indices = np.random.permutation(len(X1))
X1 = X1[indices]
X2 = X2[indices]
y = y[indices]

print(f"   Total pairs: {len(X1)}")
print(f"   Positive: {np.sum(y)}")
print(f"   Negative: {len(y) - np.sum(y)}")

# Create model
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

# Train
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

# Save
model.save('models/simple_siamese_model.keras')
print("\n✅ Model saved to models/simple_siamese_model.keras")

# Plot
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
plt.savefig('results/training_history_expanded.png')
plt.show()

print("\n🎉 TRAINING COMPLETE!")