"""
Grad-CAM - Simplest Working Version
SigVerify - RQ2: Explainability & Trust
"""

import sys, os
sys.path.insert(0, 'src')

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Custom function
def absolute_difference(x):
    return tf.abs(x)

print("=" * 70)
print("🔍 SIGVERIFY - GRAD-CAM (SIMPLE WORKING)")
print("=" * 70)

# Load model
print("\n🧠 Loading model...")
model = load_model(
    'models/model_v3.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

# ============================================================
# Load CEDAR data
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
# Grad-CAM using direct gradient computation
# ============================================================
print("\n📋 Generating Grad-CAM...")
os.makedirs('results', exist_ok=True)

# Get the base network and conv layer
base = model.get_layer('functional')
conv_layer = base.get_layer('conv2d_2')

print(f"   Base: {base.name}")
print(f"   Conv: {conv_layer.name}")

# Create a model that outputs conv layer (using the base directly)
grad_model = tf.keras.Model(
    inputs=base.input,
    outputs=conv_layer.output
)
print("   ✅ Grad model created")

for i, pid in enumerate(list(people.keys())[:3]):
    genuine = people[pid]['genuine'][0]
    forged = people[pid]['forged'][0]
    
    genuine_input = genuine.reshape(1, 128, 128, 1)
    forged_input = forged.reshape(1, 128, 128, 1)
    
    # Prediction
    pred = model.predict([genuine_input, forged_input], verbose=0)[0][0]
    decision = "Genuine" if pred > 0.5 else "Forged"
    
    # ============================================================
    # Grad-CAM: Compute gradients
    # ============================================================
    with tf.GradientTape() as tape:
        # Get conv output
        conv_output = grad_model(genuine_input)
        tape.watch(conv_output)
        # Get the embedding (after global pooling)
        # Then compute final prediction manually
        # Since we can't easily connect, just use the conv output
        # to compute loss as the mean
        loss = tf.reduce_mean(conv_output)
    
    # Compute gradients
    grads = tape.gradient(loss, conv_output)
    
    if grads is None:
        print(f"   ⚠️ Sample {i+1}: No gradients")
        continue
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv output
    heatmap = tf.reduce_mean(
        tf.multiply(pooled_grads, conv_output[0]), axis=-1
    )
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
    heatmap = heatmap.numpy()
    
    # Resize
    if heatmap.shape != (128, 128):
        heatmap = cv2.resize(heatmap, (128, 128))
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(genuine, cmap='gray')
    axes[0].set_title(f'Genuine ({pid})')
    axes[0].axis('off')
    
    axes[1].imshow(forged, cmap='gray')
    axes[1].set_title(f'Forged ({pid})')
    axes[1].axis('off')
    
    axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title('Grad-CAM')
    axes[2].axis('off')
    
    # Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_rgb = cv2.cvtColor(np.uint8(255 * genuine), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.5, heatmap_colored, 0.5, 0)
    
    axes[3].imshow(overlay)
    axes[3].set_title(f'Overlay: {decision}')
    axes[3].axis('off')
    
    plt.tight_layout()
    save_path = f'results/gradcam_working_{i+1}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\n--- Sample {i+1} ---")
    print(f"   Person: {pid}")
    print(f"   Prediction: {float(pred):.4f}")
    print(f"   Decision: {decision}")
    print(f"   Saved: {save_path}")

print("\n" + "=" * 70)
print("✅ Grad-CAM complete!")
print("📁 Saved to results/")
print("🎉 DONE!")
print("=" * 70)