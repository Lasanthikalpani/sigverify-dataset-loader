"""
Grad-CAM with Working Model
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

def absolute_difference(x):
    return tf.abs(x[0] - x[1])

print("=" * 70)
print("🔍 SIGVERIFY - GRAD-CAM (WORKING MODEL)")
print("=" * 70)

# Load model
model = load_model(
    'models/model_working.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"✅ Model loaded! Parameters: {model.count_params():,}")

# Get base network and conv layer
base = None
for layer in model.layers:
    if hasattr(layer, 'layers') and len(layer.layers) > 0:
        for sub in layer.layers:
            if isinstance(sub, tf.keras.layers.Conv2D):
                base = layer
                break
    if base:
        break

if base is None:
    print("❌ No base network found!")
    exit()

conv_layer = None
for sub in reversed(base.layers):
    if isinstance(sub, tf.keras.layers.Conv2D):
        conv_layer = sub
        break

print(f"✅ Base: {base.name}")
print(f"✅ Conv: {conv_layer.name}")

# Create grad model
grad_model = tf.keras.Model(inputs=base.input, outputs=conv_layer.output)
print("✅ Grad model created")

# Load CEDAR data
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

print(f"✅ Loaded {len(people)} people")

# Generate Grad-CAM
os.makedirs('results', exist_ok=True)
person_ids = list(people.keys())

print("\n📋 Generating Grad-CAM...\n")

for i, pid in enumerate(person_ids[:3]):
    # Genuine vs Forged
    genuine = people[pid]['genuine'][0]
    forged = people[pid]['forged'][0]
    
    genuine_input = genuine.reshape(1, 128, 128, 1)
    forged_input = forged.reshape(1, 128, 128, 1)
    
    pred = float(model.predict([genuine_input, forged_input], verbose=0)[0][0])
    decision = "Genuine" if pred > 0.5 else "Forged"
    
    # Grad-CAM
    with tf.GradientTape() as tape:
        conv_output = grad_model(genuine_input)
        loss = tf.reduce_mean(conv_output)
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
    heatmap = heatmap.numpy()
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
    axes[2].set_title('Grad-CAM Heatmap')
    axes[2].axis('off')
    
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
    
    print(f"--- Sample {i+1} ---")
    print(f"   Person: {pid}")
    print(f"   Prediction: {pred:.4f}")
    print(f"   Decision: {decision}")
    print(f"   Saved: {save_path}\n")

print("=" * 70)
print("✅ Grad-CAM complete!")
print("📁 Saved to results/")
print("🎉 DONE!")