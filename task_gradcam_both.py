"""
Grad-CAM - Both Genuine and Forged
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
    return tf.abs(x)

print("=" * 70)
print("🔍 SIGVERIFY - GRAD-CAM (BOTH GENUINE & FORGED)")
print("=" * 70)

# Load model
model = load_model(
    'models/model_v3.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"✅ Model loaded! Parameters: {model.count_params():,}")

# Setup Grad-CAM
base = model.get_layer('functional')
conv_layer = base.get_layer('conv2d_2')
grad_model = tf.keras.Model(inputs=base.input, outputs=conv_layer.output)
print("✅ Grad model created")

# Load CEDAR
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

print("\n📋 Generating Grad-CAM for BOTH cases...\n")

# Case 1: Genuine vs Genuine (should be Genuine)
print("=" * 70)
print("CASE 1: GENUINE vs GENUINE (should be Genuine)")
print("=" * 70)

for i in range(2):
    pid = person_ids[i]
    img1 = people[pid]['genuine'][0]
    img2 = people[pid]['genuine'][1]
    
    img1_input = img1.reshape(1, 128, 128, 1)
    img2_input = img2.reshape(1, 128, 128, 1)
    
    pred = float(model.predict([img1_input, img2_input], verbose=0)[0][0])
    decision = "Genuine" if pred > 0.5 else "Forged"
    
    print(f"\n--- Genuine {i+1} (Person {pid}) ---")
    print(f"   Prediction: {pred:.4f}")
    print(f"   Decision: {decision}")
    
    # Grad-CAM
    with tf.GradientTape() as tape:
        conv_output = grad_model(img1_input)
        loss = tf.reduce_mean(conv_output)
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (128, 128))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(f'Genuine #1 ({pid})')
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(f'Genuine #2 ({pid})')
    axes[1].axis('off')
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_rgb = cv2.cvtColor(np.uint8(255 * img1), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.5, heatmap_colored, 0.5, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay: {decision}')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = f'results/gradcam_genuine_{i+1}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"   Saved: {save_path}")

# Case 2: Genuine vs Forged (should be Forged)
print("\n" + "=" * 70)
print("CASE 2: GENUINE vs FORGED (should be Forged)")
print("=" * 70)

for i in range(2):
    pid = person_ids[i]
    img1 = people[pid]['genuine'][0]
    img2 = people[pid]['forged'][0]
    
    img1_input = img1.reshape(1, 128, 128, 1)
    img2_input = img2.reshape(1, 128, 128, 1)
    
    pred = float(model.predict([img1_input, img2_input], verbose=0)[0][0])
    decision = "Genuine" if pred > 0.5 else "Forged"
    
    print(f"\n--- Forged {i+1} (Person {pid}) ---")
    print(f"   Prediction: {pred:.4f}")
    print(f"   Decision: {decision}")
    
    # Grad-CAM
    with tf.GradientTape() as tape:
        conv_output = grad_model(img1_input)
        loss = tf.reduce_mean(conv_output)
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (128, 128))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(f'Genuine ({pid})')
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(f'Forged ({pid})')
    axes[1].axis('off')
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_rgb = cv2.cvtColor(np.uint8(255 * img1), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.5, heatmap_colored, 0.5, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay: {decision}')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = f'results/gradcam_forged_{i+1}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"   Saved: {save_path}")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print("""
✅ Generated Grad-CAM for BOTH cases:

📁 Genuine cases:
   - results/gradcam_genuine_1.png
   - results/gradcam_genuine_2.png

📁 Forged cases:
   - results/gradcam_forged_1.png
   - results/gradcam_forged_2.png
""")
print("🎉 DONE!")