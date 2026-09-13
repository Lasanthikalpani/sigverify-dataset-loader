"""
Grad-CAM for model_v4
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
print("🔍 SIGVERIFY - GRAD-CAM (v4)")
print("=" * 70)

# Load model
print("\n🧠 Loading model...")
model = load_model(
    'models/model_v4.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

# Find base network and conv layer
print("\n📋 Finding layers...")
base_network = model.get_layer('base_network')
conv_layer = base_network.get_layer('conv3')
print(f"   ✅ Base: {base_network.name}")
print(f"   ✅ Conv: {conv_layer.name}")

# Create grad model
grad_model = tf.keras.Model(
    inputs=base_network.input,
    outputs=conv_layer.output
)
print("   ✅ Grad model created")

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

print(f"   ✅ Loaded {len(people)} people")

# Generate Grad-CAM
print("\n📋 Generating Grad-CAM...")
os.makedirs('results', exist_ok=True)

for i, pid in enumerate(list(people.keys())[:3]):
    genuine = people[pid]['genuine'][0]
    forged = people[pid]['forged'][0]
    
    genuine_input = genuine.reshape(1, 128, 128, 1)
    forged_input = forged.reshape(1, 128, 128, 1)
    
    # Prediction
    pred = model.predict([genuine_input, forged_input], verbose=0)[0][0]
    decision = "Genuine" if pred > 0.5 else "Forged"
    
    # Grad-CAM
    with tf.GradientTape() as tape:
        conv_output = grad_model(genuine_input)
        tape.watch(conv_output)
        # Compute prediction
        pred_tensor = model([genuine_input, forged_input])
        loss = pred_tensor[:, 0]
    
    grads = tape.gradient(loss, conv_output)
    
    if grads is None:
        print(f"   ⚠️ Sample {i+1}: No gradients")
        continue
    
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
    save_path = f'results/gradcam_v4_{i+1}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"   Sample {i+1}: {decision} ({pred:.4f}) → {save_path}")

print("\n✅ Grad-CAM complete!")
print("📁 Saved to results/")
print("🎉 DONE!")