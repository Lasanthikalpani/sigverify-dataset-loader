"""
Task: Explainability Demo with New Model
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

print("=" * 70)
print("🔍 SIGVERIFY - EXPLAINABILITY DEMO")
print("   RQ2: Explainability & Trust")
print("=" * 70)

# ============================================================
# Load data (CEDAR for demo)
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
total = sum(len(p['genuine']) + len(p['forged']) for p in people.values())
print(f"   ✅ Total signatures: {total}")

# ============================================================
# Load model
# ============================================================
print("\n🧠 Loading model...")

# Define custom function (must match training)
def absolute_difference(x):
    return tf.abs(x[0] - x[1])

model = load_model(
    'models/model_v3.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

# ============================================================
# Find base network and conv layer
# ============================================================
print("\n📋 Finding base network and Conv2D layer...")

# Find base network (nested functional model)
base_network = None
for layer in model.layers:
    if hasattr(layer, 'layers') and len(layer.layers) > 0:
        for sub_layer in layer.layers:
            if isinstance(sub_layer, tf.keras.layers.Conv2D):
                base_network = layer
                break
    if base_network is not None:
        break

if base_network is None:
    print("   ❌ No base network found!")
    sys.exit(1)

print(f"   ✅ Base network: {base_network.name}")

# Find last Conv2D layer in base network
conv_layer = None
for layer in reversed(base_network.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        conv_layer = layer
        break

if conv_layer is None:
    print("   ❌ No Conv2D layer found!")
    sys.exit(1)

print(f"   ✅ Conv layer: {conv_layer.name}")

# ============================================================
# Grad-CAM function (CORRECTED)
# ============================================================
def generate_gradcam(model, img1, img2, base_network, conv_layer):
    """Generate Grad-CAM heatmap - CORRECTED for nested models"""
    
    # Create grad model using BASE NETWORK's input
    grad_model = tf.keras.Model(
        inputs=base_network.input,
        outputs=conv_layer.output
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        # Get conv output from base network
        conv_output = grad_model(img1)
        
        # Get prediction from main model
        pred = model([img1, img2])
        loss = pred[:, 0]
    
    # Get gradients
    grads = tape.gradient(loss, conv_output)
    
    if grads is None:
        print("   ⚠️ No gradients available")
        return None
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight conv output
    heatmap = tf.reduce_mean(
        tf.multiply(pooled_grads, conv_output[0]), axis=-1
    )
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
    
    return heatmap.numpy()

# ============================================================
# Generate explanations
# ============================================================
print("\n" + "=" * 70)
print("📋 Grad-CAM Explanations")
print("=" * 70)

num_samples = 3
person_ids = list(people.keys())

for i in range(num_samples):
    pid = np.random.choice(person_ids)
    
    genuine = people[pid]['genuine'][0]
    forged = people[pid]['forged'][0]
    
    genuine_input = genuine.reshape(1, 128, 128, 1)
    forged_input = forged.reshape(1, 128, 128, 1)
    
    # Prediction
    pred = model.predict([genuine_input, forged_input], verbose=0)[0][0]
    decision = "Genuine" if pred > 0.5 else "Forged"
    
    # Heatmap
    heatmap = generate_gradcam(model, genuine_input, forged_input, base_network, conv_layer)
    
    if heatmap is None:
        print(f"\n--- Sample {i+1} ---")
        print(f"   ⚠️ Could not generate heatmap")
        continue
    
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (128, 128))
    
    # Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    image_rgb = cv2.cvtColor(np.uint8(255 * genuine), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.5, heatmap_colored, 0.5, 0)
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(genuine, cmap='gray')
    axes[0].set_title(f'Genuine (Person {pid})')
    axes[0].axis('off')
    
    axes[1].imshow(forged, cmap='gray')
    axes[1].set_title(f'Forged (Person {pid})')
    axes[1].axis('off')
    
    axes[2].imshow(heatmap_resized, cmap='jet')
    axes[2].set_title('Grad-CAM Heatmap')
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title(f'Overlay\nDecision: {decision}')
    axes[3].axis('off')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    save_path = f'results/gradcam_final_{i+1}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\n--- Sample {i+1} ---")
    print(f"   Person: {pid}")
    print(f"   Prediction: {pred:.4f}")
    print(f"   Decision: {decision}")
    print(f"   Saved: {save_path}")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
✅ Used {len(people)} people
✅ Generated {num_samples} Grad-CAM explanations
✅ Saved to 'results/' folder

📁 Files:
   - results/gradcam_final_1.png
   - results/gradcam_final_2.png
   - results/gradcam_final_3.png
""")
print("🎉 DONE!")