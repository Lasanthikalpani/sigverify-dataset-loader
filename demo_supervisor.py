"""
SUPERVISOR DEMO - SigVerify
Explainable AI for Signature Forgery Detection
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
print("🎓 SIGVERIFY - SUPERVISOR DEMO")
print("   Explainable AI for Signature Forgery Detection")
print("=" * 70)

# ============================================================
# 1. LOAD MODEL
# ============================================================
print("\n📋 STEP 1: Loading Trained Model")
print("-" * 70)

model = load_model(
    'models/model_cedar.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print(f"   ✅ Model loaded successfully!")
print(f"   📊 Parameters: {model.count_params():,}")
print(f"   🏗️ Architecture: Siamese CNN")

# ============================================================
# 2. LOAD TEST DATA
# ============================================================
print("\n📋 STEP 2: Loading Test Data")
print("-" * 70)

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
print(f"   📊 Total signatures: {sum(len(p['genuine']) + len(p['forged']) for p in people.values())}")

# ============================================================
# 3. RUN ACCURACY TEST
# ============================================================
print("\n📋 STEP 3: Testing Accuracy")
print("-" * 70)

threshold = 0.3
correct_genuine = 0
correct_forged = 0
total = 0

print("\n   Testing each person:")
print("   " + "-" * 50)

for pid in sorted(people.keys()):
    pred1 = float(model.predict([
        people[pid]['genuine'][0].reshape(1, 128, 128, 1),
        people[pid]['genuine'][1].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    pred2 = float(model.predict([
        people[pid]['genuine'][0].reshape(1, 128, 128, 1),
        people[pid]['forged'][0].reshape(1, 128, 128, 1)
    ], verbose=0)[0][0])
    
    g_status = "✅" if pred1 > threshold else "❌"
    f_status = "✅" if pred2 < threshold else "❌"
    
    if pred1 > threshold:
        correct_genuine += 1
    if pred2 < threshold:
        correct_forged += 1
    
    total += 2
    
    print(f"   Person {pid}: Genuine={pred1:.4f} {g_status} | Forged={pred2:.4f} {f_status}")

accuracy = (correct_genuine + correct_forged) / total * 100

print("\n   " + "-" * 50)
print(f"   ✅ Genuine: {correct_genuine}/{len(people)} correct")
print(f"   ✅ Forged:  {correct_forged}/{len(people)} correct")
print(f"   📊 Overall Accuracy: {accuracy:.1f}%")

# ============================================================
# 4. GENERATE GRAD-CAM EXPLANATIONS
# ============================================================
print("\n📋 STEP 4: Generating Grad-CAM Explanations")
print("-" * 70)

# Find base network and conv layer
base = model.get_layer('functional')
conv_layer = base.get_layer('conv2d_2')

# Create grad model
grad_model = tf.keras.Model(inputs=base.input, outputs=conv_layer.output)

print(f"   ✅ Base network: {base.name}")
print(f"   ✅ Conv layer: {conv_layer.name}")

os.makedirs('results/demo', exist_ok=True)

# Generate for 3 samples
person_ids = list(people.keys())[:3]

for i, pid in enumerate(person_ids):
    genuine = people[pid]['genuine'][0]
    forged = people[pid]['forged'][0]
    
    genuine_input = genuine.reshape(1, 128, 128, 1)
    forged_input = forged.reshape(1, 128, 128, 1)
    
    # Prediction
    pred = float(model.predict([genuine_input, forged_input], verbose=0)[0][0])
    decision = "Genuine" if pred > threshold else "Forged"
    
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
    
    # Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    image_rgb = cv2.cvtColor(np.uint8(255 * genuine), cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.5, heatmap_colored, 0.5, 0)
    
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
    
    axes[3].imshow(overlay)
    axes[3].set_title(f'Overlay: {decision}')
    axes[3].axis('off')
    
    plt.tight_layout()
    save_path = f'results/demo/demo_{i+1}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"   Sample {i+1}: Person {pid} → {decision} ({pred:.4f})")

# ============================================================
# 5. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📋 DEMO SUMMARY")
print("=" * 70)

print(f"""
✅ MODEL
   - Architecture: Siamese CNN
   - Parameters: {model.count_params():,}
   - Model file: model_cedar.keras

✅ DATASET
   - People: {len(people)}
   - Signatures: {sum(len(p['genuine']) + len(p['forged']) for p in people.values())}
   - Source: CEDAR

✅ PERFORMANCE
   - Optimal threshold: {threshold}
   - Genuine recognition: {correct_genuine}/{len(people)} ({correct_genuine/len(people)*100:.1f}%)
   - Forged detection: {correct_forged}/{len(people)} ({correct_forged/len(people)*100:.1f}%)
   - Overall accuracy: {accuracy:.1f}%

✅ EXPLAINABILITY
   - Grad-CAM heatmaps generated
   - Visual explanations saved
   - Results in 'results/demo/' folder

✅ RESEARCH QUESTIONS
   - RQ1: Core AI Performance - COMPLETE
   - RQ2: Explainability & Trust - COMPLETE
""")

print("=" * 70)
print("🎉 SUPERVISOR DEMO READY!")
print("=" * 70)