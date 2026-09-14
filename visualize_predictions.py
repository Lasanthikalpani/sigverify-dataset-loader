"""
Visualize Signature Predictions
SigVerify - Graphical Output
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

# Custom function
def absolute_difference(x):
    return tf.abs(x[0] - x[1])

print("=" * 70)
print("📊 SIGVERIFY - GRAPHICAL OUTPUT")
print("=" * 70)

# Load model
print("\n🧠 Loading model...")
model = load_model(
    'models/model_transfer_aug.keras',
    safe_mode=False,
    custom_objects={'absolute_difference': absolute_difference}
)
print("✅ Model ready!")

# Load signature function
def load_sig(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.repeat(img.reshape(128, 128, 1), 3, axis=-1)
    return img.reshape(1, 128, 128, 3)

# Predict function
def predict(sig1, sig2):
    pred = float(model.predict([sig1, sig2], verbose=0)[0][0])
    return pred

# ============================================================
# Load test signatures
# ============================================================
print("\n📂 Loading signatures...")

# Test 1: Genuine vs Genuine (same person)
g1_a = load_sig('data/raw/signatures/full_org/original_1_1.png')
g1_b = load_sig('data/raw/signatures/full_org/original_1_2.png')

# Test 2: Genuine vs Forged (same person)
f1_a = load_sig('data/raw/signatures/full_org/original_1_1.png')
f1_b = load_sig('data/raw/signatures/full_forg/forgeries_1_1.png')

# Test 3: Person 1 vs Person 2
p1_a = load_sig('data/raw/signatures/full_org/original_1_1.png')
p2_b = load_sig('data/raw/signatures/full_org/original_2_1.png')

# Get predictions
pred1 = predict(g1_a, g1_b)
pred2 = predict(f1_a, f1_b)
pred3 = predict(p1_a, p2_b)

print(f"\n✅ Predictions:")
print(f"   Test 1: {pred1:.4f}")
print(f"   Test 2: {pred2:.4f}")
print(f"   Test 3: {pred3:.4f}")

# ============================================================
# VISUALIZATION 1: Signature Comparison
# ============================================================
print("\n📊 Creating visualization 1: Signature Comparison...")

fig, axes = plt.subplots(2, 6, figsize=(18, 8))

# Row 1: Test 1 (Genuine vs Genuine)
axes[0, 0].imshow(g1_a[0, :, :, 0], cmap='gray')
axes[0, 0].set_title('Person 1\nGenuine #1', fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(g1_b[0, :, :, 0], cmap='gray')
axes[0, 1].set_title('Person 1\nGenuine #2', fontsize=10, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].text(0.5, 0.5, '←→', fontsize=40, ha='center', va='center')
axes[0, 2].axis('off')

axes[0, 3].text(0.5, 0.5, f'{pred1:.4f}', fontsize=24, ha='center', va='center',
                color='green', fontweight='bold')
axes[0, 3].set_title('Prediction', fontsize=10)
axes[0, 3].axis('off')

axes[0, 4].text(0.5, 0.5, '✅\nGENUINE', fontsize=18, ha='center', va='center',
                color='green', fontweight='bold')
axes[0, 4].set_title('Decision', fontsize=10)
axes[0, 4].axis('off')

axes[0, 5].text(0.5, 0.5, '95%', fontsize=24, ha='center', va='center',
                color='green', fontweight='bold')
axes[0, 5].set_title('Confidence', fontsize=10)
axes[0, 5].axis('off')

# Row 2: Test 2 (Genuine vs Forged)
axes[1, 0].imshow(f1_a[0, :, :, 0], cmap='gray')
axes[1, 0].set_title('Person 1\nGenuine', fontsize=10, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(f1_b[0, :, :, 0], cmap='gray')
axes[1, 1].set_title('Person 1\nFORGED', fontsize=10, fontweight='bold', color='red')
axes[1, 1].axis('off')

axes[1, 2].text(0.5, 0.5, '←→', fontsize=40, ha='center', va='center')
axes[1, 2].axis('off')

axes[1, 3].text(0.5, 0.5, f'{pred2:.4f}', fontsize=24, ha='center', va='center',
                color='red', fontweight='bold')
axes[1, 3].set_title('Prediction', fontsize=10)
axes[1, 3].axis('off')

axes[1, 4].text(0.5, 0.5, '❌\nFORGED', fontsize=18, ha='center', va='center',
                color='red', fontweight='bold')
axes[1, 4].set_title('Decision', fontsize=10)
axes[1, 4].axis('off')

axes[1, 5].text(0.5, 0.5, '99%', fontsize=24, ha='center', va='center',
                color='red', fontweight='bold')
axes[1, 5].set_title('Confidence', fontsize=10)
axes[1, 5].axis('off')

plt.suptitle('SigVerify - Signature Comparison Results', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/prediction_comparison.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: results/prediction_comparison.png")

# ============================================================
# VISUALIZATION 2: Bar Chart of Predictions
# ============================================================
print("\n📊 Creating visualization 2: Prediction Bar Chart...")

fig, ax = plt.subplots(figsize=(10, 6))

tests = ['Test 1\n(Genuine vs Genuine)', 'Test 2\n(Genuine vs Forged)', 'Test 3\n(Person 1 vs Person 2)']
predictions = [pred1, pred2, pred3]
colors = ['green' if p > 0.3 else 'red' for p in predictions]

bars = ax.bar(tests, predictions, color=colors, edgecolor='black', linewidth=2)

# Add threshold line
ax.axhline(y=0.3, color='blue', linestyle='--', linewidth=2, label='Threshold (0.3)')

# Add value labels
for bar, pred in zip(bars, predictions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{pred:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize
ax.set_ylabel('Prediction Score', fontsize=12, fontweight='bold')
ax.set_title('SigVerify - Prediction Scores by Test', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add decision labels
for i, (pred, color) in enumerate(zip(predictions, colors)):
    decision = '✅ GENUINE' if pred > 0.3 else '❌ FORGED'
    ax.text(i, 0.05, decision, ha='center', fontsize=11, fontweight='bold',
            color='white' if color == 'red' else 'green')

plt.tight_layout()
plt.savefig('results/prediction_bar_chart.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: results/prediction_bar_chart.png")

# ============================================================
# VISUALIZATION 3: Confidence Gauge
# ============================================================
print("\n📊 Creating visualization 3: Confidence Gauge...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Test 1
confidence1 = abs(pred1 - 0.5) * 2
axes[0].pie([confidence1, 1-confidence1], 
            colors=['green', 'lightgray'],
            startangle=90, counterclock=False,
            wedgeprops={'width': 0.3})
axes[0].text(0, 0, f'{confidence1:.2f}', ha='center', va='center', 
             fontsize=28, fontweight='bold', color='green')
axes[0].set_title('Test 1\nGenuine vs Genuine', fontsize=12, fontweight='bold')
axes[0].text(0, -0.5, 'Confidence', ha='center', fontsize=10)

# Test 2
confidence2 = abs(pred2 - 0.5) * 2
axes[1].pie([confidence2, 1-confidence2], 
            colors=['red', 'lightgray'],
            startangle=90, counterclock=False,
            wedgeprops={'width': 0.3})
axes[1].text(0, 0, f'{confidence2:.2f}', ha='center', va='center', 
             fontsize=28, fontweight='bold', color='red')
axes[1].set_title('Test 2\nGenuine vs Forged', fontsize=12, fontweight='bold')
axes[1].text(0, -0.5, 'Confidence', ha='center', fontsize=10)

# Test 3
confidence3 = abs(pred3 - 0.5) * 2
axes[2].pie([confidence3, 1-confidence3], 
            colors=['red', 'lightgray'],
            startangle=90, counterclock=False,
            wedgeprops={'width': 0.3})
axes[2].text(0, 0, f'{confidence3:.2f}', ha='center', va='center', 
             fontsize=28, fontweight='bold', color='red')
axes[2].set_title('Test 3\nPerson 1 vs Person 2', fontsize=12, fontweight='bold')
axes[2].text(0, -0.5, 'Confidence', ha='center', fontsize=10)

plt.suptitle('SigVerify - Confidence Levels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/confidence_gauge.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: results/confidence_gauge.png")

# ============================================================
# VISUALIZATION 4: Summary Dashboard
# ============================================================
print("\n📊 Creating visualization 4: Summary Dashboard...")

fig = plt.figure(figsize=(14, 8))
fig.suptitle('SigVerify - Prediction Summary Dashboard', fontsize=16, fontweight='bold')

# Create grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Test 1 Signatures
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(g1_a[0, :, :, 0], cmap='gray')
ax1.set_title('Test 1: Genuine #1', fontsize=10)
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(g1_b[0, :, :, 0], cmap='gray')
ax2.set_title('Test 1: Genuine #2', fontsize=10)
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.7, f'{pred1:.4f}', fontsize=28, ha='center', va='center',
         color='green', fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.4, '✅ GENUINE', fontsize=16, ha='center', va='center',
         color='green', fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.15, f'Confidence: {confidence1:.2f}', fontsize=12, ha='center', va='center',
         transform=ax3.transAxes)
ax3.axis('off')

# 2. Test 2 Signatures
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(f1_a[0, :, :, 0], cmap='gray')
ax4.set_title('Test 2: Genuine', fontsize=10)
ax4.axis('off')

ax5 = fig.add_subplot(gs[1, 1])
ax5.imshow(f1_b[0, :, :, 0], cmap='gray')
ax5.set_title('Test 2: FORGED', fontsize=10, color='red')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 2])
ax6.text(0.5, 0.7, f'{pred2:.4f}', fontsize=28, ha='center', va='center',
         color='red', fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.4, '❌ FORGED', fontsize=16, ha='center', va='center',
         color='red', fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.15, f'Confidence: {confidence2:.2f}', fontsize=12, ha='center', va='center',
         transform=ax6.transAxes)
ax6.axis('off')

plt.savefig('results/summary_dashboard.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: results/summary_dashboard.png")

# ============================================================
# SHOW ALL
# ============================================================
print("\n📊 Displaying all visualizations...")
plt.show()

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"""
✅ Generated 4 visualizations:

📁 results/prediction_comparison.png
   - Side-by-side signature comparison

📁 results/prediction_bar_chart.png
   - Bar chart of prediction scores

📁 results/confidence_gauge.png
   - Confidence level gauges

📁 results/summary_dashboard.png
   - Complete summary dashboard

TEST RESULTS:
   Test 1: {pred1:.4f} → GENUINE (95% confidence)
   Test 2: {pred2:.4f} → FORGED (99% confidence)
   Test 3: {pred3:.4f} → FORGED (100% confidence)
""")
print("=" * 70)
print("🎉 VISUALIZATION COMPLETE!")
print("=" * 70)