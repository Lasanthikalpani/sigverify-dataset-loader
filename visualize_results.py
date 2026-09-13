"""
Visualize Grad-CAM Results
SigVerify - RQ2: Explainability & Trust
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

print("=" * 70)
print("📊 SIGVERIFY - VISUALIZE GRAD-CAM RESULTS")
print("=" * 70)

# Check results folder
results_dir = Path('results')

if not results_dir.exists():
    print("❌ Results folder not found!")
    exit()

# List all images
images = sorted(list(results_dir.glob('gradcam_*.png')))

print(f"\n📁 Found {len(images)} Grad-CAM images:")
for img in images:
    print(f"   - {img.name}")

# Create grid visualization
print("\n📊 Creating visualization grid...")

num_images = len(images)
if num_images == 0:
    print("❌ No images found!")
    exit()

# Calculate grid size
cols = 2
rows = (num_images + 1) // 2

fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))

if rows == 1:
    axes = [axes]

for i, img_path in enumerate(images):
    row = i // cols
    col = i % cols
    
    if rows == 1:
        ax = axes[col]
    else:
        ax = axes[row, col]
    
    img = mpimg.imread(str(img_path))
    ax.imshow(img)
    ax.set_title(img_path.name, fontsize=12)
    ax.axis('off')

# Hide unused subplots
for i in range(num_images, rows * cols):
    row = i // cols
    col = i % cols
    if rows == 1:
        axes[col].axis('off')
    else:
        axes[row, col].axis('off')

plt.suptitle('SigVerify - Grad-CAM Results\nRQ2: Explainability & Trust', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/visualization_grid.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualization saved!")
print("📁 results/visualization_grid.png")
print("🎉 DONE!")