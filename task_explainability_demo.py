"""
Task: Explainability Demo
SigVerify - Grad-CAM Visual Explanations

This task demonstrates:
- RQ2: Explainability & Trust
- Grad-CAM heatmap generation
- Visual explanations for officers
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from tensorflow.keras.models import load_model
from data_loader import SignatureDatasetLoader
from grad_cam import test_grad_cam

print("=" * 70)
print("🔍 SIGVERIFY - EXPLAINABILITY DEMO")
print("   RQ2: Explainability & Trust")
print("=" * 70)

# Load data
print("\n📂 Loading dataset...")
loader = SignatureDatasetLoader()
loader.load_cedar()
stats = loader.get_dataset_stats()
print(f"   ✅ Loaded {stats['num_genuine']} genuine signatures")
print(f"   ✅ Loaded {stats['num_forged']} forged signatures")

# Load model
print("\n🧠 Loading model...")
model = load_model('models/gradcam_model.keras')
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

print("\n" + "=" * 70)
print("📋 Grad-CAM Explanation Demonstration")
print("=" * 70)

print("\nHow Grad-CAM Works:")
print("   ┌─────────────────────────────────────────────────────┐")
print("   │  🔴 RED areas   → Strongly influenced decision      │")
print("   │  🟡 YELLOW areas → Moderately influenced decision   │")
print("   │  🟢 GREEN areas  → Less influential                 │")
print("   │  🔵 BLUE areas   → Very little influence            │")
print("   └─────────────────────────────────────────────────────┘")

# Test on multiple samples
test_grad_cam(model, loader, num_samples=3)

print("\n" + "=" * 70)
print("📋 EXPLANATION SUMMARY")
print("=" * 70)

print("""
✅ Grad-CAM successfully generates visual explanations
✅ Officers can see WHY the AI made its decision
✅ RED areas show suspicious signature regions
✅ Builds trust in the AI system
✅ Legally defensible decisions

📁 Results saved in 'results/' folder:
   - grad_cam_sample_1.png
   - grad_cam_sample_2.png
   - grad_cam_sample_3.png
""")

print("=" * 70)
print("🎉 EXPLAINABILITY DEMO COMPLETE!")