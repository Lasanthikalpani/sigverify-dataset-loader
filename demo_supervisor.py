"""
Complete Demo Script for Supervisor
SigVerify - Signature Verification with Explainable AI
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

sys.path.insert(0, 'src')
from data_loader import SignatureDatasetLoader
from grad_cam import test_grad_cam, explain_prediction, visualize_explanation

print("=" * 70)
print("🎓 SIGVERIFY - SUPERVISOR DEMO")
print("   Signature Verification with Explainable AI")
print("=" * 70)

print("\n📋 1. Loading Dataset...")
loader = SignatureDatasetLoader()
loader.load_cedar()
stats = loader.get_dataset_stats()
print(f"   ✅ Genuine signatures: {stats['num_genuine']}")
print(f"   ✅ Forged signatures: {stats['num_forged']}")

print("\n📋 2. Loading Trained Model...")
model = load_model('models/gradcam_model.keras')
print(f"   ✅ Model loaded! Parameters: {model.count_params():,}")

print("\n📋 3. Testing Prediction Accuracy...")
print("   Testing 10 random pairs...")

correct = 0
for i in range(10):
    idx = np.random.randint(0, min(len(loader.genuine_images), len(loader.forged_images)))
    genuine = loader.genuine_images[idx]
    forged = loader.forged_images[idx]
    
    pred = model.predict([genuine.reshape(1,128,128,1), forged.reshape(1,128,128,1)], verbose=0)[0][0]
    decision = "Genuine" if pred > 0.5 else "Forged"
    correct += 1 if decision == "Forged" else 0
    print(f"      Pair {i+1}: Prediction={pred:.4f} → {decision} {'✅' if decision == 'Forged' else '❌'}")

print(f"\n   📊 Accuracy: {correct/10*100:.1f}% on forged detection")

print("\n📋 4. Grad-CAM Explainability Demo...")
print("   Generating visual explanations for 3 samples...\n")

# Run Grad-CAM test
test_grad_cam(model, loader, num_samples=3)

print("\n" + "=" * 70)
print("✅ DEMO COMPLETE!")
print("   All visual explanations saved in 'results/' folder")
print("=" * 70)
print("\n📁 Files generated:")
print("   - results/grad_cam_sample_1.png")
print("   - results/grad_cam_sample_2.png")
print("   - results/grad_cam_sample_3.png")