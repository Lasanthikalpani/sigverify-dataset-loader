"""
Task: Accuracy Test
SigVerify - RQ1 Core AI Performance

This task demonstrates:
- RQ1: Core AI Performance
- Siamese CNN accuracy testing
- Forgery detection results
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from tensorflow.keras.models import load_model
from data_loader import SignatureDatasetLoader

print("=" * 70)
print("📊 SIGVERIFY - ACCURACY TEST")
print("   RQ1: Core AI Performance")
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

# Test accuracy
print("\n" + "=" * 70)
print("📋 Testing Accuracy on 20 Random Pairs")
print("=" * 70)

correct = 0
total = 20
results = []

for i in range(total):
    idx = np.random.randint(0, min(len(loader.genuine_images), len(loader.forged_images)))
    genuine = loader.genuine_images[idx]
    forged = loader.forged_images[idx]
    
    pred = model.predict([genuine.reshape(1,128,128,1), forged.reshape(1,128,128,1)], verbose=0)[0][0]
    decision = "Genuine" if pred > 0.5 else "Forged"
    is_correct = decision == "Forged"
    correct += 1 if is_correct else 0
    results.append((i+1, pred, decision, is_correct))

print("\nResults:")
for r in results:
    status = "✅" if r[3] else "❌"
    print(f"   Pair {r[0]:2d}: Prediction={r[1]:.4f} → {r[2]} {status}")

accuracy = correct / total * 100
print(f"\n📊 Accuracy: {accuracy:.1f}%")
print(f"   Target: ≥92%")
print(f"   {'✅ PASSED' if accuracy >= 92 else '❌ FAILED'}")

# Summary
print("\n" + "=" * 70)
print("📋 RQ1 COMPLETION SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  RQ1: Core AI Performance                                   │
│  ────────────────────────                                   │
│  Target:           ≥92%                                     │
│  Achieved:         {accuracy:.1f}%                                   │
│  Status:           {'✅ PASS' if accuracy >= 92 else '❌ FAIL'}                 │
│  Samples Tested:   {total}                                         │
└─────────────────────────────────────────────────────────────┘
""")

print("=" * 70)