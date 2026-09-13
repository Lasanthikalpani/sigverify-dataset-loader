"""
Task: Data Collection and Augmentation
SigVerify - Achieving 1000+ Original and 5000+ Augmented Signatures
"""

import sys
import os
import numpy as np

sys.path.insert(0, 'src')
from data_collection import SignatureDataCollector, SignatureAugmenter

print("=" * 70)
print("📊 SIGVERIFY - DATA COLLECTION & AUGMENTATION")
print("   Target: 1000+ Original, 5000+ Augmented")
print("=" * 70)

# ============================================================
# Task 1: Collect Original Data
# ============================================================
print("\n" + "=" * 70)
print("📋 TASK 1: Original Data Collection")
print("=" * 70)

collector = SignatureDataCollector(data_dir='data')
genuine, forged = collector.load_all()

original_genuine = len(genuine)
original_forged = len(forged)
original_total = original_genuine + original_forged

print(f"\n📊 Original Dataset:")
print(f"   Genuine: {original_genuine}")
print(f"   Forged: {original_forged}")
print(f"   Total: {original_total}")

# ============================================================
# Task 2: Apply Augmentation
# ============================================================
print("\n" + "=" * 70)
print("📋 TASK 2: Data Augmentation")
print("=" * 70)

augmenter = SignatureAugmenter()
genuine_aug, forged_aug = augmenter.augment_all(genuine, forged, target_total=5000)

# ============================================================
# Task 3: Save Augmented Data
# ============================================================
print("\n" + "=" * 70)
print("📋 TASK 3: Saving Augmented Data")
print("=" * 70)

os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/genuine_augmented.npy', genuine_aug)
np.save('data/processed/forged_augmented.npy', forged_aug)

print(f"   ✅ Saved to data/processed/")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("📋 DATA COLLECTION SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  ORIGINAL DATA                                              │
│  ─────────────                                              │
│  Genuine:                    {original_genuine}                            │
│  Forged:                     {original_forged}                            │
│  Total:                      {original_total}                            │
├─────────────────────────────────────────────────────────────┤
│  AUGMENTED DATA                                             │
│  ──────────────                                             │
│  Genuine:                    {len(genuine_aug)}                           │
│  Forged:                     {len(forged_aug)}                           │
│  Total:                      {len(genuine_aug) + len(forged_aug)}                           │
├─────────────────────────────────────────────────────────────┤
│  AUGMENTATION FACTOR                                        │
│  ───────────────────                                        │
│  Factor:                     {(len(genuine_aug) + len(forged_aug)) / original_total:.1f}x                          │
└─────────────────────────────────────────────────────────────┘
""")

print("✅ DATA COLLECTION COMPLETE!")
print("=" * 70)