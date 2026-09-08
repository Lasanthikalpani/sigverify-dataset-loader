"""
Simple test script for SignatureDatasetLoader
"""

from src.data_loader import SignatureDatasetLoader

print("=" * 50)
print("SigVerify Dataset Loader Test")
print("=" * 50)

# Initialize loader
loader = SignatureDatasetLoader(data_dir='data', verbose=True)

# Print instructions
print("\n📌 Available Methods:")
print("  - loader.download_cedar()  → Download CEDAR dataset")
print("  - loader.load_cedar()      → Load CEDAR dataset")
print("  - loader.create_pairs()    → Create training pairs")
print("  - loader.visualize_samples() → View sample signatures")
print("  - loader.get_dataset_stats() → Get dataset statistics")

print("\n✅ Test completed successfully!")