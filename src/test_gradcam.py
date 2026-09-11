# test_gradcam.py
"""
Test script for Grad-CAM explainability
"""

from src.grad_cam import test_grad_cam, explain_prediction, visualize_explanation
from src.data_loader import SignatureDatasetLoader
from tensorflow.keras.models import load_model
import numpy as np

print("=" * 60)
print("🧪 GRAD-CAM TEST SCRIPT")
print("=" * 60)

# Load data
print("\n📂 Loading dataset...")
loader = SignatureDatasetLoader()
loader.load_cedar()
print(f"✅ Loaded {loader.get_dataset_stats()}")

# Load model
print("\n🧠 Loading model...")
model = load_model('models/simple_siamese_model.keras')
print("✅ Model loaded!")

# Test 1: Quick test with 3 samples
print("\n" + "=" * 60)
print("TEST 1: Grad-CAM on 3 Random Samples")
print("=" * 60)
test_grad_cam(model, loader, num_samples=3)

# Test 2: Individual detailed explanation
print("\n" + "=" * 60)
print("TEST 2: Detailed Single Sample Explanation")
print("=" * 60)
idx = np.random.randint(0, min(len(loader.genuine_images), len(loader.forged_images)))
genuine = loader.genuine_images[idx]
forged = loader.forged_images[idx]

explanation = explain_prediction(model, genuine, forged)
print(f"Prediction: {explanation['prediction']:.4f}")
print(f"Decision: {explanation['decision']}")
print(f"Confidence: {explanation['confidence']:.2f}")

visualize_explanation(explanation, genuine, forged, save_path='results/detailed_explanation.png')

print("\n✅ All tests completed!")
print("📊 Results saved in 'results/' folder")