"""
Test script for Grad-CAM explainability
"""

import sys
import os
sys.path.insert(0, 'src')

from grad_cam import test_grad_cam, explain_prediction, visualize_explanation
from data_loader import SignatureDatasetLoader
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

print("=" * 60)
print("🧪 GRAD-CAM TEST SCRIPT")
print("=" * 60)

# Load data
print("\n📂 Loading dataset...")
loader = SignatureDatasetLoader()
loader.load_cedar()
stats = loader.get_dataset_stats()
print(f"✅ Loaded {stats['num_genuine']} genuine and {stats['num_forged']} forged signatures")

# Load model - with better error handling
print("\n🧠 Loading model...")

model_path = 'models/gradcam_model.keras'
print(f"Looking for model at: {model_path}")

if os.path.exists(model_path):
    print(f"✅ Model file found! Size: {os.path.getsize(model_path)/1024:.2f} KB")
    try:
        # Allow unsafe deserialization for Lambda layers
        model = load_model(model_path, safe_mode=False)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Try with unsafe deserialization
        try:
            import keras
            keras.config.enable_unsafe_deserialization()
            model = load_model(model_path)
            print("✅ Model loaded with unsafe deserialization!")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            exit()
else:
    print(f"❌ Model not found at '{model_path}'")
    print(f"   Current directory: {os.getcwd()}")
    print("   Files in models folder:", os.listdir('models') if os.path.exists('models') else "models folder not found")
    exit()

# Test 1: Quick test with 3 samples
print("\n" + "=" * 60)
print("TEST 1: Grad-CAM on 3 Random Samples")
print("=" * 60)
test_grad_cam(model, loader, num_samples=3)

# Test 2: Individual detailed explanation
print("\n" + "=" * 60)
print("TEST 2: Detailed Single Sample Explanation")
print("=" * 60)

if len(loader.genuine_images) > 0 and len(loader.forged_images) > 0:
    idx = np.random.randint(0, min(len(loader.genuine_images), len(loader.forged_images)))
    genuine = loader.genuine_images[idx]
    forged = loader.forged_images[idx]
    
    explanation = explain_prediction(model, genuine, forged)
    print(f"Prediction: {explanation['prediction']:.4f}")
    print(f"Decision: {explanation['decision']}")
    print(f"Confidence: {explanation['confidence']:.2f}")
    
    visualize_explanation(explanation, genuine, forged, save_path='results/detailed_explanation.png')
else:
    print("⚠️ Not enough data to test!")

print("\n✅ All tests completed!")
print("📊 Results saved in 'results/' folder")