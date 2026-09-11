"""
RQ2: Step-by-Step Signature Comparison
SigVerify - Siamese CNN Layer-by-Layer Analysis
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
from data_loader import SignatureDatasetLoader


def create_base_network():
    """Shared CNN for feature extraction"""
    inputs = Input(shape=(128, 128, 1), name='base_input')
    
    x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
    
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(64, activation='relu', name='embedding')(x)
    
    return Model(inputs, x, name='base_network')


def compare_signatures(model, sig1, sig2):
    """Compare two signatures step by step"""
    print("=" * 70)
    print("SIGVERIFY - STEP-BY-STEP COMPARISON")
    print("=" * 70)
    
    sig1_input = sig1.reshape(1, 128, 128, 1)
    sig2_input = sig2.reshape(1, 128, 128, 1)
    
    base_network = model.get_layer('base_network')
    
    conv1_model = Model(inputs=base_network.input, 
                        outputs=base_network.get_layer('conv1').output)
    conv2_model = Model(inputs=base_network.input, 
                        outputs=base_network.get_layer('conv2').output)
    conv3_model = Model(inputs=base_network.input, 
                        outputs=base_network.get_layer('conv3').output)
    gap_model = Model(inputs=base_network.input, 
                      outputs=base_network.get_layer('gap').output)
    emb_model = Model(inputs=base_network.input, 
                      outputs=base_network.get_layer('embedding').output)
    
    s1_conv1 = conv1_model.predict(sig1_input, verbose=0)
    s2_conv1 = conv1_model.predict(sig2_input, verbose=0)
    
    print("\n1. Conv2D_32:")
    print(f"   Signature 1 mean: {s1_conv1.mean():.4f}")
    print(f"   Signature 2 mean: {s2_conv1.mean():.4f}")
    print(f"   Difference: {abs(s1_conv1.mean() - s2_conv1.mean()):.4f}")
    
    s1_emb = emb_model.predict(sig1_input, verbose=0)
    s2_emb = emb_model.predict(sig2_input, verbose=0)
    
    distance = np.sqrt(np.sum((s1_emb - s2_emb) ** 2))
    print(f"\n2. Euclidean Distance: {distance:.4f}")
    
    pred = model.predict([sig1_input, sig2_input], verbose=0)[0][0]
    print(f"\n3. Final Prediction: {pred:.4f}")
    print(f"   Decision: {'Genuine' if pred > 0.5 else 'Forged'}")
    
    return distance, pred


if __name__ == "__main__":
    print("=" * 70)
    print("📂 Loading dataset...")
    print("=" * 70)
    loader = SignatureDatasetLoader()
    loader.load_cedar()
    
    print("\n🧠 Loading model...")
    model = tf.keras.models.load_model('models/gradcam_model.keras')
    
    sig1 = loader.genuine_images[0]
    sig2 = loader.forged_images[0]
    
    distance, pred = compare_signatures(model, sig1, sig2)
    
    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 70)