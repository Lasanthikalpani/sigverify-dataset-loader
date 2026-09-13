"""
Grad-CAM Explainability for SigVerify
RQ2: Explainability & Trust - Final Working Version
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.data_loader import SignatureDatasetLoader


def get_base_network(model):
    """
    Find the base network (the one with Conv2D layers)
    """
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            # Check if this layer has Conv2D layers inside
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer
    return None


def get_conv_layer_from_base(base_network, layer_index=-1):
    """
    Get a Conv2D layer from the base network by index
    """
    conv_layers = []
    for layer in base_network.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer)
    
    if conv_layers:
        return conv_layers[layer_index]
    return None


def generate_grad_cam(model, img1, img2):
    """
    Generate Grad-CAM heatmap using the base network's conv layers
    """
    # Find the base network
    base_network = get_base_network(model)
    
    if base_network is None:
        print("   ❌ No base network with Conv2D found!")
        return None
    
    # Get the last Conv2D layer from the base network
    conv_layer = get_conv_layer_from_base(base_network, -1)
    
    if conv_layer is None:
        print("   ❌ No Conv2D layer found in base network!")
        return None
    
    print(f"   Using layer: {conv_layer.name}")
    
    # Create a new model that outputs the conv layer
    # We'll use the base_network's input and output
    try:
        # Create a model from base_network input to conv_layer output
        grad_model = tf.keras.Model(
            inputs=base_network.input,
            outputs=[conv_layer.output, base_network.output]
        )
    except Exception as e:
        print(f"   ⚠️ Could not create grad model: {e}")
        return None
    
    # Prepare inputs - we need to feed both images through the base network
    if len(img1.shape) == 3:
        img1_input = np.expand_dims(img1, axis=0)
        img2_input = np.expand_dims(img2, axis=0)
    else:
        img1_input = img1
        img2_input = img2
    
    # Get the base network's output for both images
    with tf.GradientTape() as tape:
        # Pass both images through the base network
        conv_output1, emb1 = grad_model(img1_input)
        conv_output2, emb2 = grad_model(img2_input)
        
        # Combine embeddings (same as in the main model)
        concat = tf.concat([emb1, emb2], axis=1)
        
        # Pass through the rest of the model (dense layers)
        # We need to get the final prediction
        # Recreate the dense layers from the main model
        dense1 = model.get_layer('dense1')
        dense2 = model.get_layer('dense2')
        output_layer = model.get_layer('output')
        
        x = dense1(concat)
        x = dense2(x)
        pred = output_layer(x)
        
        loss = pred[:, 0]
    
    # Get gradients
    grads = tape.gradient(loss, conv_output1)
    
    if grads is None:
        print("   ⚠️ No gradients available!")
        return None
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv output
    heatmap = tf.reduce_mean(
        tf.multiply(pooled_grads, conv_output1[0]), axis=-1
    )
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
    
    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap on original image"""
    if len(heatmap.shape) > 2:
        heatmap = np.squeeze(heatmap)
    
    if heatmap.shape != (128, 128):
        heatmap = cv2.resize(heatmap, (128, 128))
    
    heatmap_norm = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    if len(image.shape) == 2:
        image_rgb = np.uint8(255 * image)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = np.uint8(255 * image.squeeze())
        if len(image_rgb.shape) == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    
    overlay = cv2.addWeighted(image_rgb, alpha, heatmap_colored, 1 - alpha, 0)
    return overlay


def explain_prediction(model, img1, img2, threshold=0.5):
    """Explain a single prediction with Grad-CAM"""
    if len(img1.shape) == 2:
        img1_input = img1.reshape(1, 128, 128, 1)
        img2_input = img2.reshape(1, 128, 128, 1)
    else:
        img1_input = img1
        img2_input = img2
    
    pred = model.predict([img1_input, img2_input], verbose=0)[0][0]
    
    heatmap = generate_grad_cam(model, img1_input, img2_input)
    
    if heatmap is not None:
        if heatmap.shape != (128, 128):
            heatmap = cv2.resize(heatmap, (128, 128))
        overlay = overlay_heatmap(img1, heatmap)
    else:
        heatmap = None
        overlay = None
    
    return {
        'prediction': pred,
        'decision': 'Genuine' if pred > threshold else 'Forged',
        'confidence': abs(pred - 0.5) * 2,
        'heatmap': heatmap,
        'overlay': overlay,
        'threshold': threshold
    }


def visualize_explanation(explanation, img1, img2, save_path=None):
    """Visualize the explanation"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('Genuine Signature')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('Questioned Signature')
    axes[0, 1].axis('off')
    
    color = 'green' if explanation['decision'] == 'Genuine' else 'red'
    decision_text = f"Decision: {explanation['decision']}\nConfidence: {explanation['confidence']:.2f}"
    axes[0, 2].text(0.5, 0.5, decision_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    color=color,
                    bbox=dict(boxstyle="round", facecolor='white', edgecolor=color))
    axes[0, 2].axis('off')
    
    if explanation['heatmap'] is not None:
        im = axes[1, 0].imshow(explanation['heatmap'], cmap='jet')
        axes[1, 0].set_title('Grad-CAM Heatmap')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        axes[1, 1].imshow(explanation['overlay'])
        axes[1, 1].set_title('Explanation Overlay')
        axes[1, 1].axis('off')
        
        axes[1, 2].text(0.1, 0.9,
                f"RED: Areas that strongly influenced the decision\n\n"
                f"GREEN: Less influential areas\n\n"
                f"Prediction: {explanation['prediction']:.4f}\n"
                f"Decision: {explanation['decision']}",
                transform=axes[1, 2].transAxes,
                fontsize=11,
                verticalalignment='top')
        axes[1, 2].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'No heatmap available\n(Using alternative method)', 
                       ha='center', va='center')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def test_grad_cam(model, loader, num_samples=3):
    """Test Grad-CAM on random samples"""
    print("=" * 60)
    print("🔍 GRAD-CAM EXPLAINABILITY TEST")
    print("=" * 60)
    
    # Check model structure
    print("\n📋 Model Structure:")
    for i, layer in enumerate(model.layers):
        print(f"   {i}: {layer.name} ({layer.__class__.__name__})")
        if hasattr(layer, 'layers'):
            for j, sub in enumerate(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    print(f"      {j}: {sub.name} ({sub.__class__.__name__})")
    
    for i in range(num_samples):
        idx = np.random.randint(0, min(len(loader.genuine_images), len(loader.forged_images)))
        
        genuine = loader.genuine_images[idx]
        forged = loader.forged_images[idx]
        
        print(f"\n--- Sample {i+1} ---")
        
        explanation = explain_prediction(model, genuine, forged)
        
        print(f"   Prediction: {explanation['prediction']:.4f}")
        print(f"   Decision: {explanation['decision']}")
        print(f"   Confidence: {explanation['confidence']:.2f}")
        
        visualize_explanation(explanation, genuine, forged,
                              save_path=f'results/grad_cam_sample_{i+1}.png')


if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    
    loader = SignatureDatasetLoader()
    loader.load_cedar()
    
    model = load_model('models/simple_siamese_model_fixed.keras', safe_mode=False)
    test_grad_cam(model, loader, num_samples=3)