"""
Grad-CAM Explainability for SigVerify
RQ2: Explainability & Trust
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.data_loader import SignatureDatasetLoader


def get_last_conv_layer(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def generate_grad_cam(model, img1, img2, layer_name=None):
    """
    Generate Grad-CAM heatmap for signature verification decision
    
    Args:
        model: Trained Siamese model
        img1: First signature image (genuine)
        img2: Second signature image (forged/questioned)
        layer_name: Name of convolutional layer to use
    
    Returns:
        heatmap: Grad-CAM heatmap for the decision
    """
    if layer_name is None:
        layer_name = get_last_conv_layer(model)
        if layer_name is None:
            print("No convolutional layer found!")
            return None
    
    # Create model that outputs conv layer and final prediction
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Prepare inputs (ensure correct shape)
    if len(img1.shape) == 3:
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
    elif len(img1.shape) == 4:
        # Already has batch dimension
        pass
    else:
        print(f"Unexpected shape: {img1.shape}")
        return None
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([img1, img2])
        loss = predictions[:, 0]
    
    # Get gradients
    grads = tape.gradient(loss, conv_output)
    
    if grads is None:
        print("No gradients available!")
        return None
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv output
    heatmap = tf.reduce_mean(
        tf.multiply(pooled_grads, conv_output[0]), axis=-1
    )
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
    
    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on original image
    
    Args:
        image: Original signature image (128x128)
        heatmap: Grad-CAM heatmap
        alpha: Transparency of overlay
    
    Returns:
        overlay: RGB image with heatmap overlay
    """
    # Normalize heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert grayscale image to RGB
    if len(image.shape) == 2:
        image_rgb = np.uint8(255 * image)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = np.uint8(255 * image.squeeze())
        if len(image_rgb.shape) == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image_rgb, alpha, heatmap_colored, 1 - alpha, 0)
    
    return overlay


def explain_prediction(model, img1, img2, threshold=0.5):
    """
    Explain a single prediction with Grad-CAM
    
    Args:
        model: Trained Siamese model
        img1: First signature (genuine)
        img2: Second signature (questioned/forged)
        threshold: Decision threshold
    
    Returns:
        dict: Explanation containing prediction, heatmap, and overlay
    """
    # Prepare inputs
    if len(img1.shape) == 2:
        img1_input = img1.reshape(1, 128, 128, 1)
        img2_input = img2.reshape(1, 128, 128, 1)
    else:
        img1_input = img1
        img2_input = img2
    
    # Get prediction
    pred = model.predict([img1_input, img2_input], verbose=0)[0][0]
    
    # Generate heatmap
    heatmap = generate_grad_cam(model, img1_input, img2_input)
    
    if heatmap is not None:
        # Resize heatmap to match image size
        if heatmap.shape != (128, 128):
            heatmap = cv2.resize(heatmap, (128, 128))
        
        # Overlay on original
        overlay = overlay_heatmap(img1, heatmap)
    else:
        heatmap = None
        overlay = None
    
    return {
        'prediction': pred,
        'decision': 'Genuine' if pred > threshold else 'Forged',
        'confidence': abs(pred - 0.5) * 2,  # 0-1 scale
        'heatmap': heatmap,
        'overlay': overlay,
        'threshold': threshold
    }


def visualize_explanation(explanation, img1, img2, save_path=None):
    """
    Visualize the explanation with images and heatmap
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original images
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('Genuine Signature')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('Questioned Signature')
    axes[0, 1].axis('off')
    
    # Decision with confidence
    decision_text = f"Decision: {explanation['decision']}\nConfidence: {explanation['confidence']:.2f}"
    axes[0, 2].text(0.5, 0.5, decision_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    bbox=dict(boxstyle="round", facecolor='lightblue'))
    axes[0, 2].axis('off')
    
    # Row 2: Heatmap and overlay
    if explanation['heatmap'] is not None:
        # Heatmap
        im = axes[1, 0].imshow(explanation['heatmap'], cmap='jet')
        axes[1, 0].set_title('Grad-CAM Heatmap')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[1, 1].imshow(explanation['overlay'])
        axes[1, 1].set_title('Explanation Overlay')
        axes[1, 1].axis('off')
        
        # Explanation text
        axes[1, 2].text(0.1, 0.9,
                        f"🔴 Red: Areas that strongly influenced\nthe decision\n\n"
                        f"🟢 Green: Less influential areas\n\n"
                        f"📊 Prediction: {explanation['prediction']:.4f}\n"
                        f"📈 Decision: {explanation['decision']}",
                        transform=axes[1, 2].transAxes,
                        fontsize=11,
                        verticalalignment='top')
        axes[1, 2].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'No heatmap available', ha='center', va='center')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def test_grad_cam(model, loader, num_samples=3):
    """
    Test Grad-CAM on random samples from the dataset
    """
    print("=" * 60)
    print("🔍 GRAD-CAM EXPLAINABILITY TEST")
    print("=" * 60)
    
    for i in range(num_samples):
        idx = np.random.randint(0, min(len(loader.genuine_images), len(loader.forged_images)))
        
        genuine = loader.genuine_images[idx]
        forged = loader.forged_images[idx]
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Genuine: {genuine.shape}, Forged: {forged.shape}")
        
        # Get explanation
        explanation = explain_prediction(model, genuine, forged)
        
        # Print results
        print(f"Prediction: {explanation['prediction']:.4f}")
        print(f"Decision: {explanation['decision']}")
        print(f"Confidence: {explanation['confidence']:.2f}")
        
        # Visualize
        visualize_explanation(explanation, genuine, forged,
                              save_path=f'results/grad_cam_sample_{i+1}.png')


# Example usage when run directly
if __name__ == "__main__":
    from src.data_loader import SignatureDatasetLoader
    
    # Load data
    loader = SignatureDatasetLoader()
    loader.load_cedar()
    
    # Load model
    from tensorflow.keras.models import load_model
    try:
        model = load_model('models/simple_siamese_model.keras')
        print("✅ Model loaded successfully!")
    except:
        print("⚠️ Model not found. Creating a new one...")
        from src.model import create_simple_siamese
        model = create_simple_siamese()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights('models/simple_siamese_model.weights.h5')
    
    # Test Grad-CAM
    test_grad_cam(model, loader, num_samples=3)