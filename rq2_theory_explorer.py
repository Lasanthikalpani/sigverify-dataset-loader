"""
RQ2 Theory Explorer - Grad-CAM Complete Mathematical Implementation
SigVerify: Explainability & Trust
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from datetime import datetime
from scipy.ndimage import zoom

os.makedirs('results/rq2_theory', exist_ok=True)

print("=" * 80)
print("🔥 RQ2 THEORY EXPLORER - Grad-CAM Complete Implementation")
print("=" * 80)


def create_simple_cnn():
    """Create a simple CNN for demonstration."""
    inputs = Input(shape=(32, 32, 1), name='input')
    x = layers.Conv2D(3, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Conv2D(5, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(4, activation='relu', name='dense1')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=output)
    return model, model.get_layer('conv3')


def create_sample_input():
    """Create a simple 32x32 pattern."""
    x = np.linspace(-2, 2, 32)
    y = np.linspace(-2, 2, 32)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    pattern = np.sin(3 * theta + r * 2) * np.exp(-r * 0.5)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    pattern = pattern + 0.02 * np.random.randn(32, 32)
    pattern = np.clip(pattern, 0, 1)
    return pattern.reshape(1, 32, 32, 1), pattern


def compute_gradcam_step_by_step(model, conv_layer, input_image):
    """Compute Grad-CAM step by step."""
    
    print("\n" + "─" * 60)
    print("STEP 1: FORWARD PASS")
    print("─" * 60)
    
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )
    
    if isinstance(input_image, np.ndarray):
        input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
    else:
        input_tensor = input_image
    
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(input_tensor)
        loss = prediction[0][0]
        
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Conv shape: {conv_output.shape}")
        print(f"   Prediction: {loss:.4f}")
    
    print("\n" + "─" * 60)
    print("STEP 2: COMPUTE GRADIENT ∂y/∂A")
    print("─" * 60)
    
    grads = tape.gradient(loss, conv_output)
    grads_np = grads.numpy()[0]
    
    print(f"   Gradient shape: {grads_np.shape}")
    print(f"   Gradient min: {grads_np.min():.4f}")
    print(f"   Gradient max: {grads_np.max():.4f}")
    print(f"   Gradient mean: {grads_np.mean():.4f}")
    
    print("\n" + "─" * 60)
    print("STEP 3: GLOBAL AVERAGE POOLING (αₖ)")
    print("─" * 60)
    
    alpha = np.mean(grads_np, axis=(0, 1))
    
    print(f"   Alpha shape: {alpha.shape}")
    print(f"   Alpha min: {alpha.min():.4f}")
    print(f"   Alpha max: {alpha.max():.4f}")
    print(f"   Alpha mean: {alpha.mean():.4f}")
    
    sorted_idx = np.argsort(np.abs(alpha))[::-1]
    print(f"\n   Top 3 channels:")
    for idx in sorted_idx[:3]:
        print(f"      Channel {idx}: α = {alpha[idx]:.4f}")
    
    print("\n" + "─" * 60)
    print("STEP 4: WEIGHTED COMBINATION (HEATMAP)")
    print("─" * 60)
    
    conv_output_np = conv_output.numpy()[0]
    
    heatmap = np.zeros((conv_output_np.shape[0], conv_output_np.shape[1]))
    for k in range(conv_output_np.shape[2]):
        heatmap += alpha[k] * conv_output_np[:, :, k]
    
    heatmap = np.maximum(heatmap, 0)
    
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Heatmap min: {heatmap.min():.4f}")
    print(f"   Heatmap max: {heatmap.max():.4f}")
    print(f"   Heatmap mean: {heatmap.mean():.4f}")
    
    print("\n" + "─" * 60)
    print("STEP 5: NORMALIZE")
    print("─" * 60)
    
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
    
    print(f"   Normalized min: {heatmap_norm.min():.4f}")
    print(f"   Normalized max: {heatmap_norm.max():.4f}")
    
    print("\n" + "─" * 60)
    print("STEP 6: RESIZE AND OVERLAY")
    print("─" * 60)
    
    if isinstance(input_image, np.ndarray):
        original_image = input_image[0, :, :, 0]
    else:
        original_image = input_image.numpy()[0, :, :, 0]
    
    zoom_factor = (original_image.shape[0] / heatmap_norm.shape[0],
                   original_image.shape[1] / heatmap_norm.shape[1])
    heatmap_resized = zoom(heatmap_norm, zoom_factor, order=1)
    
    overlay = np.zeros((original_image.shape[0], original_image.shape[1], 3))
    for i in range(3):
        overlay[:, :, i] = original_image
    
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    overlay = 0.6 * overlay + 0.4 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    print(f"   Original shape: {original_image.shape}")
    print(f"   Resized heatmap shape: {heatmap_resized.shape}")
    print(f"   Overlay shape: {overlay.shape}")
    
    return {
        'conv_output': conv_output_np,
        'gradients': grads_np,
        'alpha': alpha,
        'heatmap': heatmap,
        'heatmap_norm': heatmap_norm,
        'heatmap_resized': heatmap_resized,
        'overlay': overlay,
        'original_image': original_image,
        'prediction': float(prediction[0][0])
    }


def visualize_results(results):
    """Visualize Grad-CAM results."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1
    axes[0, 0].imshow(results['original_image'], cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    fmaps = results['conv_output']
    fmap_display = np.zeros((fmaps.shape[0], fmaps.shape[1] * 3))
    for i in range(min(3, fmaps.shape[2])):
        fmap_display[:, i*fmaps.shape[1]:(i+1)*fmaps.shape[1]] = fmaps[:, :, i]
    axes[0, 1].imshow(fmap_display, cmap='viridis')
    axes[0, 1].set_title(f'Feature Maps (3 of {fmaps.shape[2]})', fontsize=12)
    axes[0, 1].axis('off')
    
    grads = results['gradients']
    grad_display = np.zeros((grads.shape[0], grads.shape[1] * 3))
    for i in range(min(3, grads.shape[2])):
        grad_display[:, i*grads.shape[1]:(i+1)*grads.shape[1]] = grads[:, :, i]
    axes[0, 2].imshow(grad_display, cmap='RdBu')
    axes[0, 2].set_title(f'Gradients (3 of {grads.shape[2]})', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[0, 3].bar(range(len(results['alpha'])), results['alpha'])
    axes[0, 3].set_title('Alpha (αₖ) - Filter Weights', fontsize=12)
    axes[0, 3].set_xlabel('Filter Index (k)')
    axes[0, 3].set_ylabel('Weight')
    axes[0, 3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Row 2
    im1 = axes[1, 0].imshow(results['heatmap'], cmap='jet')
    axes[1, 0].set_title('Heatmap (Raw)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(results['heatmap_norm'], cmap='jet')
    axes[1, 1].set_title('Heatmap (Normalized)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 2].imshow(results['heatmap_resized'], cmap='jet')
    axes[1, 2].set_title('Heatmap (Resized)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    axes[1, 3].imshow(results['overlay'])
    axes[1, 3].set_title(f"Overlay\nPrediction: {results['prediction']:.4f}", fontsize=12)
    axes[1, 3].axis('off')
    
    if results['prediction'] < 0.5:
        decision, color = "FORGED", 'red'
    else:
        decision, color = "GENUINE", 'green'
    axes[1, 3].text(0.5, 0.95, f"Decision: {decision}",
                    transform=axes[1, 3].transAxes, fontsize=14, fontweight='bold',
                    color=color, horizontalalignment='center')
    
    plt.suptitle('Grad-CAM: Complete Mathematical Visualization', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/rq2_theory/gradcam_complete_math.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n📁 Saved: results/rq2_theory/gradcam_complete_math.png")


def main():
    print("\n🚀 RUNNING RQ2 THEORY EXPLORER\n")
    
    model, conv_layer = create_simple_cnn()
    print(f"✅ Model created! Conv layer: {conv_layer.name}")
    
    input_tensor, _ = create_sample_input()
    print(f"✅ Input shape: {input_tensor.shape}")
    
    results = compute_gradcam_step_by_step(model, conv_layer, input_tensor)
    visualize_results(results)
    
    print("\n" + "=" * 80)
    print("🎉 RQ2 THEORY EXPLORER COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()