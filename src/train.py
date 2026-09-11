"""
Training script for Siamese CNN
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
import matplotlib.pyplot as plt

# Add src to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import SignatureDatasetLoader
from model import create_siamese_cnn, contrastive_loss


class SigVerifyTrainer:
    """
    Trainer class for SigVerify
    """
    
    def __init__(self, data_dir='data', model_dir='models', img_size=(128, 128)):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.img_size = img_size
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        
        # Data loader
        self.loader = SignatureDatasetLoader(data_dir=data_dir, img_size=img_size)
        
        # Model
        self.model = None
        self.history = None
    
    def load_data(self):
        """Load dataset and create pairs"""
        print("\n📂 Loading dataset...")
        
        # Load data
        self.loader.load_cedar()
        
        # Get statistics
        stats = self.loader.get_dataset_stats()
        print(f"   Genuine: {stats['num_genuine']}")
        print(f"   Forged: {stats['num_forged']}")
        
        # Create pairs
        X1, X2, y = self.loader.create_pairs(num_pairs=10000)
        
        print(f"   Created {len(X1)} pairs")
        print(f"   Positive pairs: {np.sum(y)}")
        print(f"   Negative pairs: {len(y) - np.sum(y)}")
        
        return X1, X2, y
    
    def build_model(self):
        """Build and compile the model"""
        print("\n🧠 Building Siamese CNN model...")
        
        self.model = create_siamese_cnn((self.img_size[0], self.img_size[1], 1))
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=contrastive_loss(margin=1.0),
            metrics=['accuracy']
        )
        
        print(f"   Model built! Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def get_callbacks(self):
        """Get training callbacks"""
        return [
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.keras'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    def train(self, X1, X2, y, epochs=50, batch_size=32, test_size=0.2):
        """Train the model"""
        print(f"\n🏋️ Starting training for {epochs} epochs...")
        
        # Split data
        split_idx = int(len(X1) * (1 - test_size))
        
        X1_train = X1[:split_idx]
        X2_train = X2[:split_idx]
        y_train = y[:split_idx]
        
        X1_val = X1[split_idx:]
        X2_val = X2[split_idx:]
        y_val = y[split_idx:]
        
        print(f"   Training pairs: {len(X1_train)}")
        print(f"   Validation pairs: {len(X1_val)}")
        
        # Train
        self.history = self.model.fit(
            [X1_train, X2_train],
            y_train,
            validation_data=([X1_val, X2_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        return self.history
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def save_model(self, filename='final_model.keras'):
        """Save the trained model"""
        self.model.save(os.path.join(self.model_dir, filename))
        print(f"✅ Model saved to {self.model_dir}/{filename}")
    
    def run_pipeline(self, epochs=50, batch_size=32):
        """Run complete training pipeline"""
        # Load data
        X1, X2, y = self.load_data()
        
        # Build model
        self.build_model()
        
        # Train
        self.train(X1, X2, y, epochs=epochs, batch_size=batch_size)
        
        # Plot history
        self.plot_history(save_path='results/figures/training_history.png')
        
        # Save model
        self.save_model()
        
        print("\n🎉 Training pipeline completed successfully!")


# Run training
if __name__ == "__main__":
    # Initialize trainer
    trainer = SigVerifyTrainer(
        data_dir='data',
        model_dir='models'
    )
    
    # Run pipeline
    trainer.run_pipeline(epochs=30, batch_size=32)