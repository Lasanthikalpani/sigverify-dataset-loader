"""
Siamese CNN Model for Signature Verification
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np


def create_base_network(input_shape=(128, 128, 1)):
    """
    Create the base CNN - shared between two branches
    
    Args:
        input_shape: Shape of input images
    
    Returns:
        Base CNN model
    """
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third convolutional block
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Fourth convolutional block
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    
    return Model(inputs, x, name='base_network')


def create_siamese_cnn(input_shape=(128, 128, 1)):
    """
    Create complete Siamese CNN for signature verification
    
    Args:
        input_shape: Shape of input images
    
    Returns:
        Siamese CNN model
    """
    # Base network (shared weights)
    base_network = create_base_network(input_shape)
    
    # Two input branches
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    
    # Get embeddings
    embedding1 = base_network(input1)
    embedding2 = base_network(input2)
    
    # L1 distance (absolute difference)
    l1_distance = layers.Lambda(
        lambda x: tf.abs(x[0] - x[1])
    )([embedding1, embedding2])
    
    # Dense layers on top
    x = layers.Dense(128, activation='relu')(l1_distance)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer (sigmoid for binary classification)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input1, input2], outputs=output)
    
    return model


def contrastive_loss(margin=1.0):
    """
    Contrastive loss function for Siamese networks
    
    Args:
        margin: Margin for negative pairs
    
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        # y_true: 1 if same person, 0 if different
        # y_pred: Euclidean distance between embeddings
        
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        
        return tf.reduce_mean(
            y_true * square_pred + (1 - y_true) * margin_square
        )
    
    return loss


def get_model_summary(model):
    """Print model summary"""
    model.summary()
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_siamese_cnn()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=contrastive_loss(margin=1.0),
        metrics=['accuracy']
    )
    
    # Print summary
    get_model_summary(model)
    
    print("\n✅ Siamese CNN model created successfully!")
    print(f"Total parameters: {model.count_params():,}")