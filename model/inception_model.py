import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

def create_model(input_shape=(299, 299, 3), num_classes=8, weights='imagenet', dropout_rate=0.5):
    """
    Create a model with InceptionV3 as base and custom top layers
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes to predict
        weights: Pre-trained weights to use
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Load InceptionV3 with pre-trained weights
    base_model = InceptionV3(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer with sigmoid activation for multi-label classification
    predictions = Dense(num_classes, activation='sigmoid')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def weighted_binary_crossentropy(class_weights):
    """
    Create a weighted binary crossentropy loss function
    
    Args:
        class_weights: Dictionary mapping class indices to weights
        
    Returns:
        Weighted loss function
    """
    import tensorflow as tf
    
    def loss(y_true, y_pred):
        # Standard binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Apply weights
        weight_vector = tf.zeros_like(y_true[:, 0])
        for i, weight in class_weights.items():
            if not np.isinf(weight):  # Skip infinite weights
                weight_vector = weight_vector + (y_true[:, i] * weight)
            
        # Use 1.0 as default weight if no valid class is present
        weight_vector = tf.where(weight_vector == 0, tf.ones_like(weight_vector), weight_vector)
        
        # Apply weights to the loss
        weighted_bce = bce * weight_vector
        
        return tf.reduce_mean(weighted_bce)
    
    return loss