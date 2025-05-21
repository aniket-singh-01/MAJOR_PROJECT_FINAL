import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, Input, Activation, Flatten
from tensorflow.keras.regularizers import l2

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
    """Residual block with pre-activation."""
    shortcut = x
    
    # Pre-activation
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # First convolution
    x = Conv2D(filters, kernel_size, strides=stride, padding='same',
               use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolution
    x = Conv2D(filters, kernel_size, padding='same',
               use_bias=False, kernel_regularizer=l2(1e-4))(x)
    
    # Shortcut connection
    if conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same', 
                          use_bias=False, kernel_regularizer=l2(1e-4))(shortcut)
    
    # Add shortcut to main path
    x = Add()([x, shortcut])
    
    return x

def attention_module(x, filters):
    """Attention module for focusing on relevant features."""
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = tf.reshape(avg_pool, [-1, 1, 1, filters])
    
    # Channel attention
    channel_att = Conv2D(filters//4, kernel_size=1, use_bias=False)(avg_pool)
    channel_att = Activation('relu')(channel_att)
    channel_att = Conv2D(filters, kernel_size=1, use_bias=False)(channel_att)
    
    # Spatial attention
    spatial_att = Conv2D(1, kernel_size=7, padding='same', use_bias=False)(x)
    
    # Apply attentions
    channel_att = Activation('sigmoid')(channel_att)
    spatial_att = Activation('sigmoid')(spatial_att)
    
    x = x * channel_att * spatial_att
    
    return x

def create_enhanced_model(input_shape=(299, 299, 3), num_classes=9, dropout_rate=0.5):
    """
    Create enhanced InceptionV3-based model with residual blocks, attention modules,
    and deeper architecture for improved skin lesion classification.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled model
    """
    # Load the base model (without top)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Extract intermediate features from InceptionV3
    shallow_features = base_model.get_layer('activation_10').output  # Earlier layer
    mid_features = base_model.get_layer('activation_40').output  # Middle layer
    deep_features = base_model.output  # Final output
    
    # Process deep features
    x = GlobalAveragePooling2D()(deep_features)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Process intermediate features
    mid_x = GlobalAveragePooling2D()(mid_features)
    mid_x = BatchNormalization()(mid_x)
    mid_x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(mid_x)
    mid_x = BatchNormalization()(mid_x)
    mid_x = Dropout(dropout_rate)(x)
    
    # Process shallow features
    shallow_x = GlobalAveragePooling2D()(shallow_features)
    shallow_x = BatchNormalization()(shallow_x)
    shallow_x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(shallow_x)
    shallow_x = BatchNormalization()(shallow_x)
    shallow_x = Dropout(dropout_rate)(shallow_x)
    
    # Combine features from different depths
    combined = Concatenate()([x, mid_x, shallow_x])
    
    # Add residual blocks for refined feature extraction
    combined = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(combined)
    combined = Dropout(dropout_rate)(combined)
    combined = BatchNormalization()(combined)
    
    # Add final classification layer
    predictions = Dense(num_classes, activation='softmax')(combined)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def focal_loss(gamma=2., alpha=.25):
    """
    Focal Loss for better handling of class imbalance
    
    Args:
        gamma: focusing parameter
        alpha: balancing parameter
        
    Returns:
        Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # For multi-class
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Apply the focal term
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        
        # Sum over classes, mean over batch
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    return focal_loss_fixed

def weighted_categorical_crossentropy(class_weights):
    """Weighted categorical crossentropy for imbalanced datasets"""
    class_weights_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        # Scale predictions
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        # Calculate weighted loss
        loss = y_true * tf.math.log(y_pred) * class_weights_tensor
        loss = -tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)
    
    return loss