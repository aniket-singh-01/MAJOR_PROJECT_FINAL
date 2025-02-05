import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(data_dir='data/dataset', img_size=(299, 299), test_size=0.2):
    # Create ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=test_size,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Load training data
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, val_generator 