# resume_training.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Add parent directory to path to fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import custom modules
from utils.data_loader import load_data
from model.inception_model import create_model, weighted_binary_crossentropy

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='ISIC_2019/')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--checkpoint', type=str, default='checkpoints/20250515-034907/last_model.h5')
args = parser.parse_args()

# Set up GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load data
print("Loading data...")
train_data_dir = args.data_dir
train_csv_path = os.path.join(args.data_dir, 'ISIC_2019_Training_GroundTruth.csv')
train_generator, val_generator, class_weights, train_steps, val_steps = load_data(
    train_data_dir, 
    train_csv_path,
    img_size=(299, 299),
    batch_size=args.batch_size,
    validation_split=0.2
)

# Get class names from CSV
df = pd.read_csv(train_csv_path)
class_names = df.columns[1:].tolist()
num_classes = len(class_names)

# Create new model with the same architecture
model = create_model(
    input_shape=(299, 299, 3),
    num_classes=num_classes,
    dropout_rate=0.5
)

# Compile model with the same settings
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=weighted_binary_crossentropy(class_weights),
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Determine starting epoch from existing logs
checkpoint_dir = os.path.dirname(args.checkpoint)
timestamp = os.path.basename(checkpoint_dir)
log_path = os.path.join('logs', timestamp, 'training_log.csv')

# Determine starting epoch
initial_epoch = 84  # Hard-code to start from epoch 84

# Or keep the existing code but add an override:
if os.path.exists(log_path):
    log_df = pd.read_csv(log_path)
    detected_epoch = log_df.shape[0]
    print(f"Detected epoch from log: {detected_epoch}")
    initial_epoch = max(84, detected_epoch)  # Use 84 or the detected epoch, whichever is higher
else:
    print("Log file not found. Starting from epoch 84.")
    initial_epoch = 84

print(f"Resuming from epoch {initial_epoch}")

# Try to load weights
print(f"Attempting to load weights from {args.checkpoint}")
try:
    model.load_weights(args.checkpoint)
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    
    # Try to load model without compiling
    try:
        temp_model = tf.keras.models.load_model(args.checkpoint, compile=False)
        
        # Copy weights layer by layer
        for i, layer in enumerate(model.layers):
            if i < len(temp_model.layers):
                try:
                    layer.set_weights(temp_model.layers[i].get_weights())
                except:
                    print(f"Could not transfer weights for layer {i}: {layer.name}")
        
        print("Weights transferred layer by layer!")
    except Exception as e2:
        print(f"Failed to load weights: {e2}")
        print("Starting training from scratch.")
        initial_epoch = 0

# Create new timestamp for this continuation run
new_timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
print(f"Creating new checkpoint directory with timestamp: {new_timestamp}")

# Create new checkpoint and log directories
new_checkpoint_dir = os.path.join('checkpoints', new_timestamp)
new_log_dir = os.path.join('logs', new_timestamp)
os.makedirs(new_checkpoint_dir, exist_ok=True)
os.makedirs(new_log_dir, exist_ok=True)

# Setup callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(new_checkpoint_dir, 'model_{epoch:02d}_{val_loss:.4f}.weights.h5'),  # Changed file extension
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(new_checkpoint_dir, 'last_model.weights.h5'),  # Changed file extension
        save_best_only=False,
        save_weights_only=True,
        verbose=0
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=new_log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(new_log_dir, 'training_log.csv'),
        append=True,
        separator=','
    )
]

# Train model
print(f"Continuing training from epoch {initial_epoch}...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=args.epochs,
    initial_epoch=initial_epoch,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks
)

# Save the final model weights
final_weights_path = os.path.join(new_checkpoint_dir, 'final_model_weights.weights.h5')  # Changed file extension
model.save_weights(final_weights_path)
print(f"Final model weights saved to {final_weights_path}")

# Save training history plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('visualizations', f'resumed_training_{new_timestamp}.png'))
print(f"Training history saved to visualizations/resumed_training_{new_timestamp}.png")