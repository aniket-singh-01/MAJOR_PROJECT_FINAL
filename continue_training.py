import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt

# Import custom modules
from model.enhanced_inception_model import create_enhanced_model, focal_loss
from utils.data_loader import load_data

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='ISIC_2019/')
parser.add_argument('--batch_size', type=int, default=12)  # Slightly smaller batch for more updates
parser.add_argument('--epochs', type=int, default=100)     # Set more epochs with early stopping
parser.add_argument('--checkpoint', type=str, default='checkpoints/20250517-141717/model_stage2_35_0.7089.weights.h5')
parser.add_argument('--learning_rate', type=float, default=5e-6)  # Start with a lower learning rate
parser.add_argument('--initial_epoch', type=int, default=0, help='Epoch to start counting from')
args = parser.parse_args()

# Set up GPU memory growth
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

# Load data with enhanced augmentation - use your existing load_data with stronger augmentation
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

print(f"Dataset has {num_classes} classes")

# Create enhanced model with the same architecture as before
model = create_enhanced_model(
    input_shape=(299, 299, 3),
    num_classes=num_classes,
    dropout_rate=0.5  # Slightly reduce dropout to allow more learning
)

# Compile the model with a custom learning rate
model.compile(
    optimizer=Adam(learning_rate=args.learning_rate),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Load the best checkpoint
print(f"Loading weights from {args.checkpoint}")
try:
    model.load_weights(args.checkpoint)
    print("Best model weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Starting training from scratch.")
    exit(1)  # Exit if we can't load the weights

# Make all layers trainable for fine-tuning
for layer in model.layers:
    layer.trainable = True

print("All layers are now trainable")
model.summary()

# Create timestamp and directories
timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join('checkpoints', f'continued_{timestamp}')
log_dir = os.path.join('logs', f'continued_{timestamp}')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Enhanced callbacks with longer patience for fine-tuning
callbacks = [
    ModelCheckpoint(
        os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_accuracy:.4f}.weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(checkpoint_dir, 'last_model.weights.h5'),
        save_best_only=False,
        save_weights_only=True,
        verbose=0
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=25,  # Longer patience for fine-tuning
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,  # More aggressive reduction
        patience=15,  # Longer patience for learning rate reduction
        min_lr=1e-8,
        verbose=1,
        mode='max'
    ),
    CSVLogger(
        os.path.join(log_dir, 'training_log.csv'),
        append=True
    ),
    # Add TensorBoard callback
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )
]

# Continue training - this is Stage 3
print(f"Continuing training from best checkpoint...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=args.epochs,
    initial_epoch=args.initial_epoch,  # Add this line
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_weights_path = os.path.join(checkpoint_dir, 'final_model.weights.h5')
model.save_weights(final_weights_path)
print(f"Final model weights saved to {final_weights_path}")

# Plot training history
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 3)
plt.plot(history.history['auc'], label='Training')
plt.plot(history.history['val_auc'], label='Validation')
plt.title('AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 4)
plt.plot(history.history['precision'], label='Precision (Train)')
plt.plot(history.history['val_precision'], label='Precision (Val)')
plt.plot(history.history['recall'], label='Recall (Train)')
plt.plot(history.history['val_recall'], label='Recall (Val)')
plt.title('Precision & Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join('visualizations', f'continued_training_{timestamp}.png'), dpi=300)
print(f"Training history saved to visualizations/continued_training_{timestamp}.png")

print("Training complete!")

# Add evaluation on test set if available
test_csv_path = os.path.join(args.data_dir, 'ISIC_2019_Test_GroundTruth.csv')
if os.path.exists(test_csv_path):
    print("Evaluating on test set...")
    from utils.data_loader import load_test_data
    
    test_generator, test_steps = load_test_data(
        args.data_dir,
        test_csv_path,
        img_size=(299, 299),
        batch_size=args.batch_size
    )
    
    test_results = model.evaluate(test_generator, steps=test_steps)
    print("Test results:", dict(zip(model.metrics_names, test_results)))
else:
    print("No test set found. Skipping evaluation.")