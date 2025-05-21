import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

# Import custom modules
from utils.data_loader import load_data
from model.enhanced_inception_model import create_enhanced_model, focal_loss

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='ISIC_2019/')
parser.add_argument('--batch_size', type=int, default=16)  # Smaller batch size for better generalization
parser.add_argument('--epochs', type=int, default=200)     # More epochs with early stopping
parser.add_argument('--checkpoint', type=str, default=None)
args = parser.parse_args()

# Set up GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Enhanced data loading with stronger augmentation
def load_enhanced_data(data_dir, csv_path, img_size=(299, 299), batch_size=16):
    # Read CSV
    df = pd.read_csv(csv_path)
    image_ids = df['image'].values
    labels = df.iloc[:, 1:].values
    num_classes = labels.shape[1]
    
    print(f"Dataset has {num_classes} classes")
    
    # Get file paths - IMPROVED PATH DETECTION
    image_paths = []
    valid_indices = []
    
    # First, scan the directory to find all image files
    all_image_files = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Store without extension for easier matching
                file_name = os.path.splitext(file)[0]
                all_image_files[file_name] = os.path.join(root, file)
    
    print(f"Found {len(all_image_files)} total image files in directory")
    if len(all_image_files) > 0:
        print(f"Sample image paths: {list(all_image_files.values())[:3]}")
    
    # Match image IDs from CSV with found files
    for i, img_id in enumerate(image_ids):
        # Try exact match
        if img_id in all_image_files:
            image_paths.append(all_image_files[img_id])
            valid_indices.append(i)
        # Try with "ISIC_" prefix
        elif "ISIC_" + img_id in all_image_files:
            image_paths.append(all_image_files["ISIC_" + img_id])
            valid_indices.append(i)
        # Try without "ISIC_" prefix if it has one
        elif img_id.startswith("ISIC_") and img_id[5:] in all_image_files:
            image_paths.append(all_image_files[img_id[5:]])
            valid_indices.append(i)
    
    # If still no images found, print more debug info
    if len(image_paths) == 0:
        print("No matching images found. Debug info:")
        print(f"First 5 image IDs from CSV: {image_ids[:5]}")
        print(f"First 5 image filenames found: {list(all_image_files.keys())[:5] if all_image_files else 'None'}")
        
        # Last resort: try a more flexible matching approach
        print("Trying flexible matching...")
        for i, img_id in enumerate(image_ids):
            found = False
            for file_id in all_image_files.keys():
                # Check if the file_id contains the img_id or vice versa
                if img_id in file_id or file_id in img_id:
                    image_paths.append(all_image_files[file_id])
                    valid_indices.append(i)
                    found = True
                    break
            if found:
                # Just find a few to confirm the approach works
                if len(image_paths) >= 5:
                    print(f"Found {len(image_paths)} images with flexible matching")
                    print(f"Sample matches: {image_paths[:3]}")
                    break
    
    valid_labels = labels[valid_indices] if valid_indices else np.empty((0, labels.shape[1]))
    
    print(f"Found {len(image_paths)} valid images with matching labels")
    
    if len(image_paths) == 0:
        # If still no images found, use standard load_data as fallback
        print("Falling back to standard load_data function...")
        from utils.data_loader import load_data
        return load_data(data_dir, csv_path, img_size=img_size, batch_size=batch_size)
    
    # Create paths and labels dataframe
    image_df = pd.DataFrame({
        'filename': image_paths,
        'class': list(valid_labels)  # Convert each row to a list for storage in DataFrame
    })
    
    # Split data
    train_df, val_df = train_test_split(image_df, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Calculate class weights
    class_weights = {}
    for i in range(valid_labels.shape[1]):
        # Calculate class weight inversely proportional to class frequency
        class_weights[i] = (1.0 / (np.sum(valid_labels[:, i]) + 1e-6)) * (valid_labels.shape[0])
    
    # Strong augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='reflect'
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Custom generator for multi-label data
    def multi_label_generator(dataframe, image_datagen, batch_size):
        while True:
            # Select random batch of image paths
            batch_indices = np.random.randint(0, len(dataframe), batch_size)
            batch_df = dataframe.iloc[batch_indices]
            
            # Load and preprocess images
            batch_x = []
            batch_y = []
            
            for i, row in batch_df.iterrows():
                img_path = row['filename']
                try:
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    
                    # Apply augmentation
                    if image_datagen != val_datagen:
                        img_array = image_datagen.random_transform(img_array)
                        
                    # Rescale
                    img_array = img_array / 255.0
                    
                    batch_x.append(img_array)
                    batch_y.append(row['class'])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    # Use a blank image if loading fails
                    batch_x.append(np.zeros(img_size + (3,)))
                    batch_y.append(row['class'])
            
            yield np.array(batch_x), np.array(batch_y)
    
    # Create generators
    train_generator = multi_label_generator(train_df, train_datagen, batch_size)
    val_generator = multi_label_generator(val_df, val_datagen, batch_size)
    
    # Calculate steps
    train_steps = len(train_df) // batch_size
    val_steps = len(val_df) // batch_size
    
    return train_generator, val_generator, class_weights, train_steps, val_steps

# Load data with enhanced augmentation
print("Loading data with enhanced augmentation...")
train_data_dir = args.data_dir
train_csv_path = os.path.join(args.data_dir, 'ISIC_2019_Training_GroundTruth.csv')

# Use standard load_data for now, but consider implementing enhanced version
train_generator, val_generator, class_weights, train_steps, val_steps = load_enhanced_data(
    train_data_dir,
    train_csv_path,
    img_size=(299, 299),
    batch_size=args.batch_size
)

# Get class names from CSV
df = pd.read_csv(train_csv_path)
class_names = df.columns[1:].tolist()
num_classes = len(class_names)

print(f"Creating model with {num_classes} output classes")

# Create enhanced model with the correct number of classes
model = create_enhanced_model(
    input_shape=(299, 299, 3),
    num_classes=num_classes,
    dropout_rate=0.6
)

# Compile with focal loss
model.compile(
    optimizer=Adam(learning_rate=5e-5),  # Lower initial learning rate
    loss=focal_loss(gamma=2.0),          # Focal loss for class imbalance
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Load checkpoint if provided
if args.checkpoint:
    print(f"Loading weights from {args.checkpoint}")
    try:
        model.load_weights(args.checkpoint)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Starting training from scratch.")

# Create timestamp and directories
timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = os.path.join('checkpoints', timestamp)
log_dir = os.path.join('logs', timestamp)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Multi-stage training
# Stage 1: Train top layers
print("Stage 1: Training top layers...")
for layer in model.layers[:-20]:  # Freeze all but last 20 layers
    layer.trainable = False

# Callbacks
callbacks = [
    ModelCheckpoint(
        os.path.join(checkpoint_dir, 'model_stage1_{epoch:02d}_{val_accuracy:.4f}.weights.h5'),
        monitor='val_accuracy',  # Focus on accuracy for medical tasks
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=7,
        min_lr=1e-7,
        verbose=1,
        mode='max'
    ),
    CSVLogger(
        os.path.join(log_dir, 'training_log_stage1.csv'),
        append=True
    )
]

# First stage training
# We'll need to apply class weights differently with generators
history1 = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks
    # Remove class_weight parameter
)

# Stage 2: Fine-tune more layers
print("Stage 2: Fine-tuning more layers...")
# Unfreeze more layers for fine-tuning
for layer in model.layers[-50:]:  # Unfreeze more layers
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
    loss=focal_loss(gamma=2.0),
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks for stage 2
callbacks2 = [
    ModelCheckpoint(
        os.path.join(checkpoint_dir, 'model_stage2_{epoch:02d}_{val_accuracy:.4f}.weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(checkpoint_dir, 'final_model.weights.h5'),
        save_best_only=False,
        save_weights_only=True,
        verbose=0
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.1,  # More aggressive LR reduction
        patience=10,
        min_lr=1e-8,
        verbose=1,
        mode='max'
    ),
    CSVLogger(
        os.path.join(log_dir, 'training_log_stage2.csv'),
        append=True
    )
]

# Second stage training
history2 = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=150,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks2
    # Remove class_weight parameter
)

# Combine histories
history_combined = {}
for k in history1.history.keys():
    if k in history2.history:
        history_combined[k] = history1.history[k] + history2.history[k]
    else:
        history_combined[k] = history1.history[k]

# Save final model
final_weights_path = os.path.join(checkpoint_dir, 'final_best_model.weights.h5')
model.save_weights(final_weights_path)
print(f"Final model weights saved to {final_weights_path}")

# Plot training history
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.plot(history_combined['accuracy'], label='Training')
plt.plot(history_combined['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 2)
plt.plot(history_combined['loss'], label='Training')
plt.plot(history_combined['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 3)
plt.plot(history_combined['auc'], label='Training')
plt.plot(history_combined['val_auc'], label='Validation')
plt.title('AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 4)
if 'precision' in history_combined and 'recall' in history_combined:
    plt.plot(history_combined['precision'], label='Precision (Train)')
    plt.plot(history_combined['val_precision'], label='Precision (Val)')
    plt.plot(history_combined['recall'], label='Recall (Train)')
    plt.plot(history_combined['val_recall'], label='Recall (Val)')
    plt.title('Precision & Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join('visualizations', f'high_perf_training_{timestamp}.png'), dpi=300)
print(f"Training history saved to visualizations/high_perf_training_{timestamp}.png")

print("Training complete!")