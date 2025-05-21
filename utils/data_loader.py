import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

def load_data(data_dir, csv_path, img_size=(299, 299), batch_size=32, validation_split=0.2):
    """
    Load and preprocess the ISIC 2019 dataset
    
    Args:
        data_dir: Directory containing the images
        csv_path: Path to the ground truth CSV file
        img_size: Target image size for InceptionV3
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        train_generator, validation_generator, class_weights, train_steps, val_steps
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
        
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file does not exist: {csv_path}")
    
    # Read ground truth data
    df = pd.read_csv(csv_path)
    
    # Get image IDs and labels
    image_ids = df['image'].values
    labels = df.iloc[:, 1:].values  # All columns except the first one are labels
    
    # Debug directory structure
    print(f"Data directory: {data_dir}")
    print(f"Is data_dir a directory? {os.path.isdir(data_dir)}")
    
    # If ISIC_2019_Training_Input is a file instead of a directory, try to fix the path
    if not os.path.isdir(data_dir) and "ISIC_2019_Training_Input" in data_dir:
        parent_dir = os.path.dirname(data_dir)
        print(f"Trying parent directory: {parent_dir}")
        if os.path.isdir(parent_dir):
            data_dir = parent_dir
            print(f"Using parent directory instead: {data_dir}")
    
    # List all files recursively in the data directory to find images
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_files)} image files in total")
    if all_files:
        print(f"Sample paths: {all_files[:5]}")
    
    # Create a mapping from image ID to file path
    image_to_path = {}
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # Strip extension
        base_name = os.path.splitext(file_name)[0]
        image_to_path[base_name] = file_path
    
    print(f"Found {len(image_to_path)} unique image IDs in files")
    
    # Match image IDs from CSV to actual files
    valid_image_ids = []
    valid_image_labels = []
    
    for i, img_id in enumerate(image_ids):
        # Check if the exact ID exists
        if img_id in image_to_path:
            valid_image_ids.append(img_id)
            valid_image_labels.append(labels[i])
        else:
            # For downsampled images, try the original ID
            if "_downsampled" in img_id:
                original_id = img_id.replace("_downsampled", "")
                if original_id in image_to_path:
                    valid_image_ids.append(original_id)
                    valid_image_labels.append(labels[i])
    
    print(f"Found {len(valid_image_ids)} valid images out of {len(image_ids)} in the CSV")
    
    if len(valid_image_ids) == 0:
        # If still no matches, list some image IDs from CSV and file names for debugging
        print("CSV image IDs (first 10):", image_ids[:10])
        print("File image IDs (first 10):", list(image_to_path.keys())[:10])
        raise ValueError("No valid images found. Please check your dataset path and structure.")
    
    # Convert to numpy arrays
    valid_image_ids = np.array(valid_image_ids)
    valid_image_labels = np.array(valid_image_labels)
    
    # Split data into train and validation sets
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        valid_image_ids, valid_image_labels, test_size=validation_split, 
        random_state=42, stratify=valid_image_labels.argmax(axis=1)
    )
    
    # Calculate class weights to handle imbalanced data
    class_weights = {}
    for i in range(valid_image_labels.shape[1]):
        class_weights[i] = np.sum(valid_image_labels) / (valid_image_labels.shape[0] * np.sum(valid_image_labels[:, i]))
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Define custom generator to handle multiple labels
    def generate_from_dataframe(ids, label_array, datagen, batch_size, shuffle=True):
        n_samples = len(ids)
        indices = np.arange(n_samples)
        
        # Create a mapping from image ID to label
        id_to_label = {ids[i]: label_array[i] for i in range(n_samples)}
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
                
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_ids = [ids[i] for i in batch_indices]
                
                images = []
                labels_list = []
                
                for img_id in batch_ids:
                    img_path = image_to_path.get(img_id)
                    
                    if img_path and os.path.exists(img_path):
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, img_size)
                            else:
                                print(f"WARNING: Could not read image from {img_path}, using blank image")
                                img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
                        except Exception as e:
                            print(f"Error reading {img_path}: {e}")
                            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
                    else:
                        print(f"WARNING: Image path not found for ID {img_id}, using blank image")
                        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
                    
                    # Apply data augmentation if using training generator
                    if datagen != val_datagen:
                        img = datagen.random_transform(img)
                    
                    # Ensure float32 data type and proper scaling
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels_list.append(id_to_label[img_id])
                
                # Convert to numpy arrays
                if images:
                    # Explicitly convert to float32 arrays
                    images_array = np.array(images, dtype=np.float32)
                    labels_array = np.array(labels_list, dtype=np.float32)
                    
                    yield images_array, labels_array
    
    # Create generators
    train_generator = generate_from_dataframe(train_ids, train_labels, train_datagen, batch_size, shuffle=True)
    val_generator = generate_from_dataframe(val_ids, val_labels, val_datagen, batch_size, shuffle=False)
    
    # Calculate steps
    train_steps = len(train_ids) // batch_size
    val_steps = len(val_ids) // batch_size
    
    # Ensure at least one step
    train_steps = max(1, train_steps)
    val_steps = max(1, val_steps)
    
    return train_generator, val_generator, class_weights, train_steps, val_steps