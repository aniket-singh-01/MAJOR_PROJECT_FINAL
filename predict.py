import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Import custom modules
from model.enhanced_inception_model import create_enhanced_model
from xai.lime_explainer import LimeExplainer

def load_model(model_path, num_classes=9):
    """Load the trained model from saved weights"""
    # Create model architecture
    model = create_enhanced_model(
        input_shape=(299, 299, 3),
        num_classes=num_classes,
        dropout_rate=0.5
    )
    
    # Load weights
    try:
        model.load_weights(model_path)
        print(f"Model weights loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None
    
    return model

def preprocess_image(image_path, target_size=(299, 299)):
    """Preprocess an image for prediction"""
    try:
        # Read and resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        
        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_and_visualize(model, image_path, class_names, explain=False, output_path=None):
    """Make prediction and visualize results"""
    # Preprocess image
    img = preprocess_image(image_path)
    if img is None:
        return
    
    # Make prediction
    prediction = model.predict(img)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(prediction)[::-1][:3]
    top_predictions = [(class_names[i], prediction[i] * 100) for i in top_indices]
    
    # Display original image and predictions
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    display_img = plt.imread(image_path)
    plt.imshow(display_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Prediction results
    plt.subplot(1, 2, 2)
    bars = plt.barh([class_names[i] for i in top_indices], [prediction[i] * 100 for i in top_indices])
    plt.title("Top 3 Predictions")
    plt.xlabel("Probability (%)")
    plt.xlim(0, 100)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f"{prediction[top_indices[i]] * 100:.1f}%", 
                va='center')
    
    plt.tight_layout()
    
    # Save or display result
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_path}")
    else:
        plt.show()
    
    # Print results
    print("\nPrediction Results:")
    print("-------------------")
    for class_name, prob in top_predictions:
        print(f"{class_name}: {prob:.2f}%")
    
    # Generate LIME explanation if requested
    if explain:
        print("\nGenerating LIME explanation...")
        lime_explainer = LimeExplainer(model)
        explanation_path = os.path.splitext(output_path)[0] + "_explanation.png" if output_path else None
        lime_explainer.explain_and_visualize(img[0], save_path=explanation_path, class_idx=top_indices[0])
        print(f"Explanation saved to {explanation_path}")
    
    return top_predictions

def main():
    parser = argparse.ArgumentParser(description='Predict skin condition from an image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image')
    parser.add_argument('--model', type=str, default='checkpoints/continued_20250520-124239/model_19_0.7164.weights.h5', help='Path to the model weights')
    parser.add_argument('--explain', action='store_true', help='Generate LIME explanation')
    parser.add_argument('--output', type=str, help='Path to save the results')
    parser.add_argument('--csv', type=str, default='ISIC_2019/ISIC_2019_Training_GroundTruth.csv', help='Path to CSV with class names')
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Get class names from CSV
    try:
        df = pd.read_csv(args.csv)
        class_names = df.columns[1:].tolist()
        num_classes = len(class_names)
        print(f"Found {num_classes} classes: {class_names}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        class_names = [f"Class_{i}" for i in range(9)]  # Default class names
        num_classes = 9
    
    # Load model
    model = load_model(args.model, num_classes=num_classes)
    if model is None:
        return
    
    # Make prediction
    predict_and_visualize(model, args.image, class_names, explain=args.explain, output_path=args.output)

if __name__ == '__main__':
    main()