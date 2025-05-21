import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf

class LimeExplainer:
    """
    Class to explain model predictions using LIME
    """
    def __init__(self, model):
        """
        Initialize LIME explainer
        
        Args:
            model: Trained Keras model
        """
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()
        
    def preprocess_image(self, img):
        """
        Preprocess image for model prediction
        
        Args:
            img: Input image
            
        Returns:
            Preprocessed image
        """
        # Resize if needed
        if img.shape[:2] != (299, 299):
            img = tf.image.resize(img, (299, 299)).numpy()
        
        # Ensure RGB format
        if img.shape[-1] != 3:
            if len(img.shape) == 2:  # Grayscale
                img = np.stack([img, img, img], axis=-1)
        
        # Ensure values are in [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
            
        return img
    
    def predict_fn(self, images):
        """
        Model prediction function for LIME
        
        Args:
            images: Batch of images
            
        Returns:
            Batch of predictions
        """
        return self.model.predict(images)
    
    def explain_image(self, img, num_features=100000, num_samples=1000, class_idx=None):
        """
        Explain model prediction for an image
        
        Args:
            img: Input image
            num_features: Number of features for LIME
            num_samples: Number of samples for LIME
            class_idx: Class index to explain (None for all)
            
        Returns:
            LIME explanation
        """
        # Preprocess image
        processed_img = self.preprocess_image(img)
        
        # Get model prediction
        pred = self.model.predict(np.expand_dims(processed_img, axis=0))[0]
        
        # If class_idx not provided, use the highest probability class
        if class_idx is None:
            class_idx = np.argmax(pred)
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            processed_img, 
            self.predict_fn,
            labels=([class_idx] if class_idx is not None else None),
            top_labels=5 if class_idx is None else None,
            hide_color=0,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return explanation, class_idx, pred
    
    def visualize_explanation(self, img, explanation, class_idx, pred, save_path=None):
        """
        Visualize LIME explanation
        
        Args:
            img: Original image
            explanation: LIME explanation
            class_idx: Class index being explained
            pred: Model prediction
            save_path: Path to save visualization
            
        Returns:
            None (displays or saves visualization)
        """
        # Get explanation image
        processed_img = self.preprocess_image(img)
        temp, mask = explanation.get_image_and_mask(
            class_idx,
            positive_only=False,
            num_features=5,
            hide_rest=False
        )
        
        # Create figure with original image and explanation
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image with prediction
        axes[0].imshow(processed_img)
        axes[0].set_title(f"Prediction: Class {class_idx}\nProbability: {pred[class_idx]:.4f}")
        axes[0].axis('off')
        
        # Explanation
        axes[1].imshow(mark_boundaries(temp, mask))
        axes[1].set_title(f"LIME Explanation for Class {class_idx}")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def explain_and_visualize(self, img, save_path=None, num_features=100000, num_samples=1000, class_idx=None):
        """
        Explain and visualize in one step
        
        Args:
            img: Input image
            save_path: Path to save visualization
            num_features: Number of features for LIME
            num_samples: Number of samples for LIME
            class_idx: Class index to explain
            
        Returns:
            Explanation, class index, and prediction
        """
        explanation, class_idx, pred = self.explain_image(
            img, num_features, num_samples, class_idx
        )
        
        self.visualize_explanation(img, explanation, class_idx, pred, save_path)
        
        return explanation, class_idx, pred