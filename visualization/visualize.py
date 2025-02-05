import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical

def visualize_explanation(explanation, image, label, top_labels=5):
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title(f"Explanation for class {label}")
    plt.axis('off')
    plt.show()

def plot_class_distribution(generator):
    class_counts = np.bincount(generator.classes)
    plt.figure(figsize=(10, 6))
    plt.bar(generator.class_indices.keys(), class_counts)
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.show()

def plot_roc_curve(model, generator):
    y_true = to_categorical(generator.classes)
    y_pred = model.predict(generator)
    
    plt.figure(figsize=(10, 8))
    for i in range(y_true.shape[1]):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show() 