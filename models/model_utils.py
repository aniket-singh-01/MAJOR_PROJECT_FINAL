from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_callbacks():
    checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7
    )
    
    return [checkpoint, early_stop, reduce_lr]

def evaluate_model(model, test_generator):
    # Get true labels and predictions
    y_true = test_generator.classes
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, test_generator.class_indices)
    
    # Additional metrics
    accuracy = np.mean(y_true == y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'true_labels': y_true
    }

def plot_confusion_matrix(y_true, y_pred, class_indices):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_indices.keys(),
                yticklabels=class_indices.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show() 