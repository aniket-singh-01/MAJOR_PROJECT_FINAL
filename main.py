import yaml
import tensorflow as tf
from models.inception_v3 import build_model
from models.gwo_optimizer import GreyWolfOptimizer
from models.model_utils import get_callbacks, evaluate_model, plot_confusion_matrix
from explainability.lime_explainer import LimeExplainer
from data.data_loader import load_data
from tensorflow.keras.mixed_precision import set_global_policy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def main():
    # Load configuration
    with open('config/params.yaml') as f:
        config = yaml.safe_load(f)
    
    # Load and preprocess data
    train_generator, val_generator = load_data(
        config['data']['path'],
        tuple(config['training']['image_size']),
        config['data']['test_size']
    )
    
    # Initialize model
    model = build_model(
        input_shape=(224, 224, 3),
        num_classes=len(train_generator.class_indices)
    )
    
    # Define objective function for GWO
    def objective_function(params):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params[0]),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        history = model.fit(
            train_generator,
            epochs=1,
            validation_data=val_generator,
            verbose=0
        )
        return history.history['val_loss'][0]
    
    # Grey Wolf Optimization
    gwo = GreyWolfOptimizer(
        num_wolves=config['gwo']['num_wolves'],
        max_iter=config['gwo']['max_iter'],
        search_space=config['gwo']['search_space']
    )
    best_params = gwo.optimize(objective_function)
    
    # Initialize optimizer with proper graph context
    with tf.init_scope():
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=best_params[0],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam'
        )
    
    # Compile model with optimizer
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add this before model.fit
    print("Starting training...")
    
    # Add custom callback
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"Starting epoch {epoch + 1}")
    
    # Convert generators to TensorFlow Dataset
    def generator_wrapper(generator):
        for x, y in generator:
            yield x, y
    
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generator_wrapper(train_generator),
        output_signature=(
            tf.TensorSpec(shape=(None, *config['training']['image_size'], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(train_generator.class_indices)), dtype=tf.float32)
        )
    )
    
    # Take only 100 samples for testing
    train_dataset = train_dataset.take(100)
    
    # Apply prefetch and other optimizations
    train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
    
    # Update model.fit to use the dataset
    history = model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        validation_data=val_generator,
        batch_size=int(best_params[1]),
        callbacks=get_callbacks() + [ProgressCallback()],
        verbose=2
    )
    
    # Evaluate model with comprehensive metrics
    evaluation_results = evaluate_model(model, val_generator)
    
    # Additional visualization
    plot_class_distribution(val_generator)
    plot_roc_curve(model, val_generator)
    
    # Explain predictions
    explainer = LimeExplainer(model)
    for images, _ in val_generator:
        for image in images:
            explanation = explainer.explain(image)
            # Save or display explanation
            break
        break

    # Hide GPU devices
    tf.config.set_visible_devices([], 'GPU')
    # Hide CPU devices
    tf.config.set_visible_devices([], 'CPU')

    # Use mixed precision
    set_global_policy('mixed_float16')

    # Install and run Activity Monitor
    # Check GPU and memory usage

    tf.keras.backend.clear_session()

    # Testing Script

    # Load the trained model
    model = tf.keras.models.load_model('models/best_model.h5')  # Path to your saved model

    # Function to preprocess an image for prediction
    def preprocess_image(img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize to [0, 1]
        return img_array

    # Function to make predictions
    def predict_image(img_path, class_names):
        # Preprocess the image
        img_array = preprocess_image(img_path)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        # Display results
        plt.imshow(image.load_img(img_path))
        plt.title(f"Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.2f}")
        plt.axis('off')
        plt.show()
        
        return class_names[predicted_class], confidence

    # Example usage
    class_names = list(train_generator.class_indices.keys())  # Get class names from training data
    img_path = '/kaggle/input/isic-dataset/test_image.jpg'  # Path to your test image
    predicted_class, confidence = predict_image(img_path, class_names)
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main() 