import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Import custom modules
from utils.data_loader import load_data
from model.inception_model import create_model, weighted_binary_crossentropy
from training.trainer import ModelTrainer
from visualization.visualizer import ModelVisualizer
from xai.lime_explainer import LimeExplainer
from optimizer.gwo import tune_hyperparameters
from optimizer.genetic_algorithm import feature_selection, GeneticAlgorithm

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main(args):
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('explanations', exist_ok=True)
    
    # Load data
    print("Loading data...")
    
    # Use direct paths for better control
    train_data_dir = args.data_dir  # or directly use 'C:\\Users\\anike\\OneDrive\\Desktop\\majorrrrrrr\\ISIC_2019'
    train_csv_path = os.path.join(args.data_dir, 'ISIC_2019_Training_GroundTruth.csv')
    
    test_data_dir = args.data_dir
    test_csv_path = os.path.join(args.data_dir, 'ISIC_2019_Test_GroundTruth.csv')
    
    # Load training data
    train_generator, val_generator, class_weights, train_steps, val_steps = load_data(
        train_data_dir, 
        train_csv_path,
        img_size=(299, 299),
        batch_size=args.batch_size,
        validation_split=0.2
    )
    
    # Get class names from CSV
    df = pd.read_csv(train_csv_path)
    class_names = df.columns[1:].tolist()  # All columns except the first one (image ID)
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")
    
    if args.tune_hyperparams:
        # Define hyperparameter space for GWO
        param_space = {
            'dropout_rate': (0.3, 0.7),
            'learning_rate': (1e-5, 1e-3),
        }
        
        # Function to build model with different hyperparameters
        def model_builder(**params):
            model = create_model(
                input_shape=(299, 299, 3),
                num_classes=num_classes,
                dropout_rate=params['dropout_rate']
            )
            
            model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        # Tune hyperparameters
        print("Tuning hyperparameters with Grey Wolf Optimizer...")
        best_params = tune_hyperparameters(
            model_builder,
            train_generator,
            val_generator,
            param_space,
            n_epochs=5
        )
        
        print(f"Best hyperparameters: {best_params}")
        
        # Create model with best hyperparameters
        model = model_builder(**best_params)
    else:
        # Create model with default hyperparameters
        model = create_model(
            input_shape=(299, 299, 3),
            num_classes=num_classes,
            dropout_rate=0.5
        )
        
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=weighted_binary_crossentropy(class_weights),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    
    # Print model summary
    model.summary()
    
    # Create model trainer
    trainer = ModelTrainer(
        model,
        train_generator,
        val_generator,
        class_weights,
        train_steps,
        val_steps,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )
    
    # Train model
    print("Training model...")
    history = trainer.train(
        epochs=args.epochs,
        resume=args.resume,
        patience=10,
        fine_tune_at=249  # Fine-tune from block 5 of InceptionV3
    )
    
    # Visualize training history
    print("Visualizing training history...")
    visualizer = ModelVisualizer(model, class_names, output_dir='visualizations')
    visualizer.plot_training_history(
        history, 
        save_path='visualizations/training_history.png',
        interactive=args.interactive
    )
    
    # Load test data for evaluation
    test_generator, _, _, test_steps, _ = load_data(
        test_data_dir,
        test_csv_path,
        img_size=(299, 299),
        batch_size=args.batch_size,
        validation_split=0.0
    )
    
    # Collect test data for evaluation
    print("Evaluating on test data...")
    test_images = []
    test_labels = []
    
    for _ in tqdm(range(test_steps)):
        images, labels = next(test_generator)
        test_images.append(images)
        test_labels.append(labels)
    
    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_images)
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics_df = visualizer.calculate_metrics(test_labels, predictions)
    print(metrics_df)
    
    # Plot metrics table
    visualizer.plot_metrics_table(
        metrics_df,
        save_path='visualizations/metrics_table.png',
        interactive=args.interactive
    )
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        test_labels,
        predictions,
        save_path='visualizations/confusion_matrix.png',
        interactive=args.interactive
    )
    
    # Plot ROC curve
    visualizer.plot_roc_curve(
        test_labels,
        predictions,
        save_path='visualizations/roc_curve.png',
        interactive=args.interactive
    )
    
    # Plot precision-recall curve
    visualizer.plot_precision_recall_curve(
        test_labels,
        predictions,
        save_path='visualizations/precision_recall_curve.png',
        interactive=args.interactive
    )
    
    # Apply LIME for explainability
    if args.explain:
        print("Generating LIME explanations...")
        lime_explainer = LimeExplainer(model)
        
        # Select a subset of test images to explain
        n_explanations = min(10, len(test_images))
        
        for i in range(n_explanations):
            img = test_images[i]
            pred = predictions[i]
            
            # Get class with highest prediction
            class_idx = np.argmax(pred)
            
            # Generate explanation
            save_path = f'explanations/lime_explanation_{i}.png'
            lime_explainer.explain_and_visualize(
                img,
                save_path=save_path,
                class_idx=class_idx
            )
            
            print(f"Saved explanation {i+1}/{n_explanations} to {save_path}")
    
    # Apply feature selection using Genetic Algorithm if requested
    if args.feature_selection:
        print("Performing feature selection with Genetic Algorithm...")
        
        # Get features from a specific layer
        feature_layer = 'mixed_7'  # A layer from InceptionV3
        
        # Create feature extractor model
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(feature_layer).output
        )
        
        # Extract features for a subset of test data
        subset_size = min(500, len(test_images))
        subset_images = test_images[:subset_size]
        subset_labels = test_labels[:subset_size]
        
        features = feature_extractor.predict(subset_images)
        
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Initialize Genetic Algorithm
        ga = GeneticAlgorithm(
            population_size=20,
            n_generations=20,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        # Define fitness function
        def fitness_function(feature_mask):
            # Select features based on mask
            selected_features = features[:, feature_mask == 1]
            
            # If no features selected, return lowest fitness
            if selected_features.shape[1] == 0 or np.sum(feature_mask) < 10:
                return -np.inf
            
            # Train a simple model on selected features
            inputs = tf.keras.Input(shape=(selected_features.shape[1],))
            x = tf.keras.layers.Dense(64, activation='relu')(inputs)
            outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
            temp_model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            temp_model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train for a few epochs
            history = temp_model.fit(
                selected_features, subset_labels,
                epochs=5,
                validation_split=0.2,
                verbose=0
            )
            
            # Return validation accuracy as fitness
            val_acc = history.history['val_accuracy'][-1]
            
            # Penalize for using too many features
            penalty = 0.001 * np.sum(feature_mask) / features.shape[1]
            
            return val_acc - penalty
        
        # Run optimization
        best_mask, best_fitness = ga.optimize(fitness_function, features.shape[1])
        
        # Print results
        selected_count = np.sum(best_mask)
        total_count = features.shape[1]
        print(f"Feature selection complete. Selected {selected_count} out of {total_count} features.")
        print(f"Best fitness score: {best_fitness}")
        
        # Visualize feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(total_count), best_mask)
        plt.xlabel('Feature Index')
        plt.ylabel('Selected (1) / Not Selected (0)')
        plt.title('Feature Selection Results')
        plt.savefig('visualizations/feature_selection.png', dpi=300, bbox_inches='tight')
    
    print("All done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dermatological Diagnosis using Explainable AI')
    parser.add_argument('--data_dir', type=str, default='ISIC_2019/',
                        help='Directory containing the ISIC 2019 dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Tune hyperparameters using Grey Wolf Optimizer')
    parser.add_argument('--feature_selection', action='store_true',
                        help='Perform feature selection using Genetic Algorithm')
    parser.add_argument('--explain', action='store_true',
                        help='Generate LIME explanations for test images')
    parser.add_argument('--interactive', action='store_true',
                        help='Generate interactive Plotly visualizations')
    
    args = parser.parse_args()
    main(args)