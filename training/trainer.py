import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger
import numpy as np
import time
from datetime import datetime

class ModelTrainer:
    """
    Class to handle model training with checkpoints, early stopping, and TensorBoard
    """
    def __init__(self, 
                 model, 
                 train_generator, 
                 val_generator, 
                 class_weights,
                 train_steps,
                 val_steps,
                 checkpoint_dir='checkpoints',
                 log_dir='logs'):
        """
        Initialize trainer
        
        Args:
            model: Compiled Keras model
            train_generator: Training data generator
            val_generator: Validation data generator
            class_weights: Class weights for imbalanced data
            train_steps: Number of steps per training epoch
            val_steps: Number of steps per validation epoch
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.class_weights = class_weights
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Timestamp for unique folder names
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create checkpoint and log directories for this run
        self.run_checkpoint_dir = os.path.join(checkpoint_dir, self.timestamp)
        self.run_log_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.run_checkpoint_dir, exist_ok=True)
        os.makedirs(self.run_log_dir, exist_ok=True)
        
    def get_callbacks(self, patience=10):
        """
        Create training callbacks
        
        Args:
            patience: Patience for early stopping
            
        Returns:
            List of callbacks
        """
        # ModelCheckpoint to save best model
        checkpoint_path = os.path.join(self.run_checkpoint_dir, 'model_{epoch:02d}_{val_loss:.4f}.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,  # Save weights only to avoid custom loss issues
            mode='min',
            verbose=1
        )
        
        # ModelCheckpoint to save last model for resuming training
        last_checkpoint_path = os.path.join(self.run_checkpoint_dir, 'last_model.h5')
        last_checkpoint = ModelCheckpoint(
            last_checkpoint_path,
            save_best_only=False,
            save_weights_only=True,  # Save weights only to avoid custom loss issues
            verbose=0
        )
        
        # EarlyStopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # TensorBoard for visualizing metrics
        tensorboard = TensorBoard(
            log_dir=self.run_log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        
        # Reduce learning rate when a metric has stopped improving
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # CSV Logger to save metrics to a CSV file
        csv_logger = CSVLogger(
            os.path.join(self.run_log_dir, 'training_log.csv'),
            append=True,
            separator=','
        )
        
        return [checkpoint, last_checkpoint, early_stopping, tensorboard, reduce_lr, csv_logger]
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint to resume training"""
        checkpoints = []
        for d in os.listdir(self.checkpoint_dir):
            checkpoint_dir = os.path.join(self.checkpoint_dir, d)
            if os.path.isdir(checkpoint_dir):
                last_model_path = os.path.join(checkpoint_dir, 'last_model.h5')
                if os.path.exists(last_model_path):
                    checkpoints.append((checkpoint_dir, os.path.getmtime(last_model_path)))
        
        if checkpoints:
            # Sort by modification time (most recent first)
            checkpoints.sort(key=lambda x: x[1], reverse=True)
            latest_checkpoint_dir = checkpoints[0][0]
            latest_model_path = os.path.join(latest_checkpoint_dir, 'last_model.h5')
            return latest_model_path
        
        return None
    
    def train(self, epochs=100, resume=True, patience=10, fine_tune_at=0):
        """
        Train the model
        
        Args:
            epochs: Maximum number of epochs to train
            resume: Whether to resume from last checkpoint
            patience: Patience for early stopping
            fine_tune_at: Layer to fine-tune from (0 to train all layers)
            
        Returns:
            Training history
        """
        initial_epoch = 0
        
        # Resume training if requested
        if resume:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                print(f"Resuming training from {latest_checkpoint}")
                try:
                    # Try to load the full model
                    self.model = tf.keras.models.load_model(latest_checkpoint)
                    print("Model loaded successfully!")
                except Exception as e:
                    print(f"Error loading model: {e}")
                    print("Attempting to load weights only...")
                    
                    try:
                        # Create a temporary model with the same architecture
                        temp_model = tf.keras.models.load_model(
                            latest_checkpoint, 
                            compile=False  # Don't load the optimizer and loss function
                        )
                        
                        # Copy weights layer by layer
                        for i, layer in enumerate(self.model.layers):
                            if i < len(temp_model.layers):
                                try:
                                    layer.set_weights(temp_model.layers[i].get_weights())
                                except:
                                    print(f"Could not transfer weights for layer {i}: {layer.name}")
                        
                        print("Model weights loaded successfully via layer-by-layer transfer!")
                    except Exception as e2:
                        print(f"Error loading weights layer by layer: {e2}")
                        print("Attempting direct weights loading...")
                        
                        try:
                            # Try direct weights loading as a last resort
                            self.model.load_weights(latest_checkpoint)
                            print("Model weights loaded successfully via direct loading!")
                        except Exception as e3:
                            print(f"Error directly loading weights: {e3}")
                            print("Starting training from scratch.")
                
                # Get initial epoch from checkpoint dirname
                checkpoint_dirname = os.path.dirname(latest_checkpoint)
                log_path = os.path.join(os.path.dirname(checkpoint_dirname), 
                                       os.path.basename(checkpoint_dirname), 
                                       'training_log.csv')
                
                if os.path.exists(log_path):
                    import pandas as pd
                    log_df = pd.read_csv(log_path)
                    initial_epoch = log_df.shape[0]
                    print(f"Resuming from epoch {initial_epoch}")
        
        # Get callbacks
        callbacks = self.get_callbacks(patience=patience)
        
        # First train the top layers with base model frozen
        history1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_steps,
            epochs=int(epochs * 0.3),  # 30% of epochs for initial training
            validation_data=self.val_generator,
            validation_steps=self.val_steps,
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )
        
        # Then fine-tune if requested
        if fine_tune_at > 0:
            print("Fine-tuning model layers...")
            # Unfreeze layers for fine-tuning
            for layer in self.model.layers[fine_tune_at:]:
                layer.trainable = True
                
            # Recompile model with lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            
            # Continue training
            # Note: Removed class_weight argument since it's not supported with generators
            history2 = self.model.fit(
                self.train_generator,
                steps_per_epoch=self.train_steps,
                epochs=epochs,
                validation_data=self.val_generator,
                validation_steps=self.val_steps,
                callbacks=callbacks,
                initial_epoch=int(epochs * 0.3)
            )
            
            # Combine histories
            for k in history1.history:
                history1.history[k].extend(history2.history[k])
                
        return history1