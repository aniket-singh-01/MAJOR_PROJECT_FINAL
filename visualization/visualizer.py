import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    f1_score, 
    precision_score, 
    recall_score
)
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelVisualizer:
    """
    Class to visualize model metrics and performance
    """
    def __init__(self, model, class_names, output_dir='visualizations'):
        """
        Initialize visualizer
        
        Args:
            model: Trained model
            class_names: List of class names
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.class_names = class_names
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_training_history(self, history, save_path=None, interactive=False):
        """
        Plot training history
        
        Args:
            history: Training history from model.fit()
            save_path: Path to save the plot
            interactive: Whether to create interactive plots with Plotly
            
        Returns:
            None
        """
        if interactive:
            # Create subplot with shared x-axis
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            
            # Add loss traces
            fig.add_trace(
                go.Scatter(x=list(range(len(history.history['loss']))), 
                         y=history.history['loss'],
                         mode='lines',
                         name='Training Loss'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(len(history.history['val_loss']))), 
                         y=history.history['val_loss'],
                         mode='lines',
                         name='Validation Loss'),
                row=1, col=1
            )
            
            # Add accuracy traces
            fig.add_trace(
                go.Scatter(x=list(range(len(history.history['accuracy']))), 
                         y=history.history['accuracy'],
                         mode='lines',
                         name='Training Accuracy'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(len(history.history['val_accuracy']))), 
                         y=history.history['val_accuracy'],
                         mode='lines',
                         name='Validation Accuracy'),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Training and Validation Metrics',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                yaxis2_title='Accuracy',
                height=800,
                width=1000
            )
            
            # Save or show
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Plot loss
            ax1.plot(history.history['loss'], label='Training Loss')
            ax1.plot(history.history['val_loss'], label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracy
            ax2.plot(history.history['accuracy'], label='Training Accuracy')
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, interactive=False):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            save_path: Path to save the plot
            interactive: Whether to create interactive plots with Plotly
            
        Returns:
            None
        """
        # Convert probabilities to class labels
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        if interactive:
            # Create interactive confusion matrix with Plotly
            fig = px.imshow(
                cm,
                x=self.class_names,
                y=self.class_names,
                labels=dict(x="Predicted", y="True", color="Count"),
                text_auto=True,
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                title='Confusion Matrix',
                width=800,
                height=800
            )
            
            # Save or show
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            # Create confusion matrix with Seaborn
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def plot_roc_curve(self, y_true, y_pred, save_path=None, interactive=False):
        """
        Plot ROC curve for each class
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            save_path: Path to save the plot
            interactive: Whether to create interactive plots with Plotly
            
        Returns:
            None
        """
        n_classes = y_true.shape[1]
        
        if interactive:
            # Create interactive ROC curves with Plotly
            fig = go.Figure()
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(
                    go.Scatter(
                        x=fpr, 
                        y=tpr,
                        name=f"{self.class_names[i]} (AUC = {roc_auc:.3f})",
                        mode='lines'
                    )
                )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], 
                    y=[0, 1],
                    name='Chance',
                    mode='lines',
                    line=dict(dash='dash', color='gray')
                )
            )
            
            fig.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=800,
                height=600,
                legend=dict(x=0.7, y=0.1),
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05])
            )
            
            # Save or show
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            # Create ROC curves with Matplotlib
            plt.figure(figsize=(10, 8))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2,
                       label=f"{self.class_names[i]} (AUC = {roc_auc:.3f})")
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_pred, save_path=None, interactive=False):
        """
        Plot precision-recall curve for each class
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            save_path: Path to save the plot
            interactive: Whether to create interactive plots with Plotly
            
        Returns:
            None
        """
        n_classes = y_true.shape[1]
        
        if interactive:
            # Create interactive precision-recall curves with Plotly
            fig = go.Figure()
            
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
                
                fig.add_trace(
                    go.Scatter(
                        x=recall, 
                        y=precision,
                        name=f"{self.class_names[i]}",
                        mode='lines'
                    )
                )
            
            fig.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=800,
                height=600,
                legend=dict(x=0.7, y=0.1),
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05])
            )
            
            # Save or show
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            # Create precision-recall curves with Matplotlib
            plt.figure(figsize=(10, 8))
            
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
                
                plt.plot(recall, precision, lw=2,
                       label=f"{self.class_names[i]}")
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="best")
            plt.grid(True)
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        """
        Calculate performance metrics
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            threshold: Probability threshold for positive class
            
        Returns:
            DataFrame with metrics
        """
        n_classes = y_true.shape[1]
        
        # Convert probabilities to binary predictions using threshold
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Initialize metrics
        metrics = {
            'Class': self.class_names,
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'AUC': []
        }
        
        # Calculate metrics for each class
        for i in range(n_classes):
            precision = precision_score(y_true[:, i], y_pred_binary[:, i])
            recall = recall_score(y_true[:, i], y_pred_binary[:, i])
            f1 = f1_score(y_true[:, i], y_pred_binary[:, i])
            
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            
            metrics['Precision'].append(precision)
            metrics['Recall'].append(recall)
            metrics['F1-Score'].append(f1)
            metrics['AUC'].append(roc_auc)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Add overall accuracy
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        
        # Add macro averages
        metrics_df.loc[n_classes] = [
            'Macro Average',
            np.mean(metrics['Precision']),
            np.mean(metrics['Recall']),
            np.mean(metrics['F1-Score']),
            np.mean(metrics['AUC'])
        ]
        
        metrics_df.loc[n_classes+1] = [
            'Accuracy',
            accuracy,
            accuracy,
            accuracy,
            accuracy
        ]
        
        return metrics_df
    
    def plot_metrics_table(self, metrics_df, save_path=None, interactive=False):
        """
        Plot metrics table
        
        Args:
            metrics_df: DataFrame with metrics
            save_path: Path to save the plot
            interactive: Whether to create interactive plot with Plotly
            
        Returns:
            None
        """
        if interactive:
            # Create interactive table with Plotly
            fig = go.Figure(data=[
                go.Table(
                    header=dict(
                        values=list(metrics_df.columns),
                        fill_color='paleturquoise',
                        align='center'
                    ),
                    cells=dict(
                        values=[metrics_df[col] for col in metrics_df.columns],
                        fill_color='lavender',
                        align='center',
                        format=[None, '.4f', '.4f', '.4f', '.4f']
                    )
                )
            ])
            
            fig.update_layout(
                title='Model Performance Metrics',
                width=800,
                height=600
            )
            
            # Save or show
            if save_path:
                if not save_path.endswith('.html'):
                    save_path += '.html'
                fig.write_html(save_path)
            else:
                fig.show()
        else:
            # Create table with Matplotlib
            plt.figure(figsize=(10, 6))
            
            # Hide axes
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            # Create table
            table = plt.table(
                cellText=metrics_df.values,
                colLabels=metrics_df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            
            plt.title('Model Performance Metrics', pad=20)
            
            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()