"""
Project: Dermatological Diagnosis using Explainable AI (XAI)

Build a deep learning model using the following:
- Use InceptionV3 as the base model
- Dataset: ISIC 2019 (already present in 'ISIC_2019/' directory)
- Use CNN layers on top of InceptionV3
- Apply LIME for explainability
- Use Grey Wolf Optimizer (GWO) for hyperparameter tuning
- Use Genetic Algorithm for feature selection (if needed)
- Aim for Accuracy > 95%
- Include training resumption from checkpoints
- Save and visualize: accuracy, precision, recall, F1-score, confusion matrix, ROC curve
- Output visualizations as PNGs or interactive plots (Matplotlib/Seaborn/Plotly)
- Use separate files for each module: `model/`, `optimizer/`, `xai/`, `utils/`, `training/`, `visualization/`
- Provide support for TensorBoard and logs

Ensure:
- Training resumes from last checkpoint if interrupted
- Use early stopping and model checkpointing
"""
 