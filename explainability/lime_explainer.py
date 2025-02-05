import lime
from lime import lime_image
import numpy as np

class LimeExplainer:
    def __init__(self, model):
        self.explainer = lime_image.LimeImageExplainer()
        self.model = model
        
    def explain(self, image, top_labels=5, hide_color=0, num_samples=1000):
        explanation = self.explainer.explain_instance(
            image.astype('double'),
            self.model.predict,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=num_samples
        )
        return explanation 