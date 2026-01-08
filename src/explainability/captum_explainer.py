# src/explainability/captum_explainer.py

import numpy as np
import torch
from captum.attr import IntegratedGradients


def captum_explain_single(model, img_tensor, target_idx):
    """
    Generate Integrated Gradients heatmap for a single MRI image.
    Works 100% with PyTorch. No TensorFlow required.
    """

    model.eval()
    model.zero_grad()

    # Initialize Captum IG
    ig = IntegratedGradients(model)

    # Compute attribution
    attributions = ig.attribute(img_tensor, target=target_idx, n_steps=50)

    # Convert to numpy (C,H,W)
    attr = attributions.squeeze().detach().cpu().numpy()

    # Reduce channels -> (H,W)
    if attr.ndim == 3:
        attr = np.mean(np.abs(attr), axis=0)

    # Normalize heatmap 0-1
    attr = (attr - attr.min()) / (attr.max() + 1e-6)

    return attr
