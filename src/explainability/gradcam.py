# src/explainability/gradcam.py
import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class SafeGradCAM(GradCAM):
    """Windows-safe override for GradCAM to prevent destructor crash."""
    def __del__(self):
        try:
            if hasattr(self, "activations_and_grads"):
                self.activations_and_grads.release()
        except Exception:
            pass
        return


def generate_gradcam(model, img_tensor, original_image, target_layer):
    """
    Generates Grad-CAM heatmap for a given MRI input.

    Args:
        model: PyTorch HybridMRIModel
        img_tensor: (1, 3, 224, 224) normalized tensor
        original_image: PIL.Image object
        target_layer: model.cnn.layer4[-1]

    Returns:
        heatmap_img: uint8 (224,224,3)
    """

    model.eval()

    # Initialize SafeGradCAM
    cam = SafeGradCAM(
        model=model,
        target_layers=[target_layer]
    )

    try:
        grayscale_cam = cam(input_tensor=img_tensor, eigen_smooth=False)[0]
    except Exception as e:
        try:
            cam.activations_and_grads.release()
        except:
            pass
        del cam
        raise RuntimeError(f"GradCAM failed: {e}")

    try:
        cam.activations_and_grads.release()
    except:
        pass

    del cam

    # Prepare original image
    img_np = np.array(original_image.resize((224, 224))).astype(np.float32) / 255.0

    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Heatmap overlay
    heatmap_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return heatmap_img
