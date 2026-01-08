import os
import random
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torchvision import transforms


# ---------------------------------------------------------
# 1. DEVICE HELPER
# ---------------------------------------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# 2. SEED FIXER (SUPER IMPORTANT for reproducibility)
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# 3. IMAGE PREPROCESSING HELPERS
# ---------------------------------------------------------
def load_image_pil(path, size=(224,224)):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return img, tf(img)


# ---------------------------------------------------------
# 4. METRICS (ACC, F1, Confusion Matrix)
# ---------------------------------------------------------
def compute_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)
    return acc, f1, cm, report


# ---------------------------------------------------------
# 5. MODEL SAVE/LOAD HELPERS
# ---------------------------------------------------------
def save_model(model, path="saved_models/model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✔ Model saved at: {path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=get_device()))
    print("✔ Model loaded:", path)
    return model


# ---------------------------------------------------------
# 6. TIMER (for tracking epoch time)
# ---------------------------------------------------------
class Timer:
    def __init__(self):
        self.start_time = datetime.now()
    
    def restart(self):
        self.start_time = datetime.now()

    def elapsed(self):
        delta = datetime.now() - self.start_time
        return str(delta).split(".")[0]  # remove ms


# ---------------------------------------------------------
# 7. LR SCHEDULER WRAPPER
# ---------------------------------------------------------
def get_scheduler(optimizer, mode="cosine"):
    if mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    if mode == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    if mode == "reduce":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    return None


# ---------------------------------------------------------
# 8. COUNT PARAMETERS
# ---------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
