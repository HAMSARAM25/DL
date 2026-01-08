# src/explainability/slice_visualizer.py
import math

import numpy as np
from PIL import Image


def _make_grid(images, cols=5, img_size=(224,224)):
    """
    images: list of PIL.Image or numpy arrays (H,W) or (H,W,3)
    cols: number columns
    returns: numpy uint8 grid image (H_grid, W_grid, 3)
    """
    imgs = []
    for im in images:
        if isinstance(im, Image.Image):
            im_r = im.resize(img_size).convert("RGB")
            imgs.append(np.array(im_r))
        else:
            arr = np.array(im)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.shape[-1] != 3:
                arr = arr[..., :3]
            # resize via PIL
            imgs.append(np.array(Image.fromarray(arr).resize(img_size).convert("RGB")))
    rows = math.ceil(len(imgs) / cols)
    h, w, _ = imgs[0].shape
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8) + 255
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = im
    return grid

def visualize_slices(slices):
    """
    Accepts:
      - PIL.Image (single) -> returns resized numpy image
      - list/tuple of PIL.Image/numpy arrays -> returns tiled grid numpy image
      - numpy array of shape (N,H,W) or (N,H,W,3)
    Returns:
      numpy array HxWx3 uint8 ready for st.image()
    """
    from PIL import Image

    # single PIL.Image
    if isinstance(slices, Image.Image):
        im = slices.resize((224,224)).convert("RGB")
        return np.array(im)

    # numpy array
    if isinstance(slices, np.ndarray):
        if slices.ndim == 3:  # (N, H, W)
            imgs = [slices[i] for i in range(slices.shape[0])]
            return _make_grid(imgs)
        if slices.ndim == 4:  # (N, H, W, C)
            imgs = [slices[i] for i in range(slices.shape[0])]
            return _make_grid(imgs)

    # list/tuple
    if isinstance(slices, (list, tuple)):
        # convert elements
        return _make_grid(list(slices))

    # fallback: try to convert to PIL and return single image
    try:
        im = Image.fromarray(np.array(slices)).resize((224,224)).convert("RGB")
        return np.array(im)
    except Exception:
        # return blank image
        return np.zeros((224,224,3), dtype=np.uint8)
