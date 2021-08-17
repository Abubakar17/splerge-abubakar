import cv2
import numpy as np
import torch
from skimage import transform as sktsf
from torchvision import transforms as tvtsf


def normalize_numpy_image(image):
    # Normalizing image
    image = image / 255.0
    norm = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = norm(torch.from_numpy(image))

    return image


def resize_image(image, min_size=600, max_size=1024):
    # Rescaling Images
    H, W, C = image.shape
    min_size = min_size
    max_size = max_size
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)

    image = sktsf.resize(
        image, (round(H * scale), round(W * scale), C), mode="reflect", anti_aliasing=False
    )

    return image