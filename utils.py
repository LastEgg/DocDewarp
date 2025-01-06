import numpy as np
import torch
from torch.nn import functional as F


def to_tensor(img: np.ndarray):
    """
    Converts a numpy array image (HWC) to a Paddle tensor (NCHW).

    Args:
        img (numpy.ndarray): The input image as a numpy array.

    Returns:
        out (paddle.Tensor): The output tensor.
    """
    img = img[:, :, ::-1]
    img = img.astype("float32") / 255.0
    img = img.transpose(2, 0, 1)
    out= torch.tensor(img)
    out = torch.unsqueeze(out, axis=0)

    return out


def to_image(x):
    """
    Converts a Paddle tensor (NCHW) to a numpy array image (HWC).

    Args:
        x (paddle.Tensor): The input tensor.

    Returns:
        out (numpy.ndarray): The output image as a numpy array.
    """
    out: np.ndarray = x.squeeze()
    out = out.transpose(1, 2, 0)
    out = out * 255.0
    out = out.astype("uint8")
    out = out[:, :, ::-1]

    return out