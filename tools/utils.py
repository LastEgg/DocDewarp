import torch
import numpy as np
from torch.nn import functional as F


def load_model(model, path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)

    pretrained_dict = {
        k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict
    }

    model_dict.update(pretrained_dict)
    msg = model.load_state_dict(model_dict)

    return msg

def to_tensor(img: np.ndarray):
    img = img[:, :, ::-1]
    img = img.astype("float32") / 255.0
    img = img.transpose(2, 0, 1)
    out= torch.tensor(img)
    out = torch.unsqueeze(out, axis=0)

    return out

def to_image(x):
    out: np.ndarray = x.squeeze()
    out = out.transpose(1, 2, 0)
    out = out * 255.0
    out = out.astype("uint8")
    out = out[:, :, ::-1]

    return out