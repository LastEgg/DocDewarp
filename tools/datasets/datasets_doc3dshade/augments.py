import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(opt):
    train_transform = None

    valid_transform = None

    return train_transform, valid_transform