import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(opt):
    train_transform = A.Compose(
                [
                    A.ColorJitter(),
                ]
            )

    valid_transform = A.Compose(
                [
                    A.ColorJitter(),
                ]
            )
    return train_transform, valid_transform
