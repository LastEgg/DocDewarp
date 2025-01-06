from copy import deepcopy
from random import random, seed, shuffle
from typing import *

import torch
import torch.nn.functional as FN
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from .inv3d_util.load import load_json
from .inv3d_util.path import check_dir, list_dirs
from .loaders import *

import time
import cv2
import os

class Inv3DDataset(Dataset):
    def __init__(
        self,
        opt,
        limit_samples: Optional[int] = None,
        repeat_samples: Optional[int] = None,
        
    ):
        self.source_dir = check_dir(opt.data_path)
        self.patch_size = opt.patch_size
        self.color_transform = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        )

        self.samples = list(sorted([s.name for s in list_dirs(self.source_dir)]))
        seed(42)
        self.unwarp_factors = [random() for _ in self.samples]

        if limit_samples:
            self.samples = self.samples[:limit_samples]

        if repeat_samples:
            self.samples = self.samples * repeat_samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = self.source_dir / sample

        unwarp_factor = self.unwarp_factors[idx]
        start = time.time()

        image = prepare_masked_image(
            sample_dir / "warped_document.png",
            sample_dir / "warped_UV.npz",
            sample_dir / "warped_BM.npz",
            unwarp_factor=unwarp_factor,
        )

        albedo = prepare_masked_image(
            sample_dir / "warped_albedo.png",
            sample_dir / "warped_UV.npz",
            sample_dir / "warped_BM.npz",
            unwarp_factor=unwarp_factor,
            bg_color=(255, 255, 255),
        )
        assert image.shape == albedo.shape
        output_dir = str(sample_dir).replace("inv3d_train_part_2_of_4/train_part_2_of_4", "inv3d/train")
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "label.png"), albedo.transpose(1, 2, 0))
        cv2.imwrite(os.path.join(output_dir, "image.png"), image.transpose(1, 2, 0))

        # randomly crop single patch from images
        top, left, height, width = transforms.RandomCrop.get_params(
            image[0], (self.patch_size, self.patch_size)
        )

        image = F.crop(torch.from_numpy(image), top, left, height, width)
        # albedo = F.crop(torch.from_numpy(albedo), top, left, height, width)

        image = (image.numpy() * 255).astype("uint8")
        image = rearrange(image, "c h w -> h w c")
        # image = self.color_transform(image)
        image = rearrange(image, "h w c -> c h w")
        image = torch.from_numpy(image.astype("float32") / 255)

        # end = time.time()
        # print(f"times: {end - start}")

        return {"image": image}


class Inv3DRealUnwarpDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        num_text_evals: int = 64,
        **kwargs,
    ):
        self.source_dir = check_dir(source_dir)
        self.num_text_evals = num_text_evals
        self.samples = list(
            sorted(
                [
                    sample.absolute()
                    for sample in list_dirs(self.source_dir, recursive=True)
                    if sample.name.startswith("warped")
                ]
            )
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        data = {}
        data["sample"] = str(sample_dir)
        data["index"] = idx

        template = load_image(sample_dir / "template.png")
        template = rearrange(template, "h w c -> c h w")
        template = template.astype("float32") / 255
        data["input.template"] = template

        image = load_image(sample_dir / "norm_image.png")
        image = rearrange(image, "h w c -> c h w")
        image = image.astype("float32") / 255
        data["input.image"] = image

        return unflatten(data)


class Inv3DTestDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        unwarp_factor: float,
        limit_samples: Optional[int] = None,
    ):
        self.source_dir = check_dir(source_dir)
        self.unwarp_factor = unwarp_factor

        self.samples = list(sorted([s.name for s in list_dirs(self.source_dir)]))

        if limit_samples is not None:
            self.samples = self.samples[:limit_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = self.source_dir / sample

        image = prepare_masked_image(
            sample_dir / "warped_document.png",
            sample_dir / "warped_UV.npz",
            sample_dir / "warped_BM.npz",
            unwarp_factor=self.unwarp_factor,
            resolution=template.shape[1:],
        )

        albedo = prepare_masked_image(
            sample_dir / "warped_albedo.png",
            sample_dir / "warped_UV.npz",
            sample_dir / "warped_BM.npz",
            unwarp_factor=self.unwarp_factor,
            bg_color=(255, 255, 255),
            resolution=template.shape[1:],
        )
        assert image.shape == albedo.shape

        return {"image": image, "label": albedo}

def get_dataloader(opt):
    train_dataset = Inv3DDataset(
        opt=opt,
    )
    valid_dataset = Inv3DDataset(
        opt=opt,
        limit_samples=1000
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    from configs.option import get_option
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    for i, batch in enumerate(train_dataloader):
        print(batch["image"].shape, batch["label"].shape)
        break