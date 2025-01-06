import torch

def load_model(model, path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)

    pretrained_dict = {
        k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict
    }

    model_dict.update(pretrained_dict)
    msg = model.load_state_dict(model_dict)

    return msg

def image2patches(image: torch.Tensor, patch_res: int, overlap: int):
    check_tensor(image, "n c h w")

    shift = patch_res - overlap

    _, _, H, W = image.shape

    padH = (int((H - patch_res) / (shift) + 1) * (shift) + patch_res) - H
    padW = (int((W - patch_res) / (shift) + 1) * (shift) + patch_res) - W

    image_pad = F.pad(image, (0, padW, 0, padH))

    patches = image_pad.unfold(-2, patch_res, step=shift)
    patches = patches.unfold(-2, patch_res, step=shift)
    patches = rearrange(patches, "n c y x h w -> n y x h w c")

    patches = patches.clone()  # important before assigning

    # overwrite last row
    row = image[:, :, -patch_res:, :].unfold(-1, patch_res, step=shift)
    row = rearrange(row, "n c h a w -> n a h w c")
    patches[:, -1, : row.shape[1], ...] = row

    # overwrite last column
    col = image[:, :, :, -patch_res:].unfold(-2, patch_res, step=shift)
    col = rearrange(col, "n c a h w -> n a w h c")
    patches[:, : col.shape[1], -1, :, :, :] = col

    # overwrite corner case
    corner = image[:, :, -patch_res:, -patch_res:]
    corner = rearrange(corner, "n c h w -> n h w c")
    patches[:, -1, -1, ...] = corner

    return rearrange(patches, "n y x h w c -> n y x c h w")

def patches2images_average(
    patches: torch.Tensor,
    height: int,
    width: int,
    patch_res: int = 128,
    overlap: int = 16,
) -> torch.Tensor:
    dims = check_tensor(patches, "n y x c h w", c=3, h=patch_res, w=patch_res)
    n = dims["n"]

    patches = rearrange(patches, "n y x c h w -> n y x h w c")

    # coordinate grid with height x width
    image_coords = torch.stack(
        torch.meshgrid(torch.arange(height), torch.arange(width))
    )

    # cut out patches from the coordinate grid and flatten coordinates
    patches_coords = image2patches(
        image_coords.unsqueeze(0), patch_res=patch_res, overlap=overlap
    )
    patches_coords = rearrange(patches_coords, "n y x c h w -> n (y x h w) c")
    patches_coords = patches_coords[:, :, 0] * width + patches_coords[:, :, 1]
    patches_coords = patches_coords + (torch.arange(n) * height * width).unsqueeze(-1)
    patches_coords = patches_coords.reshape(-1)
    patches_coords = patches_coords.unsqueeze(-1).repeat(1, 3)

    # flatten patches
    patches = rearrange(patches, "n y x h w c -> (n y x h w) c")

    # calculate sum and count image
    image_sums = torch.zeros((n * height * width, 3), dtype=torch.float).scatter_add(
        0, patches_coords, patches.float()
    )
    image_counts = torch.zeros((n * height * width, 3), dtype=torch.float).scatter_add(
        0, patches_coords, torch.ones_like(patches).float()
    )

    # finalize image
    image_sums = image_sums.reshape(n, height, width, 3)
    image_counts = image_counts.reshape(n, height, width, 3)

    result = (image_sums / image_counts).type(patches.dtype)

    return rearrange(result, "n h w c -> n c h w")


def process_patches(
    model: torch.nn.Module, patches: torch.Tensor, batch_size: int
) -> torch.Tensor:
    check_tensor(patches, "n y x c h h", c=3)

    n, y, x, c, h, w = patches.shape

    patches = patches.reshape(-1, c, h, w)

    out = torch.concat(
        [model(batch).detach().cpu() for batch in torch.split(patches, batch_size)]
    )

    return out.reshape(n, y, x, c, h, w)