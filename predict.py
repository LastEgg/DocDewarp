import argparse

import cv2
import torch
from utils import *
from models.geo_tr import *


def run(args):
    image_path = args.image
    model_path = args.model
    output_path = args.output

    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    model = GeoTr().cuda()
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k.replace("model.", "")] = v
    model.load_state_dict(update_dict)
    model.eval()

    img_org = cv2.imread(image_path)
    img = cv2.resize(img_org, (288, 288))
    x = to_tensor(img).cuda()
    y = to_tensor(img_org).cuda()
    bm = model(x)
    bm = torch.nn.functional.interpolate(
        bm, y.shape[2:], mode="bilinear", align_corners=False
    )
    bm_nhwc = bm.permute([0, 2, 3, 1])
    out = torch.nn.functional.grid_sample(y, (bm_nhwc / 288 - 0.5) * 2).cpu().detach().numpy()
    out_image = to_image(out)
    print(out_image.shape)
    cv2.imwrite(output_path, out_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict")

    parser.add_argument(
        "--image",
        "-i",
        nargs="?",
        type=str,
        default="/root/dewarp/data/doc3d/img_bld/001/001-0_0001_h500-WkN0001.png",
        help="The path of image",
    )

    parser.add_argument(
        "--model",
        "-m",
        nargs="?",
        type=str,
        default="/root/dewarp/DocDewarp/checkpoints/baselinev1/epoch_88-acc_0.655.ckpt",
        help="The path of model",
    )

    parser.add_argument(
        "--output",
        "-o",
        nargs="?",
        type=str,
        default="/root/dewarp/DocDewarp/test_dir/output/output.png",
        help="The path of output",
    )

    args = parser.parse_args()

    print(args)

    run(args)
