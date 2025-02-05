import argparse
import os
import cv2
import torch
from tool.utils import *
from models.geo_tr import *


def geotr_load(model_path="./checkpoints/baseline_geotr_V1.0/epoch_88-acc_0.655.ckpt"):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    model = GeoTr().cuda()
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k.replace("model.", "")] = v
    model.load_state_dict(update_dict)
    model.eval()
    return model 


def geotr_infer(model,im_path):
    img_org = cv2.imread(im_path)
    img = cv2.resize(img_org, (288, 288))
    x = to_tensor(img).cuda()
    y = to_tensor(img_org).cuda()
    bm = model(x)
    bm = torch.nn.functional.interpolate(
        bm, y.shape[2:], mode="bilinear", align_corners=False
    )
    bm_nhwc = bm.permute([0, 2, 3, 1])
    out = torch.nn.functional.grid_sample(y, (bm_nhwc / 288 - 0.5) * 2).cpu().detach().numpy()
    out_im = to_image(out)

    return out_im


if __name__ == '__main__':

    model = geotr_load()
    im_path = "your_image_path.jpg"
    start = time.time()
    out_im = geotr_infer(model,im_path)
    cv2.imwrite("temp.jpg",out_im)
    end = time.time()
    print(f"推理时间为：{end - start}")

