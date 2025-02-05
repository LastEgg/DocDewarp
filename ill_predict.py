import os 
import cv2 
import glob
from pathlib import Path
import utils
import argparse
import numpy as np
import torch
import time
from models import UNext_full_resolution_padding_L_py_L, RetinexFormer

def stride_integral(img,stride=32):
    h,w = img.shape[:2]

    if (h%stride)!=0:
        padding_h = stride - (h%stride)
        img = cv2.copyMakeBorder(img,padding_h,0,0,0,borderType=cv2.BORDER_REPLICATE)
    else:
        padding_h = 0
    
    if (w%stride)!=0:
        padding_w = stride - (w%stride)
        img = cv2.copyMakeBorder(img,0,0,padding_w,0,borderType=cv2.BORDER_REPLICATE)
    else:
        padding_w = 0
    
    return img,padding_h,padding_w

def appearance_prompt(img):
    h,w = img.shape[:2]
    # img = cv2.resize(img,(128,128))
    img = cv2.resize(img,(1024,1024))
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm,(w,h))
    return result_norm

def drnet_load(model_path="./checkpoints/baseline_drnet_V1.1/epoch_83-loss_0.016.ckpt"):
    model = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6,img_size=512).cuda()
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k.replace("model.", "")] = v
    model.load_state_dict(update_dict)
    model.eval()
    return model 

def retinexformer_load(model_path="./checkpoints/baseline_retinexformer_V1.0/epoch_94-loss_0.021.ckpt"):
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2]).cuda()
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k.replace("model.", "")] = v
    model.load_state_dict(update_dict)
    model.eval()
    return model 

def drnet_infer(model,im_path):
    MAX_SIZE=1600
    # obtain im and prompt
    im_org = cv2.imread(im_path)
    h,w = im_org.shape[:2]
    prompt = appearance_prompt(im_org)
    in_im = np.concatenate((im_org,prompt),-1)


    # constrain the max resolution 
    if max(w,h) < MAX_SIZE:
        in_im,padding_h,padding_w = stride_integral(in_im)
    else:
        in_im = cv2.resize(in_im,(MAX_SIZE,MAX_SIZE))
    
    # normalize
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0).float().cuda()

    # inference
    with torch.no_grad():
        pred, _, _, _ = model(in_im)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        if max(w,h) < MAX_SIZE:
            out_im = pred[padding_h:,padding_w:]
        else:
            pred[pred==0] = 1
            shadow_map = cv2.resize(im_org,(MAX_SIZE,MAX_SIZE)).astype(float)/pred.astype(float)
            shadow_map = cv2.resize(shadow_map,(w,h))
            shadow_map[shadow_map==0]=0.00001
            out_im = np.clip(im_org.astype(float)/shadow_map,0,255).astype(np.uint8)

    return out_im

def retinexformer_infer(model,im_path):
    MAX_SIZE=1600
    # obtain im and prompt
    in_im = cv2.imread(im_path)
    h,w = im_org.shape[:2]

    # constrain the max resolution 
    if max(w,h) < MAX_SIZE:
        in_im,padding_h,padding_w = stride_integral(in_im)
    else:
        in_im = cv2.resize(in_im,(MAX_SIZE,MAX_SIZE))
    
    # normalize
    in_im = in_im / 255.0
    in_im = torch.from_numpy(in_im.transpose(2,0,1)).unsqueeze(0).float().cuda()

    # inference
    with torch.no_grad():
        pred = model(in_im)
        pred = torch.clamp(pred,0,1)
        pred = pred[0].permute(1,2,0).cpu().numpy()
        pred = (pred*255).astype(np.uint8)

        if max(w,h) < MAX_SIZE:
            out_im = pred[padding_h:,padding_w:]
        else:
            pred[pred==0] = 1
            shadow_map = cv2.resize(im_org,(MAX_SIZE,MAX_SIZE)).astype(float)/pred.astype(float)
            shadow_map = cv2.resize(shadow_map,(w,h))
            shadow_map[shadow_map==0]=0.00001
            out_im = np.clip(im_org.astype(float)/shadow_map,0,255).astype(np.uint8)

    return out_im

if __name__ == '__main__':

    model = drnet_load()
    im_path = "your_image_path.jpg"
    start = time.time()
    out_im = drnet_infer(model,im_path)
    cv2.imwrite("temp.jpg",out_im)
    end = time.time()
    print(f"推理时间为：{end - start}")
