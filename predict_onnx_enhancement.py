'''
用于图像增强推理, onnx版本
算法支持: drnet
功能支持：单张推理、文件夹推理
'''
import cv2
import onnxruntime as ort
import numpy as np
import os
from tqdm import tqdm

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

def load_drnet_model(model_path='./checkpoints/drnet_model_V1.2.onnx'):
    ort_session = ort.InferenceSession(model_path)
    return ort_session

def drnet_infer_onnx(ort_session, im_path):
    # 初始化 ONNX Runtime session
    MAX_SIZE = 1600
    im_org = cv2.imread(im_path)
    h, w = im_org.shape[:2]
    prompt = appearance_prompt(im_org)  # 确保实现了 appearance_prompt
    in_im = np.concatenate((im_org, prompt), -1)

    if max(w, h) < MAX_SIZE:
        in_im, padding_h, padding_w = stride_integral(in_im)
    else:
        in_im = cv2.resize(in_im, (MAX_SIZE, MAX_SIZE))

    in_im = in_im / 255.0
    in_im = in_im.transpose(2, 0, 1).astype(np.float32)
    in_im = np.expand_dims(in_im, axis=0)

    # 推理
    ort_inputs = {ort_session.get_inputs()[0].name: in_im}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = ort_outs[0][0]
    pred = np.transpose(pred, (1, 2, 0))

    pred = np.clip(pred, 0, 1)
    pred = (pred * 255).astype(np.uint8)

    if max(w, h) < MAX_SIZE:
        out_im = pred[padding_h:, padding_w:]
    else:
        pred[pred == 0] = 1
        shadow_map = cv2.resize(im_org, (MAX_SIZE, MAX_SIZE)).astype(float) / pred.astype(float)
        shadow_map = cv2.resize(shadow_map, (w, h))
        shadow_map[shadow_map == 0] = 0.00001
        out_im = np.clip(im_org.astype(float) / shadow_map, 0, 255).astype(np.uint8)

    return out_im


def infer_folders(model, im_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for im_name in tqdm(os.listdir(im_folder)):
        im_path = os.path.join(im_folder, im_name)
        out_path = os.path.join(out_folder, im_name)
        out_im = drnet_infer_onnx(model, im_path)
        cv2.imwrite(out_path, out_im)

if __name__ == '__main__':
    session = load_drnet_model("/root/gdx/DocDewarp/checkpoints/baseline_drnet_V1.3/drnet_model_V1.3.onnx")
    # image_path = "/datassd/hz/ultralytics-main/tt/img/0A8EF8BA77B5B46C4C1A4150BA0EB102_segL.jpg"
    # output_image = drnet_infer_onnx(session, image_path)
    # cv2.imwrite("output_image.png", output_image)
    infer_folders(session, "/root/gdx/DocDewarp_1/img", "out1")
