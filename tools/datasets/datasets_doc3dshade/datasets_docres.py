import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from configs.option import get_option
from .augments import get_transform
import os
import random
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt, prompt=True, train_transform=None, valid_transform=None):
        self.phase = phase
        self.data_path = opt.data_path
        self.augmentation = train_transform if "train" in phase else valid_transform
        self.image_size = opt.image_size 

        path = os.path.join(self.data_path, phase + ".txt")
        with open(path, "r") as f:
            self.image_list = [file_id.rstrip() for file_id in f.readlines()]
            self.bleed_list = [file_id for file_id in self.image_list if "_gt" not in file_id]

    def __getitem__(self, index):
        image_name = self.image_list[index]
        
        if '_gt' in image_name:
            label_path = os.path.join(self.data_path, "illumination", image_name)
            in_path = label_path.replace("_gt", "")
            cap_im = cv2.imread(in_path)
            label = cv2.imread(label_path)
            label,cap_im = self.randomcrop_realdataset(label,cap_im)
            cap_im = self.appearance_randomAugmentv1(cap_im)
            enhance_result = self.appearance_dtsprompt(cap_im)
        else:
            label_path = os.path.join(self.data_path, "label", image_name)
            bleed_path = os.path.join(self.data_path, "label",random.choice(self.bleed_list))
            shadow_path = os.path.join(self.data_path, "shadow_label",random.choice(self.bleed_list))
            shadow_im = self.load_image(shadow_path)
            label = self.load_image(label_path)
            bleed_im = self.load_image(bleed_path)

            bleed_im = cv2.resize(bleed_im,label.shape[:2][::-1])
            label = self.randomcrop([label])[0]
            bleed_im = self.randomcrop([bleed_im])[0]
            cap_im = self.bleed_trough(label,bleed_im)

            
            cap_im = self.appearance_randomAugmentv2(cap_im,shadow_im)
            enhance_result = self.appearance_dtsprompt(cap_im)

        label_2x, label_4x, label_8x = self.downsample_label(label)

        image = self.rgbim_transform(cap_im)

        label = self.rgbim_transform(label)
        label_2x = self.rgbim_transform(label_2x)
        label_4x = self.rgbim_transform(label_4x)
        label_8x = self.rgbim_transform(label_8x)

        dtsprompt = self.rgbim_transform(enhance_result)
        image = torch.cat((image,dtsprompt),0)
        

        return {"image": image, "label": label,  "label_2x": label_2x, "label_4x": label_4x, "label_8x": label_8x,
        # "shadow_label": shadow
        }

    def __len__(self):
        return len(self.image_list)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        return image
    def downsample_label(self, image):
        original_height, original_width = image.shape[:2]
    
        # 2倍下采样
        label_2x = cv2.resize(
            image, 
            (original_width // 2, original_height // 2), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # 4倍下采样
        label_4x = cv2.resize(
            image, 
            (original_width // 4, original_height // 4), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # 8倍下采样
        label_8x = cv2.resize(
            image, 
            (original_width // 8, original_height // 8), 
            interpolation=cv2.INTER_LINEAR
        )
        return label_2x, label_4x, label_8x
    
    def deshadow_dtsprompt(self,img):
        h,w = img.shape[:2]
        img = cv2.resize(img,(1024,1024))
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        bg_imgs = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            bg_imgs.append(bg_img)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        result_norm = cv2.merge(result_norm_planes)
        bg_imgs = cv2.merge(bg_imgs)
        bg_imgs = cv2.resize(bg_imgs,(w,h))
        return bg_imgs

    def appearance_dtsprompt(self,img):
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

    def randomcrop(self,im_list):
        im_num = len(im_list)
        ## random scale rotate
        if random.uniform(0,1) <= 0.8:
            y,x = im_list[0].shape[:2]
            angle = random.uniform(-180,180)
            scale = random.uniform(0.7,1.5)
            M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
            for i in range(im_num):
                im_list[i] = cv2.warpAffine(im_list[i],M,(x,y),borderValue=(255,255,255))

        ## random crop
        crop_size = self.image_size
        for i in range(im_num):
            h,w = im_list[i].shape[:2]
            h = max(h,crop_size)
            w = max(w,crop_size)
            im_list[i] = cv2.resize(im_list[i],(w,h))
        
        if h==crop_size:
            shift_y=0
        else:
            shift_y = np.random.randint(0,h-crop_size)
        if w==crop_size:
            shift_x=0
        else:
            shift_x = np.random.randint(0,w-crop_size)
        for i in range(im_num):
            im_list[i] = im_list[i][shift_y:shift_y+crop_size,shift_x:shift_x+crop_size,:]
        return im_list  

    def randomcrop_realdataset(self,gt_im,cap_im):
        if random.uniform(0,1) <= 0.5:
            y,x = gt_im.shape[:2]
            angle = random.uniform(-30,30)
            scale = random.uniform(0.8,1.5)
            M = cv2.getRotationMatrix2D((int(x/2),int(y/2)),angle,scale)
            gt_im = cv2.warpAffine(gt_im,M,(x,y),borderValue=(255,255,255))
            cap_im = cv2.warpAffine(cap_im,M,(x,y),borderValue=(255,255,255))
        crop_size = self.image_size
        if gt_im.shape[0] <= crop_size:
            gt_im = cv2.copyMakeBorder(gt_im,crop_size-gt_im.shape[0]+1,0,0,0,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
            cap_im = cv2.copyMakeBorder(cap_im,crop_size-cap_im.shape[0]+1,0,0,0,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        if gt_im.shape[1] <= crop_size:
            gt_im = cv2.copyMakeBorder(gt_im,0,0,crop_size-gt_im.shape[1]+1,0,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
            cap_im = cv2.copyMakeBorder(cap_im,0,0,crop_size-cap_im.shape[1]+1,0,borderType=cv2.BORDER_CONSTANT,value=(255,255,255))
        shift_y = np.random.randint(0,gt_im.shape[1]-crop_size)
        shift_x = np.random.randint(0,gt_im.shape[0]-crop_size)
        gt_im = gt_im[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        cap_im = cap_im[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        return gt_im,cap_im

    def rgbim_transform(self,im):
        im = im.astype("float32")/255. 
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im)
        return im

    def bleed_trough(self, in_im, bleed_im):
        if random.uniform(0,1) <= 0.5:
            if random.uniform(0,1) <= 0.8:
                ksize = np.random.randint(1,2)*2 + 1
                bleed_im = cv2.blur(bleed_im,(ksize,ksize))
            bleed_im = cv2.flip(bleed_im,1)
            alpha = random.uniform(0.75,1)
            in_im = cv2.addWeighted(in_im,alpha,bleed_im,1-alpha,0)
        return in_im

    def appearance_randomAugmentv1(self,in_img):

        ## brightness
        if random.uniform(0,1) <= 0.8:
            high = 1.3
            low = 0.5
            ratio = np.random.uniform(low,high)
            in_img = in_img.astype(np.float64)*ratio
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        if random.uniform(0,1) <= 0.8:
            high = 1.3
            low = 0.5
            ratio = np.random.uniform(low,high)
            gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            mean_array = np.ones_like(in_img).astype(np.float64)*mean
            in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## color
        if random.uniform(0,1) <= 0.8:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(self.image_size,self.image_size,1))
            in_img = in_img.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)

        if random.uniform(0, 1) <= 0.3:
            # 随机选择一个内核大小，确保为奇数
            kernel_size = random.choice([5, 7, 9])
            in_img = cv2.GaussianBlur(in_img, (kernel_size, kernel_size), 0)

        # 噪声增强
        if random.uniform(0, 1) <= 0.3:
            # 添加高斯噪声
            mean = 0
            var = random.randint(50, 150)
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, in_img.shape).astype(np.float32)
            in_img = in_img.astype(np.float32) + gauss
            in_img = np.clip(in_img, 0, 255).astype(np.uint8)     

        return in_img

    def appearance_randomAugmentv2(self,in_img,shadow_img):
        h,w = in_img.shape[:2]
        # random crop
        crop_size = random.randint(96,1024)
        if shadow_img.shape[0] <= crop_size:
            shadow_img = cv2.resize(shadow_img,(crop_size+1,crop_size+1))
        if shadow_img.shape[1] <= crop_size:
            shadow_img = cv2.resize(shadow_img,(crop_size+1,crop_size+1))

        shift_y = np.random.randint(0,shadow_img.shape[1]-crop_size)
        shift_x = np.random.randint(0,shadow_img.shape[0]-crop_size)
        shadow_img = shadow_img[shift_x:shift_x+crop_size,shift_y:shift_y+crop_size,:]
        shadow_img = cv2.resize(shadow_img,(w,h))
        in_img = in_img.astype(np.float64)*(shadow_img.astype(np.float64)+1)/255
        in_img = np.clip(in_img,0,255).astype(np.uint8)

        ## brightness
        if random.uniform(0,1) <= 0.8:
            high = 1.3
            low = 0.5
            ratio = np.random.uniform(low,high)
            in_img = in_img.astype(np.float64)*ratio
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## contrast
        if random.uniform(0,1) <= 0.8:
            high = 1.3
            low = 0.5
            ratio = np.random.uniform(low,high)
            gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            mean_array = np.ones_like(in_img).astype(np.float64)*mean
            in_img = in_img.astype(np.float64)*ratio + mean_array*(1-ratio)
            in_img = np.clip(in_img,0,255).astype(np.uint8)
        ## color
        if random.uniform(0,1) <= 0.8:
            high = 0.2
            low = 0.1
            ratio = np.random.uniform(0.1,0.3)
            random_color = np.random.randint(50,200,3).reshape(1,1,3)
            random_color = (random_color*ratio).astype(np.uint8)
            random_color = np.tile(random_color,(h,w,1))
            in_img = in_img.astype(np.float64)*(1-ratio) + random_color
            in_img = np.clip(in_img,0,255).astype(np.uint8)        

        if random.uniform(0,1) <= 0.8:
            quanlity_high = 95
            quanlity_low = 45
            quanlity = int(np.random.randint(quanlity_low,quanlity_high))
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),quanlity]
            result, encimg = cv2.imencode('.jpg',in_img,encode_param)
            in_img = cv2.imdecode(encimg,1).astype(np.uint8)
        
        if random.uniform(0, 1) <= 0.3:
            # 随机选择一个内核大小，确保为奇数
            kernel_size = random.choice([5, 7, 9])
            in_img = cv2.GaussianBlur(in_img, (kernel_size, kernel_size), 0)

        # 噪声增强
        if random.uniform(0, 1) <= 0.3:
            # 添加高斯噪声
            mean = 0
            var = random.randint(50, 150)
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, in_img.shape).astype(np.float32)
            in_img = in_img.astype(np.float32) + gauss
            in_img = np.clip(in_img, 0, 255).astype(np.uint8)

        return in_img


def get_dataloader(opt, prompt):
    train_transform, valid_transform = get_transform(opt)
    train_dataset = Dataset(
        phase="train_part",
        opt=opt,
        prompt=prompt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    valid_dataset = Dataset(
        phase="val_part",
        opt=opt,
        prompt=prompt,
        train_transform=train_transform,
        valid_transform=valid_transform,
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
    opt = get_option()
    train_dataloader, valid_dataloader = get_dataloader(opt)

    for i, batch in enumerate(train_dataloader):
        print(batch["image"].shape, torch.unique(batch["label"]))
        break
