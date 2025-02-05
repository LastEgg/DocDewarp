import torch
import numpy as np
import hdf5storage as h5
from concurrent.futures import ThreadPoolExecutor
from configs.option import get_option
from .augments import get_transform
import os
import random
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt, train_transform=None, valid_transform=None):
        self.phase = phase
        self.data_path = opt.data_path
        self.augmentation = train_transform if "train" in phase else valid_transform
        self.image_size = (
            opt.image_size if isinstance(opt.image_size, tuple) else (opt.image_size, opt.image_size)
        )

        path = os.path.join(self.data_path, phase + ".txt")
        with open(path, "r") as f:
            self.image_list = [file_id.rstrip() for file_id in f.readlines()]

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_path = os.path.join(self.data_path, "img_bld" + image_name + ".png")
        image = self.load_image(image_path)
        wc_path = os.path.join(self.data_path, "wc_bld" + image_name + ".exr")
        if not os.path.exists(wc_path):
            wc = None
        else:
            wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if os.path.exists(os.path.join(self.data_path, "bm_bld" + image_name + ".mat")):
            bm_path = os.path.join(self.data_path, "bm_bld" + image_name + ".mat")
            bm = h5.loadmat(bm_path)["bm"]
        else:
            bm_path = os.path.join(self.data_path, "bm_bld" + image_name + ".npy")
            bm = np.load(bm_path).transpose(2,1,0)

        image, bm = self.transform(wc, bm, image)
        return {"image": image, "label": bm}

    def __len__(self):
        return len(self.image_list)

    def load_image(self, path):
        image = cv2.imread(path)
        return image

    def load_images_in_parallel(self):
        with ThreadPoolExecutor(max_workers=24) as executor:
            pass
    
    def random_crop(self, wc: np.ndarray):
        # 以一定概率不裁剪
        if random.uniform(0, 1) < 0.15:
            return wc, 0, 0, 0, 0

        # 裁页面
        mask = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(np.uint8)
        mask_size = mask.shape
        [y, x] = mask.nonzero()
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        wc = wc[min_y: max_y + 1, min_x: max_x + 1, :]

        # 页面向外保留的背景
        s = 0
        # wc = np.pad(wc, ((s, s), (s, s), (0, 0)), "constant")

        r = random.uniform(0, 1)
        # 保留部分边
        if r < 0.30:
            # print("保留部分边")
            cx1, cx2, cy1, cy2 = self.random_cutside(wc, 40, 140, 80)

            # 0表示不留边，1表示留边
            t = random.uniform(0, 1)
            b = random.uniform(0, 1)
            l = random.uniform(0, 1)
            r = random.uniform(0, 1)

            top: int = min_y - s + cy1
            bottom: int = mask_size[0] - max_y - s + cy2
            left: int = min_x - s + cx1
            right: int = mask_size[1] - max_x - s + cx2

            # 以一定概率不裁剪
            if t < 0.2:
                top: int = min_y - s
            if b < 0.2:
                bottom: int = mask_size[0] - max_y - s
            if l < 0.5:
                left: int = min_x - s
            if r < 0.5:
                right: int = mask_size[1] - max_x - s
        # 不留四边
        elif r < 0.45:

            cx1, cx2, cy1, cy2 = self.random_cutside(wc, 40, 130, 80)
            # print("不留四边")
            top: int = min_y - s + cy1
            bottom: int = mask_size[0] - max_y - s + cy2
            left: int = min_x - s + cx1
            right: int = mask_size[1] - max_x - s + cx2
        # 裁长条或竖条
        elif r < 0.6:
            # print("长条竖条")

            # 短边长度80-100, 长边比例2到4倍
            short = random.randint(30, 40)
            long = short * random.uniform(2, 4)

            cx1, cx2, cy1, cy2 = self.random_cutbylen(wc, short, int(long), 20)
            top: int = min_y - s + cy1
            bottom: int = mask_size[0] - max_y - s + cy2
            left: int = min_x - s + cx1
            right: int = mask_size[1] - max_x - s + cx2
        # 不裁剪
        else:
            # print("不裁剪")
            top: int = min_y - s
            bottom: int = mask_size[0] - max_y - s
            left: int = min_x - s
            right: int = mask_size[1] - max_x - s

        # wc = wc[cy1:-cy2, cx1:-cx2, :]
        top = np.clip(top, 0, int(mask_size[0] / 1.8))
        bottom = np.clip(bottom, 1, int(mask_size[0] / 1.8) - 1)
        left = np.clip(left, 0, int(mask_size[1] / 1.8))
        right = np.clip(right, 1, int(mask_size[1]/ 1.8) - 1)

        return wc, top, bottom, left, right

    def random_cutside(self, wc: np.ndarray, minc, maxc, LEFT=80):
        # 至少留下80 * 80
        L = LEFT
        LH = wc.shape[0]
        LW = wc.shape[1]

        cx1 = random.randint(minc, maxc)
        cx1 = min(cx1, LW - L)
        cx2 = random.randint(minc, maxc) + 1
        cx2 = min(cx2, LW - cx1 - L)
        cy1 = random.randint(minc, maxc)
        cy1 = min(cy1, LH - L)
        cy2 = random.randint(minc, maxc) + 1
        cy2 = min(cy2, LH - cy1 - L)
        return cx1, cx2, cy1, cy2

    def random_cutbylen(self, wc: np.ndarray, short, long, LEFT=80):
        # 至少留下80 * 80
        L = LEFT
        LH = wc.shape[0]
        LW = wc.shape[1]

        # 裁长条
        if random.uniform(0, 1) < 0.8:
            # print("长条")
            cy1 = random.randint(10, LH - short)
            cy1 = min(cy1, LH - short)
            cy2 = LH - cy1 - short
            cy2 = min(cy2, LH - cy1 - short)

            cx1 = random.randint(20, 60)
            cx1 = min(cx1, LW - L)
            cx2 = random.randint(20, 60)
            cx2 = min(cx2, LW - cx1 - L)

        else:
            # print("竖条")
            cx1 = random.randint(10, LW - short)
            cx1 = min(cx1, LW - short)
            cx2 = LW - cx1 - short
            cx2 = min(cx2, LW - cx1 - short)

            cy1 = random.randint(20, 60)
            cy1 = min(cy1, LH - L)
            cy2 = random.randint(20, 60)
            cy2 = min(cy2, LH - cy1 - L)
        return cx1, cx2, cy1, cy2

    def transform(self, wc, bm, img):
        if wc is None:
            top, bottom, left, right = 0, 0, 0, 0
        else:
            _, top, bottom, left, right = self.random_crop(wc)
        # top, bottom, left, right = 0, 0, 0, 0

        if top==0 and bottom==0 and left==0 and right==0:
            img_cut = img
        else:
            img_cut = img[top:-bottom, left:-right, :]

        img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)
        img_cut = self.augmentation(image=img_cut)["image"]

        # 随机黑角
        h, w = img.shape[:2]
        mask = np.zeros_like(img)
        BLACK = False
        # 左上角 0.25
        if random.random() < 0.4:
            BLACK = True
            black_h = random.randint(h // 10, h // 2)
            black_w = random.randint(w // 10, w)
            if random.random() < 0.4:
                black_h += h // 2

            pts = np.array([[0, 0], [black_w, 0], [0, black_h]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))
        # 右上角
        if random.random() < 0.4:
            BLACK = True
            black_h = random.randint(h // 10, h // 2)
            black_w = random.randint(w // 10, w)
            if random.random() < 0.4:
                black_h += h // 2

            pts = np.array([[w, 0], [w - black_w, 0], [w, black_h]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))
        # 上边
        if BLACK and random.random() < 0.4:
            black_hl = random.randint(1, h // 10)
            black_hr = random.randint(1, h // 10)
            pts = np.array([[0, 0], [w, 0], [w, black_hr], [0, black_hl]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))
        
        img[mask == 255] = 0

        # resize image
        img = cv2.resize(img_cut, self.image_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)


        # resize bm
        bm = bm.astype(np.float32)
        bm[:, :, 1] = bm[:, :, 1] - top
        bm[:, :, 0] = bm[:, :, 0] - left

        imgcut_shape = img_cut.shape
        # 裁剪bm
        mask = ((bm[:, :, 0] >= 0)
                & (bm[:, :, 1] >= 0)
                & (bm[:, :, 0] < imgcut_shape[1])
                & (bm[:, :, 1] < imgcut_shape[0])).astype(np.uint8)

        [y, x] = mask.nonzero()
        
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        bm = bm[min_y: max_y + 1, min_x: max_x + 1, :]

        # 缩放
        bm = bm / np.array([(448.0 - left - right) / self.image_size[1], (448.0 - top - bottom) / self.image_size[0]])

        bm0 = cv2.resize(bm[:, :, 0], (self.image_size[0], self.image_size[1]))
        bm1 = cv2.resize(bm[:, :, 1], (self.image_size[0], self.image_size[1]))
        bm = np.stack([bm0, bm1], axis=-1)
    
        img = torch.tensor(img, dtype=torch.float32)
        bm = torch.tensor(bm, dtype=torch.float32)
        return img, bm



def get_dataloader(opt):
    train_transform, valid_transform = get_transform(opt)
    train_dataset = Dataset(
        phase="train_gdx",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    valid_dataset = Dataset(
        phase="val_gdx",
        opt=opt,
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
