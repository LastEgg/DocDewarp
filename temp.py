import cv2
import numpy as np
def compare(img1, img2):
    # 确保三张图片具有相同的高度
    height = max(img1.shape[0], img2.shape[0])

    # 如果图片高度不同，则调整到相同大小
    if img1.shape[0] != height:
        scale_factor = height / img1.shape[0]
        width = int(img1.shape[1] * scale_factor)
        img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)

    if img2.shape[0] != height:
        scale_factor = height / img2.shape[0]
        width = int(img2.shape[1] * scale_factor)
        img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)


    # 创建垂直的黑色分割线，宽度为10个像素
    separator = np.zeros((height, 10, 3), dtype=np.uint8)
    separator[:] = [0, 0, 0]  # 黑色，或者你可以使用其他颜色

    # 水平拼接五部分：img1、分隔线、img2、分隔线、img3
    combined_image = np.hstack((img1, separator, img2))

    return combined_image

name = "1D936B2716D3A0E60289A8D63BFF4569_segL.jpg"
img1_path = f"/root/gdx/DocDewarp_1/test_dir/img/{name}"
img2_path = f"/root/gdx/DocDewarp_1/test_dir/out_drnet/{name}"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
out = compare(img1, img2)
cv2.imwrite("/root/gdx/DocDewarp_1/test_dir/vis/4.jpg", out)