import torch
import cv2
import numpy as np
from models import UNext_full_resolution_padding_L_py_L

def drnet_load(model_path="./checkpoints/baseline_drnet_V1.2/epoch_97-loss_0.074.ckpt"):
    model = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512).cuda()
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k.replace("model.", "")] = v
    model.load_state_dict(update_dict)
    model.eval()
    return model

def export_model_to_onnx(model, onnx_file_path):
    # 使用动态输入创建一个虚拟输入张量。假设输入有6个通道。
    dummy_input = torch.randn(1, 6, 224, 224)  # 将宽高设置为任意值，这是用于示例的张量
    model.to('cpu')  # 将模型转移到 CPU 上以导出 ONNX

    # 导出为 ONNX 时候，将宽度和高度设置为动态
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: 'height', 3: 'width'},  # 高度和宽度是动态的
            'output': {2: 'height', 3: 'width'}  # 输出同样具有动态的高度和宽度
        },
        opset_version=11
    )
    print(f"Model has been exported to {onnx_file_path}")


# 加载模型
model = drnet_load("/root/gdx/DocDewarp/checkpoints/baseline_drnet_V1.3/epoch_95-loss_0.090.ckpt")

# 导出到 ONNX
export_model_to_onnx(model, "/root/gdx/DocDewarp/checkpoints/baseline_drnet_V1.3/drnet_model_V1.3.onnx")
