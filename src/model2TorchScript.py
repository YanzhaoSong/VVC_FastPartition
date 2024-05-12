"""
2023.11.26 added by syz
功能：将训练获得的模型转换为 可以被C++加载的文件
"""
import os
from types import SimpleNamespace

import torch

from cnn_utils import create_model


def get_shape(shape: str):
    w, h = shape.split('x')
    return int(w), int(h)


def model2TorchScript(
        model_args: SimpleNamespace,
        qp: str,
        shape: str,
        save_dir: str,
):
    model_path = os.path.join(model_args.model_dir, f"QP{qp}_{shape}",
                              f"{model_args.model_name}_QP{qp}_{shape}.pth")

    # 构建模型
    model = create_model(model_args, from_timm=model_args.from_timm)
    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    w, h = get_shape(shape)
    input = torch.rand(1, 1, w, h)
    traced_script_module = torch.jit.trace(model, input)

    # 判断存储路径是否存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_save_path = os.path.join(save_dir, f"QP{qp}_{shape}.pt")
    traced_script_module.save(model_save_path)


if __name__ == "__main__":
    model_args = SimpleNamespace()
    model_args.model_dir = "log/0408/MyNet/2"
    model_args.model_name = "MyNet"
    model_args.in_channels = 1
    model_args.num_classes = 6
    model_args.from_timm = False

    QPs = ['37', '32', '27', '22']  # '37', '32', '27', '22'
    Shapes = ['32x32']  # , '32x16', '16x32', '16x16', '8x32', '32x8', '32x32'

    for QP in QPs:
        for Shape in Shapes:
            model2TorchScript(model_args, qp=QP, shape=Shape, save_dir='scripts/cnn_scripts/0408-MyNet-2')
