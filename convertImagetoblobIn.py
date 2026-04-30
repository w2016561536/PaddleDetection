import cv2
import numpy as np


def image_to_npy(
    image_path,
    output_npy_path,
    input_mean=(0.485, 0.456, 0.406),
    input_std=(0.229, 0.224, 0.225),
):
    """
    将图像转换为 1x3x416x416 的 float32 NPY 数据

    处理流程与如下 OpenCV C++ 代码一致：

        resize -> BGR2RGB -> float32/255
        -> (x - mean) / std
        -> HWC 转 CHW
        -> 增加 batch 维度

    参数:
        image_path: 输入图像路径
        output_npy_path: 输出 npy 路径
        input_mean: 归一化均值
        input_std: 归一化标准差
    """

    # 读取图像 (BGR)
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # resize 到 416x416
    resized_image = cv2.resize(
        input_image,
        (416, 416),
        interpolation=cv2.INTER_CUBIC
    )

    # BGR -> RGB
    rgb_image = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2RGB
    )

    # 转 float32 并归一化到 [0, 1]
    rgb_float = rgb_image.astype(np.float32) / 255.0

    # mean/std
    mean = np.array(input_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(input_std, dtype=np.float32).reshape(1, 1, 3)

    normalized = (rgb_float - mean) / std

    # HWC -> CHW
    chw = np.transpose(normalized, (2, 0, 1))

    # 增加 batch 维度: 1x3x416x416
    nchw = np.expand_dims(chw, axis=0).astype(np.float32)

    print("输出 shape:", nchw.shape)
    print("输出 dtype:", nchw.dtype)

    # 保存 npy
    np.save(output_npy_path, nchw)

    print(f"已保存到: {output_npy_path}")


if __name__ == "__main__":
    image_to_npy(
        image_path="dataset/isdd-dataset-voc/images/ca_shang20.jpg",
        output_npy_path="raw_pic_blob.npy",
        input_mean= (123.675/255.0, 116.28/255.0, 103.53/255.0),
        input_std= (58.395/255.0, 57.12/255.0, 57.375/255.0),
    )
