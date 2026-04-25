import os
import glob
from typing import List, Optional, Iterator, Dict, Any

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)


class ImageFolderCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        image_dir: str,
        model_path: str,
        input_name: Optional[str] = None,
        image_size: int = 640,
        batch_size: int = 1,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            image_dir: 校准图片文件夹
            model_path: FP32 ONNX 模型路径，用于自动读取输入名
            input_name: 模型输入名；如果不传，则从 ONNX 模型中自动读取第一个输入名
            image_size: resize 到 image_size x image_size
            batch_size: 校准批大小；通常设为 1 最稳
            max_samples: 最多使用多少张图做校准；None 表示全部
        """
        self.image_dir = image_dir
        self.model_path = model_path
        self.image_size = image_size
        self.batch_size = batch_size

        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32
        ).reshape(1, 1, 3)
        self.std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32
        ).reshape(1, 1, 3)

        self.input_name = input_name or self._get_model_input_name(model_path)
        self.image_paths = self._collect_images(image_dir)

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        if len(self.image_paths) == 0:
            raise ValueError(f"在目录中未找到图片: {image_dir}")

        self._data_iter: Optional[Iterator[Dict[str, Any]]] = None
        self.rewind()

    def _get_model_input_name(self, model_path: str) -> str:
        model = onnx.load(model_path)
        if len(model.graph.input) == 0:
            raise ValueError(f"模型没有输入: {model_path}")
        return model.graph.input[0].name

    def _collect_images(self, image_dir: str) -> List[str]:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        image_paths: List[str] = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        image_paths.sort()
        return image_paths

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        # 读取 BGR 图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"读取图片失败: {image_path}")

        # resize 到 640x640
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # 归一化，仍保持 HWC/BGR
        img = self._normalize(img)

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # 增加 batch 维度，变成 1xCxHxW
        img = np.expand_dims(img, axis=0)

        # ONNX Runtime 校准通常仍喂 float32
        img = img.astype(np.float32)
        return img

    def _build_data(self) -> Iterator[Dict[str, Any]]:
        batch: List[np.ndarray] = []

        for image_path in self.image_paths:
            tensor = self._preprocess_image(image_path)
            batch.append(tensor)

            if len(batch) == self.batch_size:
                if self.batch_size == 1:
                    yield {"image": batch[0], "scale_factor": np.array([[1.0, 1.0]], dtype=np.float32)}
                else:
                    batch_tensor = np.concatenate(batch, axis=0)
                    yield {"image": batch_tensor, "scale_factor": np.array([[1.0, 1.0]], dtype=np.float32)}
                batch = []

        if len(batch) > 0:
            if self.batch_size == 1:
                yield {"image": batch[0], "scale_factor": np.array([[1.0, 1.0]], dtype=np.float32)}
            else:
                batch_tensor = np.concatenate(batch, axis=0)
                yield {"image": batch_tensor, "scale_factor": np.array([[1.0, 1.0]], dtype=np.float32)}

    def get_next(self) -> Optional[Dict[str, Any]]:
        return next(self._data_iter, None)

    def rewind(self):
        self._data_iter = iter(self._build_data())


if __name__ == "__main__":
    model_fp32 = "model_conv_bias_to_add.onnx"
    model_int8 = "picodet_m_416_coco_qat_sim_all.onnx"
    calib_image_dir = "./dataset/isdd-dataset-voc/images"

    reader = ImageFolderCalibrationDataReader(
        image_dir=calib_image_dir,
        model_path=model_fp32,
        input_name=None,     # 不写则自动取模型第一个输入名
        image_size=416,
        batch_size=1,        # 建议先用 1
        max_samples=50,     # 可按需要调整
    )

    quantize_static(
        model_input=model_fp32,
        model_output=model_int8,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],  # 只量化卷积层
        nodes_to_quantize =["p2o.Conv.94_nobias", "p2o.Conv.108_nobias", "p2o.Conv.80_nobias" ,
                            "p2o.Conv.66_nobias","p2o.Conv.111_nobias", "p2o.Conv.97_nobias",
                            "p2o.Conv.83_nobias", "p2o.Conv.69_nobias", "p2o.Conv.110_nobias",
                            "p2o.Conv.96_nobias", "p2o.Conv.82_nobias", "p2o.Conv.68_nobias"],
        per_channel=True,
        
    )

    print(f"量化完成，输出文件: {model_int8}")
    