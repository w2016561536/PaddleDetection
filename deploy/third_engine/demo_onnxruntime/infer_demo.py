# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import argparse
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2


def xywh_to_xyxy(boxes):
    """
    boxes: (N, 4), [l, u, r, d]
    return: (N, 4), [x1, y1, x2, y2]
    """
    boxes = boxes.astype(np.float32).copy()
    l, u, r, d = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = l
    y1 = u
    x2 =  r
    y2 = d
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_xywh_for_opencv(boxes):
    """
    OpenCV NMSBoxes 需要 [x, y, w, h]
    输入:
        boxes: (N, 4), [x1, y1, x2, y2]
    返回:
        list of [x, y, w, h]
    """
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        result.append([float(x1), float(y1), float(w), float(h)])
    return result


def multiclass_nms_opencv(out0, out1, conf_thresh=0.25, nms_thresh=0.5):
    """
    out0: (1, 3598, 4), xywh
    out1: (1, 5, 3598), class scores

    返回:
        final_boxes: (M, 4), xyxy
        final_scores: (M,)
        final_classes: (M,)
    """
    boxes = out0[0]          # (3598, 4)
    scores = out1[0].T       # (3598, 5)

    boxes_xyxy = xywh_to_xyxy(boxes)

    final_boxes = []
    final_scores = []
    final_classes = []

    num_classes = scores.shape[1]

    for cls_id in range(num_classes):
        cls_scores = scores[:, cls_id]
        mask = cls_scores > conf_thresh

        if not np.any(mask):
            continue

        cls_boxes_xyxy = boxes_xyxy[mask]
        cls_scores_keep = cls_scores[mask]

        # OpenCV NMSBoxes 要求 boxes 是 [x, y, w, h]
        cls_boxes_xywh = xyxy_to_xywh_for_opencv(cls_boxes_xyxy)

        indices = cv2.dnn.NMSBoxes(
            bboxes=cls_boxes_xywh,
            scores=cls_scores_keep.tolist(),
            score_threshold=conf_thresh,
            nms_threshold=nms_thresh
        )

        if indices is None or len(indices) == 0:
            continue

        # 兼容不同 OpenCV 版本返回格式
        indices = np.array(indices).reshape(-1)

        final_boxes.append(cls_boxes_xyxy[indices])
        final_scores.append(cls_scores_keep[indices])
        final_classes.append(np.full(len(indices), cls_id, dtype=np.int32))

    if len(final_boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    final_boxes = np.concatenate(final_boxes, axis=0).astype(np.float32)
    final_scores = np.concatenate(final_scores, axis=0).astype(np.float32)
    final_classes = np.concatenate(final_classes, axis=0).astype(np.int32)

    # 最终按分数从高到低排序
    order = np.argsort(-final_scores)
    final_boxes = final_boxes[order]
    final_scores = final_scores[order]
    final_classes = final_classes[order]

    return final_boxes, final_scores, final_classes


def draw_detections(image, boxes, scores, classes, class_names=None):
    """
    image: OpenCV 读取的 BGR 图像
    boxes: (N, 4), xyxy
    """
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if class_names is None:
        num_classes = int(classes.max()) + 1 if len(classes) > 0 else 0
        class_names = [f"class_{i}" for i in range(num_classes)]

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    h, w = img.shape[:2]

    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(np.int32)
        x1 = x1 / 416 * w
        y1 = y1 / 416 * h
        x2 = x2 / 416 * w
        y2 = y2 / 416 * h

        x1 = max(0, min(x1, w - 1)).astype(np.int32)
        y1 = max(0, min(y1, h - 1)).astype(np.int32)
        x2 = max(0, min(x2, w - 1)).astype(np.int32)
        y2 = max(0, min(y2, h - 1)).astype(np.int32)
        print(f"box: {x1}, {y1}, {x2}, {y2}, score: {score:.4f}, class: {class_names[int(cls_id)]}")

        color = colors[int(cls_id) % len(colors)]
        label = f"{class_names[int(cls_id)]}: {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_top = max(0, y1 - th - 6)

        cv2.rectangle(img, (x1, text_top), (x1 + tw, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return img



class PicoDet():
    def __init__(self,
                 model_pb_path,
                 label_path,
                 prob_threshold=0.4,
                 iou_threshold=0.3):
        self.classes = list(
            map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
        inputs_name = [a.name for a in self.net.get_inputs()]
        inputs_shape = {
            k: v.shape
            for k, v in zip(inputs_name, self.net.get_inputs())
        }
        self.input_shape = inputs_shape['image'][2:]

    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=False):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        origin_shape = srcimg.shape[:2]
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        img_shape = np.array([
            [float(self.input_shape[0]), float(self.input_shape[1])]
        ]).astype('float32')
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')

        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] /
                                                      hw_scale)
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    self.input_shape[1] - neww - left,
                    cv2.BORDER_CONSTANT,
                    value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] *
                                 hw_scale), self.input_shape[1]
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    self.input_shape[0] - newh - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0)
        else:
            img = cv2.resize(
                srcimg, self.input_shape, interpolation=cv2.INTER_LINEAR)

        return img, img_shape, scale_factor

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        return color_map

    def detect(self, srcimg):
        img, im_shape, scale_factor = self.resize_image(srcimg)
        img = self._normalize(img)

        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        inputs_dict = {
            'im_shape': im_shape,
            'image': blob,
            'scale_factor': scale_factor
        }
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}

        outs = self.net.run(None, net_inputs)

        out0 = np.array(outs[0])
        out1 = np.array(outs[1])

        # image = cv2.imread("results/700zhengchang.jpg")
        # image = cv2.resize(image, (416, 416))
        class_names = ["ca_shang", "zang_wu", "zhe_zhou", "zhen_kong", "zheng_chang"]

        boxes, scores, classes = multiclass_nms_opencv(
            out0,
            out1,
            conf_thresh=0.4,
            nms_thresh=0.5
        )

        result = draw_detections(
            image=srcimg,
            boxes=boxes,
            scores=scores,
            classes=classes,
            class_names=class_names
        )

        # cv2.imwrite("result.jpg", result)
        print("检测框数量:", len(boxes))
        

        return result

    def detect_folder(self, img_fold, result_path):
        img_fold = Path(img_fold)
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        img_name_list = filter(
            lambda x: str(x).endswith(".png") or str(x).endswith(".jpg"),
            img_fold.iterdir(), )
        img_name_list = list(img_name_list)
        print(f"find {len(img_name_list)} images")

        for img_path in tqdm(img_name_list):
            img = cv2.imread(str(img_path), 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            srcimg = net.detect(img)
            save_path = str(result_path / img_path.name.replace(".png", ".jpg"))
            cv2.imwrite(save_path, srcimg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modelpath',
        type=str,
        default='onnx_file/picodet_s_320_lcnet_postprocessed.onnx',
        help="onnx filepath")
    parser.add_argument(
        '--classfile',
        type=str,
        default='coco_label.txt',
        help="classname filepath")
    parser.add_argument(
        '--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument(
        '--nmsThreshold', default=0.6, type=float, help='nms iou thresh')
    parser.add_argument(
        "--img_fold", dest="img_fold", type=str, default="./imgs")
    parser.add_argument(
        "--result_fold", dest="result_fold", type=str, default="results")
    args = parser.parse_args()

    net = PicoDet(
        args.modelpath,
        args.classfile,
        prob_threshold=args.confThreshold,
        iou_threshold=args.nmsThreshold)

    net.detect_folder(args.img_fold, args.result_fold)
    print(
        f'infer results in ./deploy/third_engine/demo_onnxruntime/{args.result_fold}'
    )
