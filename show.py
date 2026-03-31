#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import onnx
import numpy as np
from onnx import numpy_helper


def sanitize_name(name: str) -> str:
    if not name:
        return "unnamed"
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name


def load_initializer_map(model: onnx.ModelProto):
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)
    return init_map


def get_node_name(node, index: int) -> str:
    if node.name:
        return node.name
    if len(node.output) > 0 and node.output[0]:
        return node.output[0]
    return f"Conv_{index}"


def get_attribute_dict(node):
    """
    提取 Conv 节点属性，转成普通 Python dict
    """
    attr_dict = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INT:
            attr_dict[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.INTS:
            attr_dict[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOAT:
            attr_dict[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.FLOATS:
            attr_dict[attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            attr_dict[attr.name] = attr.s.decode("utf-8", errors="ignore")
        elif attr.type == onnx.AttributeProto.STRINGS:
            attr_dict[attr.name] = [
                s.decode("utf-8", errors="ignore") for s in attr.strings
            ]
        else:
            attr_dict[attr.name] = "UNSUPPORTED"
    return attr_dict


def write_array_txt(file_path: Path, arr: np.ndarray, title: str):
    flat = arr.reshape(-1)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"shape: {list(arr.shape)}\n")
        f.write("data: ")
        f.write(" ".join(f"{x:.8g}" for x in flat))
        f.write("\n")


def extract_conv_params_to_txt(onnx_path: str, output_dir: str):
    model = onnx.load(onnx_path)
    init_map = load_initializer_map(model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_txt_path = output_dir / "conv_summary.txt"
    conv_count = 0

    with open(summary_txt_path, "w", encoding="utf-8") as summary_f:
        for idx, node in enumerate(model.graph.node):
            if node.op_type != "Conv":
                continue

            conv_count += 1
            conv_name = get_node_name(node, conv_count)
            conv_name_safe = sanitize_name(conv_name)

            weight_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None

            weight = init_map.get(weight_name, None)
            bias = init_map.get(bias_name, None) if bias_name else None

            weight_found = weight is not None
            bias_found = bias is not None
            weight_shape = list(weight.shape) if weight is not None else None
            bias_shape = list(bias.shape) if bias is not None else None

            attr_dict = get_attribute_dict(node)

            # 给常见 Conv 参数补默认值，便于查看
            kernel_shape = attr_dict.get("kernel_shape", None)
            strides = attr_dict.get("strides", [1, 1])
            pads = attr_dict.get("pads", [0, 0, 0, 0])
            dilations = attr_dict.get("dilations", [1, 1])
            group = attr_dict.get("group", 1)
            auto_pad = attr_dict.get("auto_pad", "NOTSET")

            # 每个卷积层一行，不包含任何 name 字段
            summary_line = (
                f"conv_index={conv_count}; "
                f"weight_found={weight_found}; "
                f"bias_found={bias_found}; "
                f"weight_shape={weight_shape}; "
                f"bias_shape={bias_shape}; "
                f"kernel_shape={kernel_shape}; "
                f"strides={strides}; "
                f"pads={pads}; "
                f"dilations={dilations}; "
                f"group={group}; "
                f"auto_pad={auto_pad}"
            )
            summary_f.write(summary_line + "\n")

            if weight is None:
                print(f"[WARN] 第 {conv_count} 个 Conv 的 weight 未在 initializer 中找到，已跳过参数导出。")
                continue

            weight_txt = output_dir / f"{conv_count:03d}_{conv_name_safe}_weight.txt"
            write_array_txt(weight_txt, weight, f"conv_{conv_count} / weight")
            print(f"[OK] 导出: {weight_txt}")

            if bias is not None:
                bias_txt = output_dir / f"{conv_count:03d}_{conv_name_safe}_bias.txt"
                write_array_txt(bias_txt, bias, f"conv_{conv_count} / bias")
                print(f"[OK] 导出: {bias_txt}")

    print(f"\n总共找到 {conv_count} 个 Conv 节点")
    print(f"摘要已保存到: {summary_txt_path}")


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX 模型中每个 Conv 层的参数为 TXT")
    parser.add_argument("onnx_model", type=str, help="输入 ONNX 模型路径")
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="conv_params_txt",
        help="输出目录，默认: conv_params_txt"
    )
    args = parser.parse_args()

    extract_conv_params_to_txt(args.onnx_model, args.output_dir)


if __name__ == "__main__":
    main()