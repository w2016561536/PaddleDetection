#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a .npy tensor to C/C++ array files.

Default output:
  input_data.h
  input_data.cc

The generated symbols are:
  INPUT_NDIM
  INPUT_SHAPE
  INPUT_SIZE
  INPUT_DATA

Example:
  python3 npy_to_c.py input.npy --name INPUT --out-dir .
"""

import argparse
import math
from pathlib import Path

import numpy as np


DTYPE_TO_C = {
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
    np.dtype("int8"): "int8_t",
    np.dtype("uint8"): "uint8_t",
    np.dtype("int16"): "int16_t",
    np.dtype("uint16"): "uint16_t",
    np.dtype("int32"): "int32_t",
    np.dtype("uint32"): "uint32_t",
    np.dtype("int64"): "int64_t",
    np.dtype("uint64"): "uint64_t",
}


def format_scalar(x, dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.floating):
        v = float(x)
        if math.isnan(v):
            return "NAN"
        if math.isinf(v):
            return "INFINITY" if v > 0 else "-INFINITY"
        # max precision needed for round-tripping float32/float64 text.
        if dtype == np.dtype("float32"):
            return format(v, ".9g")
        return format(v, ".17g")
    return str(int(x))


def write_array_values(f, arr: np.ndarray, ctype: str, values_per_line: int = 12) -> None:
    flat = arr.reshape(-1)
    dtype = arr.dtype
    for i in range(0, flat.size, values_per_line):
        line = ", ".join(format_scalar(v, dtype) for v in flat[i:i + values_per_line])
        if i + values_per_line < flat.size:
            f.write(f"  {line},\n")
        else:
            f.write(f"  {line}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npy", help="input .npy file")
    parser.add_argument("--name", default="INPUT",
                        help="C symbol prefix, e.g. INPUT -> INPUT_DATA")
    parser.add_argument("--out-dir", default=".",
                        help="directory for generated .h/.cc files")
    parser.add_argument("--header", default=None,
                        help="header filename; default: <name.lower()>_data.h")
    parser.add_argument("--source", default=None,
                        help="source filename; default: <name.lower()>_data.cc")
    args = parser.parse_args()

    arr = np.load(args.npy)
    arr = np.ascontiguousarray(arr)

    dtype = arr.dtype
    if dtype not in DTYPE_TO_C:
        raise TypeError(f"Unsupported dtype {dtype}. Convert the npy dtype first, e.g. astype(np.float32).")

    name = args.name.upper()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header_path = out_dir / (args.header or f"{name.lower()}_data.h")
    source_path = out_dir / (args.source or f"{name.lower()}_data.cc")

    ctype = DTYPE_TO_C[dtype]
    ndim = arr.ndim
    shape = list(arr.shape)
    size = int(arr.size)

    with open(header_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <cstddef>\n")
        f.write("#include <cstdint>\n")
        if np.issubdtype(dtype, np.floating):
            f.write("#include <cmath>\n")
        f.write("\n")
        f.write(f"extern const int {name}_NDIM;\n")
        f.write(f"extern const int64_t {name}_SHAPE[{ndim}];\n")
        f.write(f"extern const size_t {name}_SIZE;\n")
        f.write(f"extern const {ctype} {name}_DATA[{size}];\n")

    with open(source_path, "w", encoding="utf-8") as f:
        f.write(f'#include "{header_path.name}"\n\n')
        f.write(f"const int {name}_NDIM = {ndim};\n")
        f.write(f"const int64_t {name}_SHAPE[{ndim}] = " + "{")
        f.write(", ".join(str(x) for x in shape))
        f.write("};\n")
        f.write(f"const size_t {name}_SIZE = {size};\n")
        f.write(f"const {ctype} {name}_DATA[{size}] = {{\n")
        write_array_values(f, arr, ctype)
        f.write("};\n")

    print(f"wrote: {header_path}")
    print(f"wrote: {source_path}")
    print(f"dtype={dtype}, shape={shape}, size={size}")


if __name__ == "__main__":
    main()
