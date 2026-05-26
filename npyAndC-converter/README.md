# PaddleLite 单层 conv2d 验证脚本

## 文件

- `npy_to_c.py`：把输入 `.npy` 转成 `input_data.h + input_data.cc`
- `run_paddle_lite_conv.cc`：加载 `conv2dTest.nb`，喂入 C 数组，运行 Paddle Lite，输出 `output_data.cc`
- `c_array_to_npy.py`：把 `output_data.cc` 转回 `.npy`
- `CMakeLists.txt`：可选构建文件

## 运行流程

假设输入是 `input.npy`，shape 是 `1x16x208x208`，dtype 是 `float32`。

```bash
python3 npy_to_c.py input.npy --name INPUT --out-dir .
```

生成：

```text
input_data.h
input_data.cc
```

编译：

```bash
export PADDLE_LITE_DIR=/path/to/Paddle-Lite-inference_lite_lib
mkdir -p build
cd build
cmake .. -DPADDLE_LITE_DIR=${PADDLE_LITE_DIR}
cmake --build . -j
cd ..
```

运行：

```bash
./build/run_paddle_lite_conv conv2dTest.nb output_data.cc 1
```

转回 npy：

```bash
python3 c_array_to_npy.py output_data.cc output.npy --name OUTPUT
```

## 注意

1. `run_paddle_lite_conv.cc` 默认输入和输出 API tensor 都用 `float`。
2. 这与你图里的结构匹配：`calib` 在模型内部把 fp32 输入转成 int8，conv 是 int8/fp32_out。
3. 如果你的 `.nb` 真正暴露的输出 tensor 是 int8，需要把 C++ 里的 `output_tensor->data<float>()` 改成 `data<int8_t>()`，并同步修改输出 C array 类型。
4. Android 端运行时需要把 `libpaddle_light_api_shared.so` 放到可加载路径，例如 `LD_LIBRARY_PATH` 指向对应目录。
