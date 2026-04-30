import os
import numpy as np
import paddle

paddle.enable_static()

model_prefix = "output_inference/picodet_s_416_lcnet_quant_lastall/model"   # 不带 .pdmodel / .pdiparams
save_npy = "fullOutvalidate.npy"

# 你的新输入，也就是 hardswish_1.tmp_0
input_name = "hardswish_1.tmp_0"

# 替换成真实输入数据
# shape 必须和 hardswish_1.tmp_0 在原模型里的 shape 一致
input_data = np.load("output_hardwish_1.npy").astype("float32")

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)

program, feed_names, fetch_targets = paddle.static.load_inference_model(
    model_prefix,
    exe
)

print("feed_names:", feed_names)
print("fetch_targets:", [v.name for v in fetch_targets])

print(np.array(
            [[0.866667, 0.65]]).astype('float32').shape)
print(np.array(
            [[0.866667, 0.65]]).astype('float32').dtype)

result = exe.run(
    program=program,
    feed={input_name: input_data, "scale_factor": 
          np.array(
            [[0.866667, 0.65]]).astype('float32')},
    fetch_list=fetch_targets
)

# 如果只有一个输出
output = result[0]
np.save(save_npy, output)

# 按照 classid, score ,x1, y1, x2, y2 的格式打印，只保留score大于0.5的框
for det in output:
    class_id, score, x1, y1, x2, y2 = det
    if score > 0.5:
        print(f"class_id: {class_id}, score: {score}, bbox: [{x1}, {y1}, {x2}, {y2}]")



print("saved:", save_npy)
print("output shape:", output.shape)
print("output dtype:", output.dtype)
