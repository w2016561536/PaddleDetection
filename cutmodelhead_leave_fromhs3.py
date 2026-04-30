import os
import paddle

paddle.enable_static()

src_prefix = "output_inference/picodet_s_416_lcnet_quant/model"       # 你上一步保存的模型，不带后缀
dst_prefix = "output_inference/picodet_s_416_lcnet_quant_fromhs3/model"        # 新输出，不带后缀
new_input_name = "hardswish_3.tmp_0"

exe = paddle.static.Executor(paddle.CPUPlace())

program, old_feed_names, old_fetch_vars = paddle.static.load_inference_model(
    src_prefix,
    exe
)

block = program.global_block()

# 确认变量存在
assert new_input_name in block.vars, f"{new_input_name} not found"

new_input = block.var(new_input_name)
new_input_scale_factor = block.var("scale_factor")
# 关键：把中间变量改成 feedable
new_input.is_data = True
new_input.persistable = False
new_input.stop_gradient = True

# 使用原模型输出作为新模型输出
fetch_vars = old_fetch_vars

os.makedirs(os.path.dirname(dst_prefix), exist_ok=True)

paddle.static.save_inference_model(
    path_prefix=dst_prefix,
    feed_vars=[new_input, new_input_scale_factor],
    fetch_vars=fetch_vars,
    executor=exe,
    program=program
)
