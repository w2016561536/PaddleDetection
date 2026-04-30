import paddle

paddle.enable_static()

model_prefix = "output_inference/picodet_s_416_lcnet_quant_only_tail/model"   # 不带 .pdmodel/.pdiparams
save_prefix = "output_inference/picodet_s_416_lcnet_quant_only_tail_conv2d/model"

exe = paddle.static.Executor(paddle.CPUPlace())

program, feed_names, fetch_targets = paddle.static.load_inference_model(
    model_prefix,
    exe
)

block = program.global_block()

# 查看所有变量名，找到你想截断处的 tensor
for name in block.vars:
    print(name)

# 例如你想把这个中间 tensor 作为新输出
target_var_name = "hardswish_2.tmp_0"
new_fetch = block.var(target_var_name)

feed_vars = [
    block.var(name)
    for name in feed_names
    if name != "scale_factor"
]

paddle.static.save_inference_model(
    path_prefix=save_prefix,
    feed_vars=feed_vars,
    fetch_vars=[new_fetch],
    executor=exe,
    program=program
)
