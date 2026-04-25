import onnx
import numpy as np
from onnx import helper, numpy_helper, shape_inference

def get_initializer_dict(graph):
    return {init.name: init for init in graph.initializer}

def replace_conv_bias_with_add(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    init_dict = get_initializer_dict(graph)

    new_nodes = []
    new_initializers = []

    for node in graph.node:
        if node.op_type != "Conv" or len(node.input) != 3:
            new_nodes.append(node)
            continue

        x, w, b = node.input
        y = node.output[0]

        if b not in init_dict:
            # bias 不是常量，先跳过
            new_nodes.append(node)
            continue

        bias_arr = numpy_helper.to_array(init_dict[b])
        if bias_arr.ndim != 1:
            new_nodes.append(node)
            continue

        c = bias_arr.shape[0]

        # 从权重维度判断是 Conv1D/2D/3D
        w_arr = numpy_helper.to_array(init_dict[w])
        # Conv 权重通常是 [M, C/group, k1, k2, ...]
        spatial_rank = w_arr.ndim - 2
        reshape_shape = [1, c] + [1] * spatial_rank

        conv_out = y + "_conv_nobias"
        bias_reshaped = b + "_reshape_out"
        shape_name = b + "_reshape_shape"

        # 1) Conv 去掉 bias
        new_conv = helper.make_node(
            "Conv",
            inputs=[x, w],
            outputs=[conv_out],
            name=(node.name + "_nobias") if node.name else ""
        )
        # 保留原 attributes
        new_conv.attribute.extend(node.attribute)

        # 2) 生成 reshape shape 常量
        shape_init = numpy_helper.from_array(
            np.asarray(reshape_shape, dtype=np.int64),
            name=shape_name
        )
        new_initializers.append(shape_init)

        # 3) Reshape bias
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[b, shape_name],
            outputs=[bias_reshaped],
            name=(node.name + "_bias_reshape") if node.name else ""
        )

        # 4) Add
        add_node = helper.make_node(
            "Add",
            inputs=[conv_out, bias_reshaped],
            outputs=[y],
            name=(node.name + "_bias_add") if node.name else ""
        )

        new_nodes.extend([new_conv, reshape_node, add_node])

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)

    onnx.checker.check_model(model)
    model = shape_inference.infer_shapes(model)
    return model

model = onnx.load("picodet_m_416_coco_qat_sim.onnx")
model = replace_conv_bias_with_add(model)
onnx.save(model, "model_conv_bias_to_add.onnx")
