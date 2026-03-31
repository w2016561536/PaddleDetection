import onnx
from onnx import shape_inference

model = onnx.load("picodet_m_416_coco_sim.onnx")

# for inp in model.graph.input:
#     shape = inp.type.tensor_type.shape
#     if len(shape.dim) > 0:
#         shape.dim[0].dim_value = 1
#         shape.dim[0].dim_param = ""

model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1


model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

onnx.save(model, "picodet_m_416_coco_sim_bs1.onnx")

