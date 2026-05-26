// run_paddle_lite_conv.cc
//
// Build this together with the generated input_data.cc.
//
// Example:
//   export PADDLE_LITE_DIR=/path/to/Paddle-Lite-inference_lite_lib.android.armv8.gcc
//   g++ -std=c++11 -O2 \
//     run_paddle_lite_conv.cc input_data.cc \
//     -I${PADDLE_LITE_DIR}/cxx/include \
//     -L${PADDLE_LITE_DIR}/cxx/lib -lpaddle_light_api_shared \
//     -Wl,-rpath,${PADDLE_LITE_DIR}/cxx/lib \
//     -ldl -pthread \
//     -o run_paddle_lite_conv
//
// Run:
//   ./run_paddle_lite_conv conv2dTest.nb output_data.cc 1
//
// Args:
//   argv[1] model nb path, e.g. conv2dTest.nb
//   argv[2] output C array path, e.g. output_data.cc
//   argv[3] optional thread count, default 1
//
// Assumption:
//   The model input and output tensor are float API tensors.
//   In your graph, calib converts fp32 input to int8 internally, and conv is int8/fp32_out.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "paddle_api.h"
#include "input_data.h"

using paddle::lite_api::CreatePaddlePredictor;
using paddle::lite_api::MobileConfig;
using paddle::lite_api::PaddlePredictor;
using paddle::lite_api::PowerMode;
using paddle::lite_api::Tensor;
using paddle::lite_api::shape_t;

static int64_t ShapeProduction(const shape_t& shape) {
  int64_t n = 1;
  for (auto v : shape) {
    n *= v;
  }
  return n;
}

static std::vector<int64_t> InputShapeVector() {
  std::vector<int64_t> shape;
  shape.reserve(INPUT_NDIM);
  for (int i = 0; i < INPUT_NDIM; ++i) {
    shape.push_back(INPUT_SHAPE[i]);
  }
  return shape;
}

static void WriteFloatArrayAsC(const std::string& path,
                               const float* data,
                               const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (auto v : shape) {
    size *= v;
  }

  std::ofstream ofs(path.c_str());
  if (!ofs) {
    std::cerr << "Cannot open output file: " << path << std::endl;
    std::exit(2);
  }

  ofs << "#include <cstddef>\n";
  ofs << "#include <cstdint>\n";
  ofs << "#include <cmath>\n\n";

  ofs << "const int OUTPUT_NDIM = " << shape.size() << ";\n";
  ofs << "const int64_t OUTPUT_SHAPE[" << shape.size() << "] = {";
  for (size_t i = 0; i < shape.size(); ++i) {
    ofs << shape[i];
    if (i + 1 != shape.size()) ofs << ", ";
  }
  ofs << "};\n";
  ofs << "const size_t OUTPUT_SIZE = " << size << ";\n";
  ofs << "const float OUTPUT_DATA[" << size << "] = {\n";

  ofs << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (int64_t i = 0; i < size; ++i) {
    ofs << "  " << data[i];
    if (i + 1 != size) ofs << ",";
    ofs << "\n";
  }
  ofs << "};\n";
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " conv2dTest.nb output_data.cc [threads]\n";
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string output_c_path = argv[2];
  const int threads = (argc >= 4) ? std::max(1, std::atoi(argv[3])) : 1;

  MobileConfig config;
  config.set_model_from_file(model_path);
  config.set_threads(threads);
  config.set_power_mode(PowerMode::LITE_POWER_NO_BIND);

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  if (!predictor) {
    std::cerr << "CreatePaddlePredictor failed\n";
    return 2;
  }

  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  if (!input_tensor) {
    std::cerr << "GetInput(0) failed\n";
    return 3;
  }

  std::vector<int64_t> input_shape = InputShapeVector();
  input_tensor->Resize(input_shape);

  const int64_t input_size = ShapeProduction(input_tensor->shape());
  if (input_size != static_cast<int64_t>(INPUT_SIZE)) {
    std::cerr << "Input size mismatch. Tensor wants " << input_size
              << ", C array has " << INPUT_SIZE << "\n";
    return 4;
  }

  float* input_ptr = input_tensor->mutable_data<float>();
  std::copy(INPUT_DATA, INPUT_DATA + INPUT_SIZE, input_ptr);

  if (!predictor->Run()) {
    std::cerr << "predictor->Run() failed\n";
    return 5;
  }

  std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  if (!output_tensor) {
    std::cerr << "GetOutput(0) failed\n";
    return 6;
  }

  shape_t output_shape_lite = output_tensor->shape();
  std::vector<int64_t> output_shape(output_shape_lite.begin(), output_shape_lite.end());
  const int64_t output_size = ShapeProduction(output_shape_lite);

  const float* output_ptr = output_tensor->data<float>();
  if (!output_ptr) {
    std::cerr << "output_tensor->data<float>() returned null\n";
    return 7;
  }

  WriteFloatArrayAsC(output_c_path, output_ptr, output_shape);

  std::cout << "model: " << model_path << "\n";
  std::cout << "input shape: ";
  for (auto v : input_shape) std::cout << v << " ";
  std::cout << "\noutput shape: ";
  for (auto v : output_shape) std::cout << v << " ";
  std::cout << "\nwrote: " << output_c_path << "\n";

  return 0;
}
