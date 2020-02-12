// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "all_kernel_faked.cc"  // NOLINT
#include "kernel_src_map.h"     // NOLINT
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/core/op_registry.h"
//#include "lite/utils/cp_logging.h"
//#include "lite/utils/string.h"
#include "lite/api/py_optimize/pybind/pybind.h"

namespace py = pybind11;
namespace paddle {
namespace lite {
namespace pybind {

int add(int i, int j) {
        return i + j;
}

void RunOptimize(const std::string& model_dir,
                 const std::string& model_file,
                 const std::string& param_file,
                 const std::string& optimize_out,
                 const std::string& optimize_out_type,
                 const std::vector<Place>& valid_places,
                 bool record_tailoring_info) {
  if (!model_file.empty() && !param_file.empty()) {
    LOG(WARNING)
        << "Load combined-param model. Option model_dir will be ignored";
  }

  lite_api::CxxConfig config;
  config.set_model_dir(model_dir);
  config.set_model_file(model_file);
  config.set_param_file(param_file);

  config.set_valid_places(valid_places);

  auto predictor = lite_api::CreatePaddlePredictor(config);

  paddle::lite_api::LiteModelType  model_type;
  if (optimize_out_type == "protobuf") {
    model_type = paddle::lite_api::LiteModelType::kProtobuf;
  } else if (optimize_out_type == "naive_buffer") {
    model_type = paddle::lite_api::LiteModelType::kNaiveBuffer;
  } else {
    LOG(FATAL) << "Unsupported Model type :" << optimize_out_type;
  }

  OpKernelInfoCollector::Global().SetKernel2path(kernel2path_map);
  predictor->SaveOptimizedModel(
      optimize_out, model_type, record_tailoring_info);
  if (record_tailoring_info) {
    LOG(INFO) << "Record the information of tailored model into :"
              << optimize_out;
  }
}

void BindLitePlace(py::module *m) {
  // TargetType
  py::enum_<TargetType>(*m, "TargetType")
      .value("Host", TargetType::kHost)
      .value("X86", TargetType::kX86)
      .value("CUDA", TargetType::kCUDA)
      .value("ARM", TargetType::kARM)
      .value("OpenCL", TargetType::kOpenCL)
      .value("FPGA", TargetType::kFPGA)
      .value("NPU", TargetType::kNPU)
      .value("Any", TargetType::kAny);

  // PrecisionType
  py::enum_<PrecisionType>(*m, "PrecisionType")
      .value("FP16", PrecisionType::kFP16)
      .value("FP32", PrecisionType::kFloat)
      .value("INT8", PrecisionType::kInt8)
      .value("INT16", PrecisionType::kInt16)
      .value("INT32", PrecisionType::kInt32)
      .value("INT64", PrecisionType::kInt64)
      .value("BOOL", PrecisionType::kBool)
      .value("Any", PrecisionType::kAny);

  // DataLayoutType
  py::enum_<DataLayoutType>(*m, "DataLayoutType")
      .value("NCHW", DataLayoutType::kNCHW)
      .value("NHWC", DataLayoutType::kNHWC)
      .value("ImageDefault", DataLayoutType::kImageDefault)
      .value("ImageFolder", DataLayoutType::kImageFolder)
      .value("ImageNW", DataLayoutType::kImageNW)
      .value("Any", DataLayoutType::kAny);

  // Place
  py::class_<Place>(*m, "Place")
      .def(py::init<TargetType, PrecisionType, DataLayoutType, int16_t>(),
           py::arg("target"),
           py::arg("percision") = PrecisionType::kFloat,
           py::arg("layout") = DataLayoutType::kNCHW,
           py::arg("device") = 0)
      .def("is_valid", &Place::is_valid);
}

void BindLiteApi(py::module *m) {
    BindLitePlace(m);
    m->def("optimize", &RunOptimize, "optimize and convert paddle model to lite model",
            py::arg("model_dir"),
            py::arg("model_file"),
            py::arg("param_file"),
            py::arg("optimize_out"),
            py::arg("optimize_out_type"),
            py::arg("valida_places"),
            py::arg("record_tailoring_info")
            );
}

}  // namespace pybind
}  // namespace lite
}  // namespace paddle

