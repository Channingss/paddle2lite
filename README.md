1. introduction

paddle2lite is a python package for paddle, which can convert&optimize model for Paddle-Lite. For more detail, please refer to [model_optimize_tool](https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/model_optimize_tool/)

2. Usage

2.1 example 

`from deploy_paddle.lite import optimze, Place, TargetType, PrecisionType`
   
   `place = [Place(TargetType.ARM, PrecisionType.float32)]`
   
   `optimize(model_dir,model_file, param_file, optimize_out_type, optimize_out, valid_targets, record_tailoring_info)`

2.2 API

2.2.1 optimize

- API:

optimize(model_dir,model_file, param_file, optimize_out_type, optimize_out, valid_targets, record_tailoring_info)
- parameter：
   model_dir: model_param_dir
   model_file: model_path
   param_file: param_path
   optimize_out_type: protobuf or naive_buffer
   optimize_out: output_optimize_model_dir
   valid_targets: arm / opencl / x86 / npu / xpu
   record_tailoring_info: true / false
   
   future add：
   prefer_int8_kernel: true / false
   
2.2.2 Place
Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)

- TargetType

"Host": kHost

"X86": kX86

"CUDA": kCUDA

"ARM": kARM

"OpenCL": kOpenCL

"FPGA": kFPGA

"NPU": kNPU

"Any": kAny

- PrecisionType

"FP16": kFP16

"FP32": kFloat

"INT8": kInt8

"INT16": kInt16

"INT32": kInt32

"INT64": kInt64

"BOOL": kBool

"Any": kAny

- DataLayoutType

"NCHW": kNCHW

"NHWC": kNHWC

"ImageDefault": kImageDefault

"ImageFolder": kImageFolder

"ImageNW": kImageNW

"Any": kAny
