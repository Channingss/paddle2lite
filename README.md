# 1. introduction

paddle2lite是一个工具，用于将paddle的模型转换成Paddle-Lite可运行的格式。本项目对[model_optimize_tool](https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/model_optimize_tool/)使用pybind11进行封装后，提供了python的接口方便用户调用。

## 1.1 example 
```
from deploy_paddle.lite import optimze, Place, TargetType, PrecisionType
   
place = [Place(TargetType.ARM, PrecisionType.float32)]
   
optimize(model_dir,model_file, param_file, optimize_out_type, optimize_out, valid_targets, record_tailoring_info)
```

# 2. Install

## 2.1 compile
从源码编译model_optimize_tool前需要先，[安装Paddle-Lite开发环境](https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/source_compile/)

```
git clone https://github.com/Channingss/paddle2lite.git
cd paddle2lite
git checkout <release-version-tag>
./lite/tools/build.sh --py_version=3.6 build_optimize_tool
```

编译结果位于paddle2lite/dist/**.wkl
## 2.2 pip 

pip install -i https://test.pypi.org/simple/ paddle2lite

## 2.2 Usage

### 2.2.1 optimize

- API:
optimize(model_dir,model_file, param_file, optimize_out_type, optimize_out, valid_targets, record_tailoring_info)

- parameter：

|parameter| mean|
|-|-|
|model_dir|model_param_dir|
|model_file|model_path|
|param_file|param_path|
|optimize_out_type|'protobuf' or 'naive_buffer'|
|optimize_out|output_optimize_model_dir|
|valid_targets| Place(TargetType, PrecisionType, DataLayoutType)|
|prefer_int8_kernel|true / false|
|record_tailoring_info|true / false|

### 2.2.2 Place

Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)

|TargetType|PrecisionType|DataLayoutType|
|-|-|-|
|CUDA|FP16|NCHW
|ARM|FP32|NHWC
|OpenCL|INT8|ImageDefault
|FPGA|INT16|ImageFolder
|NPU|INT32|ImageNW
|Any|INT64|Any
||BOOL|
||Any|

