# 1. introduction

paddle2lite是一个工具，用于将paddle的模型转换成Paddle-Lite可运行的格式。本项目使用pybind11对model_optimize_tool进行封装，提供python接口方便用户调用。model_optimize_tool更详细的介绍请参考：[model_optimize_tool](https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/model_optimize_tool/)。

## 1.1 example 
```
from deploy_paddle.lite import optimze, Place, TargetType, PrecisionType
   
place = [Place(TargetType.ARM, PrecisionType.float32)]
   
optimize(model_dir,model_file, param_file, optimize_out_type, optimize_out, valid_places, record_tailoring_info)
```

# 2. Install

- compile
- pip

## 2.1 compile
从源码编译model_optimize_tool前需要先，[安装Paddle-Lite开发环境](https://paddlepaddle.github.io/Paddle-Lite/v2.2.0/source_compile/)

```
git clone https://github.com/Channingss/paddle2lite.git
cd paddle2lite
git checkout release/v2.2.0
./lite/tools/build.sh --py_version=3.6 build_optimize_tool
```

编译结果位于paddle2lite/dist/**.wkl
## 2.2 pip 

pip install -i https://test.pypi.org/simple/ paddle2lite

# 3. Usage

## 3.1 optimize

- API:

optimize(model_dir,model_file, param_file, optimize_out_type, optimize_out, valid_targets, record_tailoring_info)

- parameter：

|parameter| mean|
|-|-|
| model_dir| 待优化的PaddlePaddle模型（非combined形式）的路径|
| model_file| 待优化的PaddlePaddle模型（combined形式）的网络结构文件路径|
| param_file| 待优化的PaddlePaddle模型（combined形式）的权重文件路径|
| optimize_out_type| 输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf|
| optimize_out| 优化模型的输出路径|
| valid_places| 指定模型可执行的[places](#Place)(TargetType, PrecisionType, DataLayoutType)可以同时指定多个places(list)，Model Optimize Tool将会自动选择最佳方式。
| prefer_int8_kernel| 若待优化模型为int8量化模型（如量化训练得到的量化模型），则设置该选项为true以使用int8内核函数进行推理加速，默认为false|
| record_tailoring_info| 当使用根据模型裁剪库文件功能时，则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false|

## 3.2 Place
- API:

<a id='Place'>Place</a>(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)

- parameter：

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

