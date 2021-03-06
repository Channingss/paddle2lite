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

#pragma once

#include <xtcl/xtcl.h>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

// Type/tensor converters for converting Paddle type/tensor to XPU type/tensor
bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

xtcl::DataType CvtPrecisionType(PrecisionType in_type);

DLDataType CvtDataType(PrecisionType in_type);

xtcl::Array<xtcl::xIndexExpr> CvtShape(const std::vector<int>& in_shape);

xtcl::Array<xtcl::xIndexExpr> CvtShape(const std::vector<int64_t>& in_shape);

xtcl::Array<xtcl::xIndexExpr> CvtShape(const DDim& in_dims);

std::shared_ptr<xtcl::xNDArray> CvtTensor(
    const Tensor& in_tensor,
    std::vector<int64_t> out_shape = {},
    PrecisionType in_ptype = PRECISION(kFloat),
    DataLayoutType in_ltype = DATALAYOUT(kNCHW));

xtcl::Array<xtcl::Integer> Cvt2ArrayInt(const std::vector<int64_t>& input);
xtcl::Array<xtcl::Integer> Cvt2ArrayInt(const DDim& input);

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
