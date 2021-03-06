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

#include "lite/operators/lookup_table_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LookupTableOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.W)
  CHECK_OR_FALSE(param_.Ids)
  CHECK_OR_FALSE(param_.Out)

  auto table_dims = param_.W->dims();
  auto ids_dims = param_.Ids->dims();

  int ids_rank = ids_dims.size();

  CHECK_EQ_OR_FALSE(table_dims.size(), 2)
  CHECK_EQ_OR_FALSE(ids_dims[ids_rank - 1], 1)

  return true;
}

bool LookupTableOpLite::InferShape() const {
  auto table_dims = param_.W->dims();
  auto ids_dims = param_.Ids->dims();

  int ids_rank = ids_dims.size();

  auto output_dims = ids_dims.Slice(0, ids_rank - 1);

  std::vector<int64_t> out_dims;
  for (int i = 0; i < ids_rank - 1; ++i) {
    out_dims.push_back(ids_dims[i]);
  }
  out_dims.push_back(table_dims[1]);
  param_.Out->Resize(lite::DDim{out_dims});
  param_.Out->set_lod(param_.Ids->lod());
  return true;
}

bool LookupTableOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                   lite::Scope *scope) {
  auto input = op_desc.Input("W").front();
  auto ids = op_desc.Input("Ids").front();
  auto out = op_desc.Output("Out").front();

  param_.W = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.Ids = scope->FindVar(ids)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  param_.padding_idx = op_desc.GetAttr<int64_t>("padding_idx");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lookup_table, paddle::lite::operators::LookupTableOpLite)
