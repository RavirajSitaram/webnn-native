// Copyright 2021 The WebNN-native Authors
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

#ifndef WEBNN_NATIVE_NNAPI_MODEL_NN_H_
#define WEBNN_NATIVE_NNAPI_MODEL_NN_H_

#include <map>
#include <set>
#include <unordered_set>
#include <vector>

#include "webnn_native/Error.h"
#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"
#include "webnn_native/nnapi/ContextNnapi.h"
#include "webnn_native/ops/BatchNorm.h"
#include "webnn_native/ops/Binary.h"
#include "webnn_native/ops/Clamp.h"
#include "webnn_native/ops/Concat.h"
#include "webnn_native/ops/Constant.h"
#include "webnn_native/ops/Conv2d.h"
#include "webnn_native/ops/Gemm.h"
#include "webnn_native/ops/Input.h"
#include "webnn_native/ops/InstanceNorm.h"
#include "webnn_native/ops/LeakyRelu.h"
#include "webnn_native/ops/Pad.h"
#include "webnn_native/ops/Pool2d.h"
#include "webnn_native/ops/Reduce.h"
#include "webnn_native/ops/Resample2d.h"
#include "webnn_native/ops/Reshape.h"
#include "webnn_native/ops/Slice.h"
#include "webnn_native/ops/Split.h"
#include "webnn_native/ops/Squeeze.h"
#include "webnn_native/ops/Transpose.h"
#include "webnn_native/ops/Unary.h"

#include "NeuralNetworksTypes.h"
#include "NnapiManager.h"
#include "NnapiUtils.h"
#include "nnapi_implementation.h"
#include "webnn_native/nnapi/ErrorNnapi.h"

namespace webnn_native { namespace nnapi {

    class Graph : public GraphBase {
      public:
        explicit Graph(Context* context);
        ~Graph() override;

        virtual MaybeError AddConstant(const op::Constant* constant) override;
        virtual MaybeError AddInput(const op::Input* input) override;
        virtual MaybeError AddOutput(const std::string& name, const OperandBase* ouput) override;
        virtual MaybeError AddBatchNorm(const op::BatchNorm* batchNorm) override;
        virtual MaybeError AddBinary(const op::Binary* binary) override;
        virtual MaybeError AddClamp(const op::Clamp* clamp) override;
        virtual MaybeError AddConv2d(const op::Conv2d* conv2d) override;
        virtual MaybeError AddPad(const op::Pad* pad) override;
        virtual MaybeError AddPool2d(const op::Pool2d* pool2d) override;
        virtual MaybeError AddReduce(const op::Reduce* reduce) override;
        virtual MaybeError AddResample2d(const op::Resample2d* resample) override;
        virtual MaybeError AddReshape(const op::Reshape* reshape) override;
        virtual MaybeError AddSlice(const op::Slice* slice) override;
        virtual MaybeError AddSplit(const op::Split* split) override;
        virtual MaybeError AddSqueeze(const op::Squeeze* squeeze) override;
        virtual MaybeError AddTranspose(const op::Transpose* transpose) override;
        virtual MaybeError AddUnary(const op::Unary* unary) override;
        virtual MaybeError AddConcat(const op::Concat* concat) override;
        virtual MaybeError AddGemm(const op::Gemm* Gemm) override;
        virtual MaybeError AddInstanceNorm(const op::InstanceNorm* InstanceNorm) override;
        virtual MaybeError Finish() override;

        MaybeError AddTransposeImpl(NodeInfo& filterNode,
                                    NodeInfo& outputNode,
                                    int32_t* permute,
                                    uint32_t permuteSize);
        MaybeError AddExpandDimsImpl(NodeInfo& node, int32_t dim_index, uint32_t& index);
        MaybeError AddClampImpl(NodeInfo& inputNode, NodeInfo& outputNode, float min, float max);
        MaybeError AddLeakyReluImpl(NodeInfo& inputNode, NodeInfo& outputNode, float alpha);
        MaybeError AddSigmoidImpl(NodeInfo& inputNode, NodeInfo& outputNode);
        MaybeError AddSoftMax(NodeInfo& input0Node, NodeInfo& outputNode);

        MaybeError CreateNode(NodeInfo& outNode, ml::OperandType type, std::vector<int32_t> dims);
        MaybeError AddMatMulImpl(NodeInfo& input0NodeInfo,
                                 NodeInfo& input1NodeInfo,
                                 NodeInfo& outputNode,
                                 std::vector<int32_t> dims,
                                 uint32_t& outputIndex);

      private:
        uint32_t getOperandIdx() {
            return operandCount++;
        }

        MaybeError CompileImpl() override;
        MLComputeGraphStatus ComputeImpl(NamedInputsBase* inputs,
                                         NamedOutputsBase* outputs) override;

        // Map the input name to NNAPI internal input number.
        std::map<std::string, NodeInfo> mInputIdMap;
        // Map the output name to IE internal original output name that will be updated after
        // TransposeSinking.
        std::map<std::string, NodeInfo> mOutputNameMap;
        // Map the operand to IE internal id
        std::map<const OperandBase*, std::string> mOperandIdMap;
        // store the constant operands
        // std::unordered_set<const OperandBase*> mConstantSet;
        std::map<const OperandBase*, uint32_t> mGraphNodeMap;  // Add operand index
        std::vector<uint32_t> mGraphOutputs;
        std::vector<uint32_t> mGraphInputs;

        // const NnApi* mNnapi;
        std::map<uint32_t, NodeInfo> mGraphOperandInfo;
        uint32_t operandCount;
        ANeuralNetworksOperandType mScalarInt32Operand, mScalarBoolOperand;
        NnapiManager* mNnapiMgr;

        std::vector<std::unique_ptr<int32_t>> memInt32Vec;
    };

}}  // namespace webnn_native::nnapi

#endif  // WEBNN_NATIVE_NNAPI_MODEL_NN_H_
