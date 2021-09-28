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

#include "webnn_native/nnapi/GraphNnapi.h"

#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <vector>

#include "NnapiUtils.h"
#include "common/Assert.h"
#include "common/Log.h"
#include "nnapi_implementation.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOperands.h"
#include "webnn_native/NamedOutputs.h"

#define WEBNN_ASSERT(condition, message) \
    do {                                 \
        dawn::ErrorLog() << message;     \
        DAWN_ASSERT(condition);          \
    } while (0)

namespace webnn_native { namespace nnapi {

    Graph::Graph(Context* context) : GraphBase(context) {
        operandCount = 0;

        mScalarInt32Operand.type = ANEURALNETWORKS_INT32;
        mScalarInt32Operand.dimensionCount = 0;
        mScalarInt32Operand.dimensions = nullptr;
        mScalarInt32Operand.scale = 0.0f;
        mScalarInt32Operand.zeroPoint = 0;

        mScalarBoolOperand.type = ANEURALNETWORKS_BOOL;
        mScalarBoolOperand.dimensionCount = 0;
        mScalarBoolOperand.dimensions = nullptr;
        mScalarBoolOperand.scale = 0.0f;
        mScalarBoolOperand.zeroPoint = 0;

        mNnapiMgr = new NnapiManager();
    }

    Graph::~Graph() {
        for (auto node : mGraphOperandInfo) {
            if (node.second.mem)
                mNnapiMgr->FreeMemory(node.second.mem);

            close(node.second.fd);
        }

        delete mNnapiMgr;
    }

    MaybeError Graph::AddConstant(const op::Constant* constant) {
        NodeInfo node;
        void* buffer = const_cast<void*>(constant->GetBuffer());

        CreateNodeFromOperandDescriptor(constant->GetOperandDescriptor(), &node);
        DAWN_TRY(mNnapiMgr->CreateOperandAndSetMemory("const", &node, buffer));

        mGraphOperandInfo[node.opIndex] = node;
        mGraphNodeMap[constant->PrimaryOutput()] = node.opIndex;

        return {};
    }

    MaybeError Graph::AddInput(const op::Input* input) {
        NodeInfo node;

        CreateNodeFromOperandDescriptor(input->GetOperandDescriptor(), &node);
        DAWN_TRY(mNnapiMgr->CreateInputOutputOperand(input->GetName(), &node));

        mGraphOperandInfo[node.opIndex] = node;
        mGraphNodeMap[input->PrimaryOutput()] = node.opIndex;
        mInputIdMap[input->GetName()] = node;
        mGraphInputs.push_back(node.opIndex);

        return {};
    }

    MaybeError Graph::AddOutput(const std::string& name, const OperandBase* output) {
        uint32_t index = mGraphNodeMap[output];
        auto node = mGraphOperandInfo[index];

        DAWN_TRY(mNnapiMgr->CreateInputOutputOperand(name, &node, false));

        mGraphOutputs.push_back(index);
        mOutputNameMap[name] = node;
        return {};
    }

    MaybeError Graph::AddInstanceNorm(const op::InstanceNorm* instanceNorm) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Instance norm"));
        return {};
    }

    MaybeError Graph::AddBatchNorm(const op::BatchNorm* batchNorm) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Batch norm"));
        return {};
    }

    MaybeError Graph::AddBinary(const op::Binary* binary) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi AddBinary"));
        return {};
    }

    MaybeError Graph::AddClamp(const op::Clamp* clamp) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Clamp"));
        return {};
    }

    MaybeError Graph::AddSlice(const op::Slice* slice) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi AddSlice"));
        return {};
    }

    MaybeError Graph::AddTransposeImpl(NodeInfo& filterNode,
                                       ml::FilterOperandLayout layout,
                                       uint32_t& index) {
        index = filterNode.opIndex;
        if (layout != ml::FilterOperandLayout::Ohwi) {
            NodeInfo permNode, outputNode;
            std::vector<uint32_t> inputList;
            int32_t* perm = nullptr;

            switch (layout) {
                case ml::FilterOperandLayout::Oihw:
                    memInt32Vec.emplace_back(new int(4));
                    perm = memInt32Vec.back().get();
                    perm[0] = 0;
                    perm[1] = 2;
                    perm[2] = 3;
                    perm[3] = 1;
                    break;
                case ml::FilterOperandLayout::Hwio:
                    memInt32Vec.emplace_back(new int(4));
                    perm = memInt32Vec.back().get();
                    perm[0] = 3;
                    perm[1] = 0;
                    perm[2] = 1;
                    perm[3] = 2;
                    break;
                case ml::FilterOperandLayout::Ihwo:
                    memInt32Vec.emplace_back(new int(4));
                    perm = memInt32Vec.back().get();
                    perm[0] = 3;
                    perm[1] = 1;
                    perm[2] = 2;
                    perm[3] = 0;
                    break;
                default:
                    break;
            }

            std::vector<uint32_t> outDims(filterNode.dimensions.size());
            for (size_t i = 0; i < filterNode.dimensions.size(); i++) {
                outDims[i] = filterNode.dimensions[perm[i]];
            }

            permNode.type = ml::OperandType::Int32;
            permNode.dimensions = {4};
            DAWN_TRY(mNnapiMgr->CreateOperand(&permNode));
            DAWN_TRY(mNnapiMgr->AddVecOperand(permNode.opIndex, perm, sizeof(int32_t) * 4));

            inputList.push_back(filterNode.opIndex);
            inputList.push_back(permNode.opIndex);
            outputNode.type = filterNode.type;
            outputNode.dimensions = outDims;

            DAWN_TRY(mNnapiMgr->CreateOperand(&outputNode));
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_TRANSPOSE, 2, inputList.data(), 1,
                                             &outputNode.opIndex));
            index = outputNode.opIndex;
        }

        return {};
    }

    MaybeError Graph::AddConv2d(const op::Conv2d* conv2d) {
        auto options = conv2d->GetOptions();

        auto getOutputChannels = [&](std::vector<uint32_t>& filterDims) {
            switch (options->filterLayout) {
                case ml::FilterOperandLayout::Hwio:
                    return filterDims[3];
                case ml::FilterOperandLayout::Ihwo:
                    return filterDims[3];
                case ml::FilterOperandLayout::Oihw:
                default:
                    return filterDims[0];
            }
        };

        // input
        auto inputOpIndex = mGraphNodeMap[conv2d->Inputs()[0].Get()];
        auto inputNodeInfo = mGraphOperandInfo[inputOpIndex];

        // output
        auto outputDims = conv2d->PrimaryOutput()->Shape();
        NodeInfo outputNode;
        outputNode.type = inputNodeInfo.type;
        outputNode.dimensions.resize(outputDims.size());

        for (size_t i = 0; i < outputDims.size(); i++) {
            outputNode.dimensions[i] = static_cast<uint32_t>(outputDims[i]);
        }

        DAWN_TRY(mNnapiMgr->CreateOperand(&outputNode));
        mGraphOperandInfo[outputNode.opIndex] = outputNode;
        mGraphNodeMap[conv2d->PrimaryOutput()] = outputNode.opIndex;

        // filter
        auto filterOpIndex = mGraphNodeMap[conv2d->Inputs()[1].Get()];
        auto filterNodeInfo = mGraphOperandInfo[filterOpIndex];

        // Check if transpose OP is needed
        uint32_t filterIndex;
        DAWN_TRY(AddTransposeImpl(filterNodeInfo, options->filterLayout, filterIndex));

        // bias
        uint32_t biasOpIndex = 0;
        if (options->bias == nullptr) {
            std::vector<float> biasMem(getOutputChannels(filterNodeInfo.dimensions), 0);

            NodeInfo biasNode;
            biasNode.type = inputNodeInfo.type;
            biasNode.dimensions = {
                static_cast<uint32_t>(getOutputChannels(filterNodeInfo.dimensions))};
            DAWN_TRY(mNnapiMgr->CreateOperandAndSetMemory("bias", &biasNode, &biasMem[0]));
            biasOpIndex = biasNode.opIndex;
        } else {
            biasOpIndex = mGraphNodeMap[conv2d->Inputs()[2].Get()];
        }

        bool isDepthwiseConv2d = false;
        {
            if (options->groups > 1) {
                int32_t inputChannels = 0;
                if (options->inputLayout == ml::InputOperandLayout::Nchw)
                    inputChannels = inputNodeInfo.dimensions[1];
                else if (options->inputLayout == ml::InputOperandLayout::Nhwc)
                    inputChannels = inputNodeInfo.dimensions[3];

                if (options->groups == inputChannels) {
                    int32_t filterChannels = 0;
                    switch (options->filterLayout) {
                        case ml::FilterOperandLayout::Oihw:
                        case ml::FilterOperandLayout::Ohwi:
                            filterChannels = static_cast<int32_t>(filterNodeInfo.dimensions[0]);
                            break;
                        case ml::FilterOperandLayout::Hwio:
                        case ml::FilterOperandLayout::Ihwo:
                            filterChannels = static_cast<int32_t>(filterNodeInfo.dimensions[3]);
                            break;
                        default:
                            break;
                    }

                    if (filterChannels == options->groups) {
                        isDepthwiseConv2d = true;
                    }
                }
            }
        }

        int32_t paddingLeft = options->padding ? options->padding[2] : 0;
        int32_t paddingRight = options->padding ? options->padding[3] : 0;
        int32_t paddingTop = options->padding ? options->padding[0] : 0;
        int32_t paddingBottom = options->padding ? options->padding[1] : 0;
        int32_t strideWidth = options->strides ? options->strides[1] : 0;
        int32_t strideHeight = options->strides ? options->strides[0] : 0;
        int32_t dilationsWidth = options->dilations ? options->dilations[1] : 0;
        int32_t dilationsHeight = options->dilations ? options->dilations[0] : 0;
        int8_t layout = (options->inputLayout == ml::InputOperandLayout::Nhwc) ? 0 : 1;
        int32_t groups = options->groups;
        uint32_t fuseOperation = 0;

        uint32_t paddingLeftWOp, paddingRightWOp, paddingTopHOp, paddingBottomHOp, strideWOp,
            strideHOp;
        uint32_t fuseOp, layoutOp, dilationWOp, dilationHOp, groupsOp;
        if (options->autoPad == ml::AutoPad::Explicit) {
            if (options->groups > 1 && !isDepthwiseConv2d) {
                int32_t groups = options->groups;
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingLeft,
                                                        paddingLeftWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingRight,
                                                        paddingRightWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingTop,
                                                        paddingTopHOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingBottom,
                                                        paddingBottomHOp));
                DAWN_TRY(
                    mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideWidth, strideWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideHeight,
                                                        strideHOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &groups, groupsOp));
                DAWN_TRY(
                    mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseOperation, fuseOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_BOOL, &layout, layoutOp));

                std::vector<uint32_t> inputList = {inputOpIndex,     filterIndex,     biasOpIndex,
                                                   paddingLeftWOp,   paddingRightWOp, paddingTopHOp,
                                                   paddingBottomHOp, strideWOp,       strideHOp,
                                                   groupsOp,         fuseOp,          layoutOp};

                DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_GROUPED_CONV_2D, inputList.size(),
                                                 inputList.data(), 1, &outputNode.opIndex));
            } else if (options->groups > 1 && !isDepthwiseConv2d) {
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingLeft,
                                                        paddingLeftWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingRight,
                                                        paddingRightWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingTop,
                                                        paddingTopHOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingBottom,
                                                        paddingBottomHOp));
                DAWN_TRY(
                    mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideWidth, strideWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideHeight,
                                                        strideHOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &groups, groupsOp));
                DAWN_TRY(
                    mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseOperation, fuseOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_BOOL, &layout, layoutOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &dilationsWidth,
                                                        dilationWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &dilationsHeight,
                                                        dilationHOp));

                std::vector<uint32_t> inputList = {
                    inputOpIndex,  filterIndex,      biasOpIndex, paddingLeftWOp, paddingRightWOp,
                    paddingTopHOp, paddingBottomHOp, strideWOp,   strideHOp,      groupsOp,
                    fuseOp,        layoutOp,         dilationWOp, dilationHOp};

                DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_DEPTHWISE_CONV_2D,
                                                 inputList.size(), inputList.data(), 1,
                                                 &outputNode.opIndex));
            } else {
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingLeft,
                                                        paddingLeftWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingRight,
                                                        paddingRightWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingTop,
                                                        paddingTopHOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &paddingBottom,
                                                        paddingBottomHOp));
                DAWN_TRY(
                    mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideWidth, strideWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &strideHeight,
                                                        strideHOp));
                DAWN_TRY(
                    mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &fuseOperation, fuseOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_BOOL, &layout, layoutOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &dilationsWidth,
                                                        dilationWOp));
                DAWN_TRY(mNnapiMgr->CreateScalarOperand(ANEURALNETWORKS_INT32, &dilationsHeight,
                                                        dilationHOp));

                std::vector<uint32_t> inputList = {
                    inputOpIndex,  filterIndex,      biasOpIndex, paddingLeftWOp, paddingRightWOp,
                    paddingTopHOp, paddingBottomHOp, strideWOp,   strideHOp,      fuseOp,
                    layoutOp,      dilationWOp,      dilationHOp};

                DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_CONV_2D, inputList.size(),
                                                 inputList.data(), 1, &outputNode.opIndex));
            }
        }

        return {};
    }

    MaybeError Graph::AddPad(const op::Pad* pad) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi AddPad"));
        return {};
    }

    MaybeError Graph::AddPool2d(const op::Pool2d* pool2d) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi AddPool2D"));
        return {};
    }

    MaybeError Graph::AddUnary(const op::Unary* unary) {
        auto inputOpIndex = mGraphNodeMap[unary->Inputs()[0].Get()];
        auto inputNodeInfo = mGraphOperandInfo[inputOpIndex];

        NodeInfo outputNode;
        outputNode.type = inputNodeInfo.type;
        auto outShape = unary->PrimaryOutput()->Shape();

        for (size_t i = 0; i < outShape.size(); i++) {
            outputNode.dimensions.push_back(static_cast<int32_t>(outShape[i]));
        }

        DAWN_TRY(mNnapiMgr->CreateOperand(&outputNode));
        mGraphOperandInfo[outputNode.opIndex] = outputNode;
        mGraphNodeMap[unary->PrimaryOutput()] = outputNode.opIndex;

        if (unary->GetType() == op::UnaryOpType::kSigmoid) {
            DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_LOGISTIC, 1, &inputOpIndex, 1,
                                             &outputNode.opIndex));
        } else {
            DAWN_TRY(
                CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi AddUnary unsupported operation"));
        }

        return {};
    }

    MaybeError Graph::AddReduce(const op::Reduce* reduce) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Reduce"));
        return {};
    }

    MaybeError Graph::AddResample2d(const op::Resample2d* resample) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Resample"));
        return {};
    }

    MaybeError Graph::AddReshape(const op::Reshape* reshape) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi Reshape"));
        return {};
#if 0
        auto inputOpIndex = mGraphNodeMap[reshape->Inputs()[0].Get()];
        auto inputNodeInfo = mGraphOperandInfo[inputOpIndex];

        NodeInfo newShapeNode, outputNode;
        newShapeNode.type = inputNodeInfo.type;
        newShapeNode.dimensions = {static_cast<uint32_t>(reshape->GetNewShape().size())};
 
        DAWN_TRY(mNnapiMgr->CreateOperandAndSetMemory("const", &newShapeNode, reshape->GetNewShape().data()));
        mGraphOperandInfo[newShapeNode.opIndex] = newShapeNode;

        outputNode.type = inputNodeInfo.type;
        auto tmpShape = reshape->PrimaryOutput()->Shape();
        for(size_t i=0; i < tmpShape.size(); i++) {
            outputNode.dimensions.push_back(static_cast<uint32_t>(tmpShape[i]));
        }
        DAWN_TRY(mNnapiMgr->CreateOperand(&outputNode));
        mGraphOperandInfo[outputNode.opIndex]=outputNode;
        mGraphNodeMap[reshape->PrimaryOutput()] = outputNode.opIndex;

        std::vector<uint32_t> inputList = {inputOpIndex, newShapeNode.opIndex};

        DAWN_TRY(mNnapiMgr->AddOperation(ANEURALNETWORKS_RESHAPE, inputList.size(), inputList.data(),
                                            1, &outputNode.opIndex));
        return {};
#endif
    }

    MaybeError Graph::AddSplit(const op::Split* split) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi split"));
        return {};
    }

    MaybeError Graph::AddSqueeze(const op::Squeeze* squeeze) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi squeeze"));
        return {};
    }

    MaybeError Graph::AddTranspose(const op::Transpose* transpose) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi transpose"));
        return {};
    }

    MaybeError Graph::AddConcat(const op::Concat* concat) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi concat"));
        return {};
    }

    MaybeError Graph::AddGemm(const op::Gemm* gemm) {
        DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_OP_FAILED, "nnapi gemm"));
        return {};
    }

    MaybeError Graph::Finish() {
        return {};
    }

    MaybeError Graph::CompileImpl() {
        return mNnapiMgr->Compile(mGraphInputs.size(), mGraphInputs.data(), mGraphOutputs.size(),
                                  mGraphOutputs.data());
    }

    MLComputeGraphStatus Graph::ComputeImpl(NamedInputsBase* inputs, NamedOutputsBase* outputs) {
        if (mNnapiMgr->InitExecutionContext() != MLComputeGraphStatus_Success)
            return MLComputeGraphStatus_Error;

        auto namedInputs = inputs->GetRecords();
        for (auto& input : mInputIdMap) {
            // All the inputs must be set.
            if (namedInputs.find(input.first) == namedInputs.end()) {
                dawn::ErrorLog() << "The input isn't set";
                return MLComputeGraphStatus_Error;
            }

            NodeInfo nodeInfo = input.second;
            size_t index = 0;
            for (; index < mGraphInputs.size(); index++) {
                if (mGraphInputs[index] == nodeInfo.opIndex)
                    break;
            }

            if (index == mGraphInputs.size()) {
                dawn::ErrorLog() << "Failed to find the input node in nodeinfo";
                return MLComputeGraphStatus_Error;
            }

            auto& resource = namedInputs[input.first]->resource;
            void* inputTensorPtr = reinterpret_cast<void*>(mmap(
                nullptr, resource.byteLength, PROT_READ | PROT_WRITE, MAP_SHARED, nodeInfo.fd, 0));
            std::memcpy(inputTensorPtr, static_cast<int8_t*>(resource.buffer) + resource.byteOffset,
                        resource.byteLength);
            munmap(inputTensorPtr, resource.byteLength);

            int32_t status =
                mNnapiMgr->SetInputMemory(index, nullptr, nodeInfo.mem, 0, resource.byteLength);
            if (status != ANEURALNETWORKS_NO_ERROR) {
                dawn::ErrorLog() << "Failed ANeuralNetworksExecution_setInputFromMemory";
                return MLComputeGraphStatus_Error;
            }
        }

        auto namedOutputs = outputs->GetRecords();
        for (auto& output : mOutputNameMap) {
            auto nodeInfo = mOutputNameMap[output.first];
            // All the inputs must be set.
            if (namedOutputs.find(output.first) == namedOutputs.end()) {
                dawn::ErrorLog() << "The output isn't set";
                return MLComputeGraphStatus_Error;
            }

            size_t index = 0;
            for (; index < mGraphOutputs.size(); index++) {
                if (mGraphOutputs[index] == nodeInfo.opIndex)
                    break;
            }

            if (index == mGraphOutputs.size()) {
                dawn::ErrorLog() << "Failed to find the input node in nodeinfo";
                return MLComputeGraphStatus_Error;
            }

            const ArrayBufferView* outputBuffer = namedOutputs[output.first];
            int32_t status = mNnapiMgr->SetOutputMemory(index, nullptr, nodeInfo.mem, 0,
                                                        outputBuffer->byteLength);
            if (status != ANEURALNETWORKS_NO_ERROR) {
                dawn::ErrorLog() << "Failed ANeuralNetworksExecution_setOutputFromMemory";
                return MLComputeGraphStatus_Error;
            }
        }

        if (mNnapiMgr->ComputeAndWait() != MLComputeGraphStatus_Success) {
            return MLComputeGraphStatus_Error;
        }

        for (auto namedOutput : outputs->GetRecords()) {
            const ArrayBufferView* output = namedOutput.second;
            DAWN_ASSERT(output->buffer != nullptr && output->byteLength != 0);
            // Get output id with friendly name.
            NodeInfo nodeInfo = mOutputNameMap[namedOutput.first];
            float* outputTensorPtr = reinterpret_cast<float*>(
                mmap(nullptr, output->byteLength, PROT_READ, MAP_SHARED, nodeInfo.fd, 0));
            if (outputTensorPtr == MAP_FAILED) {
                dawn::ErrorLog() << "Failed to mmap output buffer";
                return MLComputeGraphStatus_Error;
            }

            std::memcpy(static_cast<int8_t*>(output->buffer) + output->byteOffset, outputTensorPtr,
                        output->byteLength);

            munmap(outputTensorPtr, output->byteLength);
        }

        return MLComputeGraphStatus_Success;
    }
}}  // namespace webnn_native::nnapi
