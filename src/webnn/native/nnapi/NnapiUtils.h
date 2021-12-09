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

#ifndef WEBNN_NATIVE_NNAPI_UTILS_H_
#define WEBNN_NATIVE_NNAPI_UTILS_H_
#include <functional>
#include <numeric>

#include "webnn_native/Error.h"
#include "webnn_native/Graph.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Operator.h"

#include "NeuralNetworksTypes.h"
#include "nnapi_implementation.h"
#include "webnn_native/nnapi/ErrorNnapi.h"

namespace webnn_native { namespace nnapi {

    struct NodeInfo {
        int fd;
        ANeuralNetworksMemory* mem;
        ml::OperandType type;
        std::vector<uint32_t> dimensions;
        std::string name;
        uint32_t opIndex;

        NodeInfo() {
            fd = -1;
            mem = nullptr;
        }

        size_t getDimsSize() {
            size_t count = std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                                           std::multiplies<size_t>());
            return count;
        }

        size_t GetByteCount() {
            size_t count = std::accumulate(std::begin(dimensions), std::end(dimensions), 1,
                                           std::multiplies<size_t>());

            switch (type) {
                case ml::OperandType::Float32:
                case ml::OperandType::Uint32:
                case ml::OperandType::Int32:
                    count *= 4;
                    break;
                case ml::OperandType::Float16:
                    count *= 2;
                    break;
                default:
                    UNREACHABLE();
            }

            return count;
        }
    };

    inline int32_t ConvertToNnapiType(ml::OperandType type) {
        int32_t nnapiType;

        switch (type) {
            case ml::OperandType::Float32:
                nnapiType = ANEURALNETWORKS_TENSOR_FLOAT32;
                break;
            case ml::OperandType::Int32:
                nnapiType = ANEURALNETWORKS_TENSOR_INT32;
                break;
            case ml::OperandType::Float16:
                nnapiType = ANEURALNETWORKS_TENSOR_FLOAT16;
                break;
            case ml::OperandType::Uint32:
                nnapiType = ANEURALNETWORKS_UINT32;
                break;
            default:
                UNREACHABLE();
        }

        return nnapiType;
    }

    inline void GetTensorDesc(NodeInfo* node, ANeuralNetworksOperandType& tensorType) {
        tensorType.dimensions = &(node->dimensions[0]);
        tensorType.dimensionCount = node->dimensions.size();
        tensorType.scale = 0.0f;
        tensorType.zeroPoint = 0;
        tensorType.type = ConvertToNnapiType(node->type);
    }

    inline void CreateNodeFromOperandDescriptor(const OperandDescriptor* desc, NodeInfo* node) {
        node->fd = -1;
        node->mem = nullptr;
        node->type = desc->type;

        for (uint32_t i = 0; i < desc->dimensionsCount; i++)
            node->dimensions.push_back(static_cast<uint32_t>(desc->dimensions[i]));
    }
}}  // namespace webnn_native::nnapi

#endif
