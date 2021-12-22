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
#include <sys/mman.h>

#include <errno.h>
#include <string.h>

#include "NnapiManager.h"
#include "common/Assert.h"

namespace webnn_native { namespace nnapi {

    NnapiManager::NnapiManager() : operandIndex(0) {
        mNnapi = NnApiImplementation();

        // TODO: Move this code
        if (mNnapi != nullptr) {
            mNnapi->ANeuralNetworksModel_create(&mNnModel);
        }

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

        mScalarFloat32Operand.type = ANEURALNETWORKS_FLOAT32;
        mScalarFloat32Operand.dimensionCount = 0;
        mScalarFloat32Operand.dimensions = nullptr;
        mScalarFloat32Operand.scale = 0.0f;
        mScalarFloat32Operand.zeroPoint = 0;
    }

    MaybeError NnapiManager::SetVecOperand(int32_t index, const void* buffer, size_t length) {
        int32_t status =
            mNnapi->ANeuralNetworksModel_setOperandValue(mNnModel, index, buffer, length);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_setOperandValueFromMemory failed"));
        return {};
    }

    MaybeError NnapiManager::CreateOperandAndSetMemory(std::string name,
                                                       NodeInfo* node,
                                                       const void* buffer) {
        uint32_t totalBytes = node->GetByteCount();

        uint32_t operandIndex = GetOperandIdx();
        name = name + std::to_string(operandIndex);
        int fd = mNnapi->ASharedMemory_create(name.c_str(), totalBytes);
        DAWN_TRY(CheckStatusCode(fd == -1 ? ANEURALNETWORKS_OP_FAILED : ANEURALNETWORKS_NO_ERROR,
                                 "ASharedMemory_create failed"));  // use different error code

        ANeuralNetworksMemory* nnMemory;
        int32_t status = mNnapi->ANeuralNetworksMemory_createFromFd(
            totalBytes, PROT_READ | PROT_WRITE, fd, 0, &nnMemory);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksMemory_createFromFd failed"));

        /** Copy the buffer data to NNAPI operand **/
        void* inputTensorPtr = reinterpret_cast<void*>(
            mmap(nullptr, totalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (inputTensorPtr == MAP_FAILED) {
            DAWN_TRY(CheckStatusCode(ANEURALNETWORKS_UNMAPPABLE,
                                     "Failed to mmap memory for input tensor"));
        }

        std::memcpy(inputTensorPtr, buffer, totalBytes);
        munmap(inputTensorPtr, totalBytes);

        ANeuralNetworksOperandType tensorType;
        GetTensorDesc(node, tensorType);
        status = mNnapi->ANeuralNetworksModel_addOperand(mNnModel, &tensorType);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_addOperand failed"));

        status = mNnapi->ANeuralNetworksModel_setOperandValueFromMemory(mNnModel, operandIndex,
                                                                        nnMemory, 0, totalBytes);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_setOperandValueFromMemory failed"));

        node->name = name;
        node->fd = fd;
        node->mem = nnMemory;
        node->opIndex = operandIndex;

        return {};
    }

    size_t NnapiManager::SetInputMemory(int32_t index,
                                        const ANeuralNetworksOperandType* type,
                                        const ANeuralNetworksMemory* memory,
                                        size_t offset,
                                        size_t length) {
        int32_t status = mNnapi->ANeuralNetworksExecution_setInputFromMemory(
            mNnExecution, index, nullptr, memory, 0, length);
        return status;
    }

    size_t NnapiManager::SetOutputMemory(int32_t index,
                                         const ANeuralNetworksOperandType* type,
                                         const ANeuralNetworksMemory* memory,
                                         size_t offset,
                                         size_t length) {
        int32_t status = mNnapi->ANeuralNetworksExecution_setOutputFromMemory(
            mNnExecution, index, nullptr, memory, 0, length);
        return status;
    }

    MaybeError NnapiManager::CreateScalarOperand(uint32_t type,
                                                 const void* data,
                                                 uint32_t& index,
                                                 bool optional) {
        ANeuralNetworksOperandType nnOpType;
        size_t opSize = 1;

        switch (type) {
            case ANEURALNETWORKS_BOOL:
                nnOpType = mScalarBoolOperand;
                opSize = sizeof(int8_t);
                break;
            case ANEURALNETWORKS_INT32:
                nnOpType = mScalarInt32Operand;
                opSize = sizeof(int32_t);
                break;
            case ANEURALNETWORKS_FLOAT32:
                nnOpType = mScalarFloat32Operand;
                opSize = sizeof(float);
                break;
            default:
                return DAWN_UNIMPLEMENTED_ERROR("Unsupported scalar type !!!");
        }

        index = GetOperandIdx();
        int32_t status = mNnapi->ANeuralNetworksModel_addOperand(mNnModel, &nnOpType);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_addOperand failed"));

        if (!optional) {
            status = mNnapi->ANeuralNetworksModel_setOperandValue(mNnModel, index, data, opSize);
            DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_setOperandValue failed"));
        } else {
            status = mNnapi->ANeuralNetworksModel_setOperandValue(mNnModel, index, nullptr, 0);
            DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_setOperandValue failed"));
        }

        return {};
    }

    MaybeError NnapiManager::CreateInputOutputOperand(std::string name,
                                                      NodeInfo* node,
                                                      bool input) {
        int32_t status, fd;
        ANeuralNetworksMemory* nnMemory = nullptr;

        if (input) {
            uint32_t operandIndex = GetOperandIdx();
            ANeuralNetworksOperandType tensorType;
            GetTensorDesc(node, tensorType);
            status = mNnapi->ANeuralNetworksModel_addOperand(mNnModel, &tensorType);
            DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_addOperand failed"));
            node->opIndex = operandIndex;

            name = name + std::to_string(operandIndex);
            fd = mNnapi->ASharedMemory_create(name.c_str(), node->GetByteCount());
            DAWN_TRY(
                CheckStatusCode(fd == -1 ? ANEURALNETWORKS_OP_FAILED : ANEURALNETWORKS_NO_ERROR,
                                "ASharedMemory_create failed"));  // use different error code

            status = mNnapi->ANeuralNetworksMemory_createFromFd(
                node->GetByteCount(), PROT_READ | PROT_WRITE, fd, 0, &nnMemory);
            DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksMemory_createFromFd failed"));
        } else {
            auto byteCount = node->GetByteCount();
            fd = mNnapi->ASharedMemory_create(name.c_str(), byteCount);
            DAWN_TRY(
                CheckStatusCode(fd == -1 ? ANEURALNETWORKS_OP_FAILED : ANEURALNETWORKS_NO_ERROR,
                                "ASharedMemory_create failed"));  // use different error code

            status = mNnapi->ANeuralNetworksMemory_createFromFd(byteCount, PROT_READ | PROT_WRITE,
                                                                fd, 0, &nnMemory);
            DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksMemory_createFromFd failed"));
        }

        node->name = name;
        node->fd = fd;
        node->mem = nnMemory;

        return {};
    }

    MaybeError NnapiManager::CreateOperand(NodeInfo* node) {
        uint32_t operandIndex = GetOperandIdx();
        ANeuralNetworksOperandType tensorType;
        GetTensorDesc(node, tensorType);
        int32_t status = mNnapi->ANeuralNetworksModel_addOperand(mNnModel, &tensorType);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_addOperand failed"));

        node->opIndex = operandIndex;
        return {};
    }

    MaybeError NnapiManager::AddOperation(int32_t opCode,
                                          size_t inputLen,
                                          const uint32_t* input,
                                          size_t outputLen,
                                          const uint32_t* output) {
        int32_t status = mNnapi->ANeuralNetworksModel_addOperation(mNnModel, opCode, inputLen,
                                                                   input, outputLen, output);
        return CheckStatusCode(status, "ANeuralNetworksModel_addOperand failed");
    }

    MaybeError NnapiManager::Compile(uint32_t inputCount,
                                     const uint32_t* inputs,
                                     uint32_t outputCount,
                                     const uint32_t* outputs) {
        int32_t status = mNnapi->ANeuralNetworksModel_identifyInputsAndOutputs(
            mNnModel, inputCount, inputs, outputCount, outputs);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_identifyInputsAndOutputs failed"));
        status = mNnapi->ANeuralNetworksModel_finish(mNnModel);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksModel_finish failed"));
        status = mNnapi->ANeuralNetworksCompilation_create(mNnModel, &mNnCompilation);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksCompilation_create failed"));
        status = mNnapi->ANeuralNetworksCompilation_setPreference(
            mNnCompilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksCompilation_setPreference failed"));
        status = mNnapi->ANeuralNetworksCompilation_finish(mNnCompilation);
        DAWN_TRY(CheckStatusCode(status, "ANeuralNetworksCompilation_finish failed"));
        return {};
    }

    MLComputeGraphStatus NnapiManager::InitExecutionContext() {
        int32_t status = mNnapi->ANeuralNetworksExecution_create(mNnCompilation, &mNnExecution);
        if (status != ANEURALNETWORKS_NO_ERROR) {
            return MLComputeGraphStatus_Error;
        }

        return MLComputeGraphStatus_Success;
    }

    MLComputeGraphStatus NnapiManager::ComputeAndWait() {
        ANeuralNetworksEvent* event = nullptr;
        int32_t status = mNnapi->ANeuralNetworksExecution_startCompute(mNnExecution, &event);
        if (status != ANEURALNETWORKS_NO_ERROR) {
            return MLComputeGraphStatus_Error;
        }

        status = mNnapi->ANeuralNetworksEvent_wait(event);
        if (status != ANEURALNETWORKS_NO_ERROR) {
            return MLComputeGraphStatus_Error;
        }

        mNnapi->ANeuralNetworksEvent_free(event);
        mNnapi->ANeuralNetworksExecution_free(mNnExecution);

        return MLComputeGraphStatus_Success;
    }
}}  // namespace webnn_native::nnapi
