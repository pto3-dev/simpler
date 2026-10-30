/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "src/a1/platform/onboard/host/single_launch.cpp"

int main() {
    HostTensorArg arg{};
    arg.address = 4096;
    arg.rank = 2;
    arg.shape[0] = 2;
    arg.shape[1] = 64;
    for (const auto type : {DataType::FLOAT32, DataType::FLOAT16, DataType::INT32, DataType::UINT32}) {
        arg.dtype = static_cast<uint32_t>(type);
        arg.bytes = 128 * (type == DataType::FLOAT16 ? 2 : 4);
        try {
            Validate(arg);
        } catch (const std::invalid_argument &) {
            return 1;
        }
    }
    auto rejects = [](HostTensorArg candidate) {
        try {
            Validate(candidate);
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    for (const auto type : {DataType::INT16, DataType::BFLOAT16, DataType::INT64, DataType::UINT16, DataType::UINT64}) {
        auto candidate = arg;
        candidate.dtype = static_cast<uint32_t>(type);
        if (!rejects(candidate)) {
            return 2;
        }
    }
    auto candidate = arg;
    candidate.dtype = UINT32_MAX;
    if (!rejects(candidate)) {
        return 3;
    }
    candidate = arg;
    candidate.bytes -= 4;
    if (!rejects(candidate)) {
        return 4;
    }
    candidate = arg;
    candidate.address = UINT64_MAX - candidate.bytes + 1;
    if (!rejects(candidate)) {
        return 5;
    }
    candidate = arg;
    candidate.rank = 0;
    if (!rejects(candidate)) {
        return 6;
    }
    candidate = arg;
    candidate.rank = MAX_TENSOR_DIMS + 1;
    if (!rejects(candidate)) {
        return 7;
    }
    candidate = arg;
    candidate.shape[0] = 0;
    if (!rejects(candidate)) {
        return 8;
    }
    candidate = arg;
    candidate.shape[0] = candidate.shape[1] = 65536;
    return rejects(candidate) ? 0 : 9;
}
