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

#pragma once

#include <runtime/rt.h>
#include <runtime/rt_preload_task.h>

#include <cstdint>

namespace simpler::a1 {

struct L2CacheInfo {
    uint64_t alias_offset;
    int32_t query_status;
};

// The caller owns and selects the device context. Failed queries must not
// publish a possibly written offset, and zero is an explicit no-op alias.
inline L2CacheInfo QueryL2CacheOffset(uint32_t device, decltype(&rtGetL2CacheOffset) query = rtGetL2CacheOffset) {
    uint64_t offset = 0;
    const int32_t status = query(device, &offset);
    return {status == RT_ERROR_NONE ? offset : 0, status};
}

}  // namespace simpler::a1
