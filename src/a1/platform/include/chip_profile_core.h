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

#include "chip_profile_wire.h"

namespace simpler::a1 {

AICORE inline __gm__ ChipProfileFlag *ChipFlags(__gm__ ChipProfileHeader *wire, uint64_t offset) {
    return reinterpret_cast<__gm__ ChipProfileFlag *>(reinterpret_cast<__gm__ uint8_t *>(wire) + offset);
}

AICORE inline void ChipPublish(__gm__ ChipProfileFlag &flag, uint64_t value) {
    flag.value = value;
    asm volatile("" ::: "memory");
    dcci(&flag, 0);
    asm volatile("" ::: "memory");
}

AICORE inline bool ChipWait(__gm__ ChipProfileFlag &flag) {
    for (uint32_t poll = 0; poll < 1000000; ++poll) {
        dcci(&flag, 0);
        asm volatile("" ::: "memory");
        uint64_t value = flag.value;
        if (value == kChipProfileAbort) return false;
        if (value == 1) return true;
    }
    return false;
}

}  // namespace simpler::a1
