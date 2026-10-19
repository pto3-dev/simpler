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

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace simpler::a1 {

constexpr uint32_t kChipProfileVersion = 1;
constexpr uint32_t kChipProfileCores = 8;
constexpr uint64_t kChipProfileAbort = UINT64_MAX;

struct ChipProfileFlag {
    volatile uint64_t value;
    uint64_t padding[15];
};

struct ChipProfileHeader {
    uint32_t version;
    uint32_t tasks;
    uint32_t cores;
    uint32_t reserved;
    uint64_t frequency;
    uint64_t status;
    uint64_t timeout_us;
    uint64_t padding[11];
};

struct ChipProfileRecord {
    uint64_t dispatch_begin;
    uint64_t dispatch_end;
    uint64_t released;
    uint64_t received[kChipProfileCores];
    uint64_t finished[kChipProfileCores];
};

constexpr uint64_t ChipCommandsOffset() { return sizeof(ChipProfileHeader); }
constexpr uint64_t ChipReadyOffset(uint32_t tasks) {
    return ChipCommandsOffset() + uint64_t(tasks) * sizeof(ChipProfileFlag);
}
constexpr uint64_t ChipStatesOffset(uint32_t tasks) {
    return ChipReadyOffset(tasks) + uint64_t(tasks) * sizeof(ChipProfileFlag);
}
constexpr uint64_t ChipHelloOffset(uint32_t tasks, uint32_t cores) {
    return ChipStatesOffset(tasks) + uint64_t(cores) * sizeof(ChipProfileFlag);
}
constexpr uint64_t ChipRecordsOffset(uint32_t tasks, uint32_t cores) {
    return ChipHelloOffset(tasks, cores) + uint64_t(cores) * sizeof(ChipProfileFlag);
}
constexpr uint64_t ChipProfileBytes(uint32_t tasks, uint32_t cores) {
    return ChipRecordsOffset(tasks, cores) + uint64_t(tasks) * sizeof(ChipProfileRecord);
}

static_assert(sizeof(ChipProfileHeader) == 128 && sizeof(ChipProfileFlag) == 128);
static_assert(sizeof(ChipProfileRecord) == 152);
static_assert(std::is_standard_layout_v<ChipProfileHeader> && std::is_trivially_copyable_v<ChipProfileHeader>);
static_assert(std::is_standard_layout_v<ChipProfileFlag> && std::is_trivially_copyable_v<ChipProfileFlag>);
static_assert(std::is_standard_layout_v<ChipProfileRecord> && std::is_trivially_copyable_v<ChipProfileRecord>);

}  // namespace simpler::a1
