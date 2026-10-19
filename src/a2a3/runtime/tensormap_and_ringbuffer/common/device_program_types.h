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
#include <stdint.h>
#include <type_traits>

namespace simpler::device_program {
constexpr uint32_t kMagic = 0x44505247;
constexpr uint32_t kVersion = 1;
constexpr uint32_t kSignalStride = 128;

struct Header {
    uint32_t magic, version, byte_size, grid;
    uint32_t step_count, binding_count, wait_count, event_count;
    uint32_t ranges_offset, steps_offset, bindings_offset, waits_offset;
    uint32_t epoch_arg, root_arg_count, max_wait_cycles, flags;
};
struct Step {
    uint32_t func_id, lane, owner, logical_block;
    uint32_t logical_count, binding_begin, binding_count, wait_begin;
    uint32_t wait_count, event_index, reserved[6];
};
struct Range {
    uint32_t begin, end;
};
static_assert(sizeof(Header) == 64 && sizeof(Step) == 64 && sizeof(Range) == 8);
static_assert(std::is_trivially_copyable_v<Header> && std::is_standard_layout_v<Header>);
static_assert(std::is_trivially_copyable_v<Step> && std::is_standard_layout_v<Step>);
static_assert(std::is_trivially_copyable_v<Range> && std::is_standard_layout_v<Range>);
}  // namespace simpler::device_program
