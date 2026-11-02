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

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "src/a1/platform/onboard/host/single_launch.cpp"

namespace {
uint64_t query_offset = 0;
rtError_t query_status = RT_ERROR_NONE;
uint32_t query_calls = 0;

rtError_t FakeQuery(uint32_t device, uint64_t *offset) {
    assert(device == 17);
    ++query_calls;
    *offset = query_offset;
    return query_status;
}

void CheckQueries() {
    for (const auto status : {RT_ERROR_NONE, static_cast<rtError_t>(1), static_cast<rtError_t>(507015)}) {
        for (const uint64_t offset : {uint64_t{0}, uint64_t{1} << 40, UINT64_MAX}) {
            query_status = status;
            query_offset = offset;
            const auto info = QueryL2CacheOffset(17, FakeQuery);
            assert(info.query_status == status);
            assert(info.alias_offset == (status == RT_ERROR_NONE ? offset : 0));
        }
    }
    assert(query_calls == 9);
    assert(simpler_a1_argument_abi_version() == 2);
}

void CheckSlots() {
    const uint64_t base = 4096, alias = uint64_t{1} << 40;
    const uint32_t counts[] = {1, 2, 1};
    const auto slots = ArgumentSlots(base, counts, 3, 2, alias);
    const std::vector<uint64_t> expected = {base, alias, base + 128, base + 256, alias, base + 384, alias};
    assert(slots == expected);
    const auto legacy = ArgumentSlots(base, counts, 3, 1, alias);
    const std::vector<uint64_t> legacy_expected = {base, base + 128, base + 256, base + 384};
    assert(legacy == legacy_expected);
    for (const uint32_t tensors : {uint32_t{1}, uint32_t{48}, uint32_t{64}}) {
        const auto single = ArgumentSlots(base, &tensors, 1, 2, UINT64_MAX);
        assert(single.size() == tensors + 1);
        assert(single.back() == UINT64_MAX);
        for (uint32_t index = 0; index < tensors; ++index)
            assert(single[index] == base + index * sizeof(KernelTensor));
    }
    const std::vector<uint32_t> maximum(4096, 64);
    const auto large = ArgumentSlots(base, maximum.data(), maximum.size(), 2, 0);
    assert(large.size() == 4096 * 65);
    for (uint64_t call = 0; call < maximum.size(); ++call) {
        for (uint64_t index = 0; index < 64; ++index)
            assert(large[call * 65 + index] == base + (call * 64 + index) * sizeof(KernelTensor));
        assert(large[call * 65 + 64] == 0);
    }
}

void CheckBounds() {
    const uint32_t valid[] = {1}, zero[] = {0}, excessive[] = {65};
    const auto rejects = [](uint64_t base, const uint32_t *counts, uint32_t calls) {
        try {
            ArgumentSlots(base, counts, calls, 2, 0);
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    assert(rejects(4096, nullptr, 1));
    assert(rejects(4096, valid, 0));
    assert(rejects(4096, valid, 4097));
    assert(rejects(4096, zero, 1));
    assert(rejects(4096, excessive, 1));
    assert(rejects(0, valid, 1));
    assert(rejects(UINT64_MAX - sizeof(KernelTensor) + 1, valid, 1));
    const auto boundary = ArgumentSlots(UINT64_MAX - sizeof(KernelTensor), valid, 1, 2, 7);
    assert(boundary[0] == UINT64_MAX - sizeof(KernelTensor));
    assert(boundary[1] == 7);
}
}  // namespace

int main(int argc, char **argv) {
    assert(argc == 2);
    switch (argv[1][0]) {
    case 'q':
        CheckQueries();
        break;
    case 's':
        CheckSlots();
        break;
    case 'b':
        CheckBounds();
        break;
    default:
        return 1;
    }
}
