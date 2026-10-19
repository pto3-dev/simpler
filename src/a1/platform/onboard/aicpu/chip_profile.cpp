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

#include "chip_profile_wire.h"

namespace simpler::a1 {
namespace {

uint64_t Tick() {
    uint64_t value;
    asm volatile("isb\n\tmrs %0, cntvct_el0\n\tisb" : "=r"(value)::"memory");
    return value;
}

void Publish(ChipProfileFlag &flag, uint64_t value) {
    flag.value = value;
    asm volatile("dc cvac, %0\n\tdsb sy" ::"r"(&flag) : "memory");
}

bool Observe(ChipProfileFlag *flags, uint32_t cores, uint64_t desired, uint64_t *times, uint64_t deadline) {
    uint32_t seen = 0;
    const uint32_t all = (1U << cores) - 1;
    while (seen != all && Tick() < deadline) {
        for (uint32_t core = 0; core < cores; ++core) {
            if (seen & (1U << core)) continue;
            asm volatile("dc civac, %0\n\tdsb sy" ::"r"(&flags[core]) : "memory");
            const uint64_t value = flags[core].value;
            if (value == kChipProfileAbort) return false;
            if (value >= desired) {
                times[core] = Tick();
                seen |= 1U << core;
            }
        }
    }
    return seen == all;
}

void Flush(void *begin, uint64_t bytes) {
    for (uint64_t offset = 0; offset < bytes; offset += 64) {
        auto address = reinterpret_cast<uint64_t>(begin) + offset;
        asm volatile("dc cvac, %0\n\tdsb sy" ::"r"(address) : "memory");
    }
}

}  // namespace
}  // namespace simpler::a1

extern "C" uint32_t simpler_aicpu_init(void *) { return 0; }

extern "C" uint32_t simpler_aicpu_exec(void *payload) {
    using simpler::a1::ChipProfileFlag;
    using simpler::a1::ChipProfileHeader;
    using simpler::a1::ChipProfileRecord;
    auto *wire = reinterpret_cast<ChipProfileHeader *>(*static_cast<uint64_t *>(payload));
    using namespace simpler::a1;
    if (wire->version != kChipProfileVersion || wire->tasks == 0 || wire->tasks > UINT32_MAX / 2 || wire->cores == 0 ||
        wire->cores > kChipProfileCores || wire->timeout_us == 0 || wire->timeout_us > 1000000) {
        wire->status = 3;
        Flush(wire, sizeof(*wire));
        return 0;
    }
    asm volatile("mrs %0, cntfrq_el0" : "=r"(wire->frequency));
    const uint64_t deadline = Tick() + wire->frequency * wire->timeout_us / 1000000;
    auto flags = [wire](uint64_t offset) {
        return reinterpret_cast<ChipProfileFlag *>(reinterpret_cast<uint8_t *>(wire) + offset);
    };
    auto *commands = flags(ChipCommandsOffset());
    auto *ready = flags(ChipReadyOffset(wire->tasks));
    auto *states = flags(ChipStatesOffset(wire->tasks));
    auto *hello = flags(ChipHelloOffset(wire->tasks, wire->cores));
    auto *records = reinterpret_cast<ChipProfileRecord *>(
        reinterpret_cast<uint8_t *>(wire) + ChipRecordsOffset(wire->tasks, wire->cores)
    );
    uint64_t hello_observed[kChipProfileCores]{};
    bool success = Observe(hello, wire->cores, 1, hello_observed, deadline);
    if (success) {
        records[0].dispatch_begin = Tick();
        Publish(commands[0], 1);
        records[0].dispatch_end = Tick();
    }
    for (uint32_t task = 0; task < wire->tasks && success; ++task) {
        auto &record = records[task];
        success = Observe(states, wire->cores, uint64_t(task) * 2 + 1, record.received, deadline);
        if (!success) break;
        record.released = Tick();
        Publish(ready[task], 1);
        if (task + 1 < wire->tasks) {
            auto &next = records[task + 1];
            next.dispatch_begin = Tick();
            Publish(commands[task + 1], 1);
            next.dispatch_end = Tick();
        }
        success = Observe(states, wire->cores, uint64_t(task) * 2 + 2, record.finished, deadline);
    }
    wire->status = success ? 1 : 2;
    if (!success) {
        for (uint32_t task = 0; task < wire->tasks; ++task) {
            Publish(commands[task], kChipProfileAbort);
            Publish(ready[task], kChipProfileAbort);
        }
    }
    Flush(wire, sizeof(*wire));
    Flush(records, uint64_t(wire->tasks) * sizeof(ChipProfileRecord));
    return 0;
}
