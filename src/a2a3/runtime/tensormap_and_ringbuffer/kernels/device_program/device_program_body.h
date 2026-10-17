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

using namespace simpler::device_program;
struct alignas(128) ProgramFrame {
    int64_t arguments[50];
    LocalContext local;
};
static_assert(sizeof(ProgramFrame) == 512);
static_assert(std::is_trivially_copyable_v<ProgramFrame> && std::is_standard_layout_v<ProgramFrame>);

static __aicore__ __attribute__((always_inline)) void program_error(__gm__ int64_t *args) {
    auto ctx = get_async_ctx(args);
    pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    pto2::detail::defer_flush(ctx);
}

static __aicore__ __attribute__((always_inline)) void run_program(__gm__ int64_t *args, uint32_t lane) {
    auto *pt = reinterpret_cast<__gm__ Tensor *>(args[0]);
    if (pt->dtype != DataType::UINT8 || pt->ndims != 1 || !pt->is_contiguous || pt->start_offset != 0 ||
        pt->shapes[0] < sizeof(Header)) {
        program_error(args);
        return;
    }
    auto *base = reinterpret_cast<__gm__ uint8_t *>(pt->buffer.addr);
    auto *h = reinterpret_cast<__gm__ Header *>(base);
    uint32_t owner = get_block_idx(args);
    auto *global = reinterpret_cast<__gm__ GlobalContext *>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
    if (h->magic != kMagic || h->version != kVersion || h->grid != get_block_num(args) || owner >= h->grid ||
        h->root_arg_count > SPMD_LOCAL_CONTEXT_INDEX || h->epoch_arg < 4 || h->epoch_arg >= h->root_arg_count ||
        global->callable_entry_table == 0) {
        program_error(args);
        return;
    }
    const uint64_t steps_offset = (sizeof(Header) + uint64_t(h->grid) * 3 * sizeof(Range) + 63) / 64 * 64;
    const uint64_t bindings_offset = steps_offset + uint64_t(h->step_count) * sizeof(Step);
    const uint64_t waits_offset = bindings_offset + uint64_t(h->binding_count) * sizeof(uint32_t);
    const uint64_t byte_size = waits_offset + uint64_t(h->wait_count) * sizeof(uint32_t);
    if (h->ranges_offset != sizeof(Header) || h->steps_offset != steps_offset ||
        h->bindings_offset != bindings_offset || h->waits_offset != waits_offset || h->byte_size != byte_size ||
        byte_size > pt->shapes[0] || byte_size > pt->buffer.size || h->event_count != h->step_count ||
        h->max_wait_cycles == 0 || h->flags > 1) {
        program_error(args);
        return;
    }
    const bool record_trace = (h->flags == 1);
    const int64_t epoch = args[h->epoch_arg];
    if (epoch <= 0) {
        program_error(args);
        return;
    }
    auto *ft = reinterpret_cast<__gm__ Tensor *>(args[1]);
    if (ft->dtype != DataType::UINT8 || ft->ndims != 1 || !ft->is_contiguous || ft->start_offset != 0 ||
        uint64_t(h->grid) * 3 * sizeof(ProgramFrame) > ft->shapes[0] || ft->shapes[0] > ft->buffer.size ||
        ft->buffer.addr % alignof(ProgramFrame) != 0) {
        program_error(args);
        return;
    }
    auto *frames = reinterpret_cast<__gm__ ProgramFrame *>(ft->buffer.addr);
    auto *frame = frames + lane * h->grid + owner;
    auto *et = reinterpret_cast<__gm__ Tensor *>(args[2]);
    if (et->dtype != DataType::INT64 || et->ndims != 1 || !et->is_contiguous || et->start_offset != 0 ||
        uint64_t(h->event_count) * kSignalStride > uint64_t(et->shapes[0]) * sizeof(int64_t) ||
        uint64_t(et->shapes[0]) * sizeof(int64_t) > et->buffer.size || et->buffer.addr % kSignalStride != 0) {
        program_error(args);
        return;
    }
    auto *events = reinterpret_cast<__gm__ uint8_t *>(et->buffer.addr);
    auto *tt = reinterpret_cast<__gm__ Tensor *>(args[3]);
    if (record_trace &&
        (tt->dtype != DataType::INT64 || tt->ndims != 1 || !tt->is_contiguous || tt->start_offset != 0 ||
         uint64_t(h->step_count) * kSignalStride > uint64_t(tt->shapes[0]) * sizeof(int64_t) ||
         uint64_t(tt->shapes[0]) * sizeof(int64_t) > tt->buffer.size || tt->buffer.addr % kSignalStride != 0)) {
        program_error(args);
        return;
    }
    auto *traces = reinterpret_cast<__gm__ uint8_t *>(tt->buffer.addr);
    auto *ranges = reinterpret_cast<__gm__ Range *>(base + h->ranges_offset);
    auto *range = ranges + lane * h->grid + owner;
    if (range->begin > range->end || range->end > h->step_count) {
        program_error(args);
        return;
    }
    auto *steps = reinterpret_cast<__gm__ Step *>(base + h->steps_offset);
    auto *bindings = reinterpret_cast<__gm__ uint32_t *>(base + h->bindings_offset);
    auto *waits = reinterpret_cast<__gm__ uint32_t *>(base + h->waits_offset);
    auto *entries = reinterpret_cast<__gm__ uint64_t *>(global->callable_entry_table);
    using Kernel = void (*)(__gm__ int64_t *);
    for (uint32_t index = range->begin; index < range->end; ++index) {
        auto *s = steps + index;
        if (s->lane != lane || s->owner != owner || s->func_id >= global->callable_entry_count ||
            s->logical_count == 0 || s->logical_block >= s->logical_count || s->logical_count >= (1u << 31) ||
            s->binding_count > SPMD_LOCAL_CONTEXT_INDEX || s->binding_begin > h->binding_count ||
            s->binding_count > h->binding_count - s->binding_begin || s->wait_begin > h->wait_count ||
            s->wait_count > h->wait_count - s->wait_begin || s->event_index != index || entries[s->func_id] == 0) {
            program_error(args);
            return;
        }
        auto *trace = reinterpret_cast<volatile __gm__ int64_t *>(traces + index * kSignalStride);
        if (record_trace) trace[0] = get_sys_cnt_aicore();
        for (uint32_t w = 0; w < s->wait_count; ++w) {
            uint32_t event_index = waits[s->wait_begin + w];
            if (event_index >= h->event_count) {
                program_error(args);
                return;
            }
            auto *event = reinterpret_cast<volatile __gm__ int64_t *>(events + event_index * kSignalStride);
            const uint64_t begin = get_sys_cnt_aicore();
            do {
                if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL ||
                    get_sys_cnt_aicore() - begin > h->max_wait_cycles) {
                    program_error(args);
                    return;
                }
                dcci((__gm__ void *)event, cache_line_t::SINGLE_CACHE_LINE);
            } while (*event != epoch);
        }
        for (uint32_t a = 0; a < s->binding_count; ++a) {
            auto binding = bindings[s->binding_begin + a];
            if (binding < 4 || binding >= h->root_arg_count || binding == h->epoch_arg) {
                program_error(args);
                return;
            }
            frame->arguments[a] = args[binding];
        }
        auto *parent = reinterpret_cast<__gm__ LocalContext *>(args[SPMD_LOCAL_CONTEXT_INDEX]);
        frame->local.async_ctx.completion_count = parent->async_ctx.completion_count;
        frame->local.async_ctx.completion_error_code = parent->async_ctx.completion_error_code;
        frame->local.async_ctx.completion_entries = parent->async_ctx.completion_entries;
        frame->local.async_ctx.completion_capacity = parent->async_ctx.completion_capacity;
        TaskId::assign(frame->local.async_ctx.task_token, parent->async_ctx.task_token);
        frame->local.block_idx = s->logical_block;
        frame->local.block_num = s->logical_count;
        frame->arguments[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&frame->local);
        frame->arguments[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(global);
        if (record_trace) trace[1] = get_sys_cnt_aicore();
        // Registered entries use the same int64-vector call ABI as the outer executor.
        reinterpret_cast<Kernel>(entries[s->func_id])(frame->arguments);
        // Synchronous leaf entries return with all engines drained.
        if (record_trace) {
            trace[2] = get_sys_cnt_aicore();
            trace[3] = get_physical_core_id();
            trace[4] = epoch;
            dcci((__gm__ void *)trace, SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        dsb(DSB_DDR);
        auto *own = reinterpret_cast<volatile __gm__ int64_t *>(events + s->event_index * kSignalStride);
        *own = epoch;
        dcci((__gm__ void *)own, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb(DSB_DDR);
    }
}
