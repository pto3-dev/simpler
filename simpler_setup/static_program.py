# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile fixed per-core kernel sequences into one Simpler dispatch boundary.

Leaf sources must use the PyPTO int64 argument-vector ABI. Each leaf owns its
on-chip arena until it returns with all engines drained. No asynchronous or
external completion may escape a leaf. Inter-core dependencies require explicit
region-owned, zero-initialized signal storage and strictly increasing epochs.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class StaticStep:
    name: str
    source: Path
    arguments: tuple[int, ...]
    block_index: Optional[int] = None
    block_count: Optional[int] = None


def _leaf(step: StaticStep, index: int) -> str:
    if not re.fullmatch(r"[A-Za-z_]\w*", step.name):
        raise ValueError("Step names must be C++ identifiers")
    if any(i < 0 or i >= 48 for i in step.arguments):
        raise ValueError("Argument bindings must refer to the user argument prefix")
    if (step.block_index is None) != (step.block_count is None):
        raise ValueError("A specialized logical block requires both index and count")
    if step.block_count is not None and (step.block_index is None or not 0 <= step.block_index < step.block_count):
        raise ValueError("Invalid logical block identity")
    text = step.source.read_text()
    marker = "// --- Kernel entry point ---"
    if text.count(marker) != 1:
        raise ValueError("Expected exactly one generated PyPTO entry point")
    body, entry = text.split(marker)
    if "defer_completion" in text or "tprefetch_async" in text:
        raise ValueError("Escaping asynchronous work cannot cross a static step boundary")
    required = {int(x) for x in re.findall(r"args\[(\d+)\]", entry) if int(x) < 48}
    if required != set(range(len(step.arguments))):
        raise ValueError(f"Unbound or non-contiguous leaf arguments: {required}")
    entry = re.sub(r"args\[(\d+)\]", lambda m: f"args[{step.arguments[int(m[1])]}]" if int(m[1]) < 48 else m[0], entry)
    if step.block_count is not None:
        entry = entry.replace("get_block_idx(args)", str(step.block_index))
        entry = entry.replace("get_block_num(args)", str(step.block_count))
    # Includes are outside namespaces; helper definitions remain leaf-local.
    includes = re.findall(r"^#include[^\n]*", body, re.M)
    body = re.sub(r"^#include[^\n]*", "", body, flags=re.M)
    entry = entry.replace('extern "C" ', "").replace("kernel_entry(", f"entry_{index}(")
    return "\n".join(includes) + f"\nnamespace static_leaf_{index} {{\n" + body + entry + "\n}\n"


def _assemble(leaves: list[str], entry: str) -> str:
    includes = list(dict.fromkeys(line for text in leaves for line in re.findall(r"^#include[^\n]*", text, re.M)))
    indices = []
    for text in leaves:
        match = re.search(r"namespace static_leaf_(\d+)", text)
        if match is None:
            raise ValueError("Missing static leaf namespace")
        indices.append(int(match[1]))
    forward = [
        f"namespace static_leaf_{i} {{ __aicore__ __attribute__((always_inline)) void entry_{i}(__gm__ int64_t*); }}"
        for i in indices
    ]
    # AICore payload entry is at byte zero; out-of-line leaf bodies follow it.
    return (
        "#ifndef __gm__\n#define __gm__\n#endif\n#ifndef __aicore__\n"
        "#if defined(__CPU_SIM)\n#define __aicore__\n#else\n#define __aicore__ [aicore]\n#endif\n#endif\n"
        + "\n".join(includes + forward)
        + "\n"
        + entry
        + "\n"
        + "\n".join(leaves)
    )


def render_static_sequence(steps: tuple[StaticStep, ...]) -> str:
    """Expand same-core steps; one dispatch/ACK/FIN encloses the whole sequence.

    There are no per-step task descriptors or scheduler queue operations. The
    caller declares the union of all tensor access directions on the region,
    keeps intermediate tensors alive to region completion, and proves every
    step's inputs ready. A logical block specialization executes on the region's
    existing physical core; it does not launch another worker.
    """
    if not steps:
        raise ValueError("A static sequence must contain at least one step")
    if len({step.name for step in steps}) != len(steps):
        raise ValueError("Static step names must be unique")
    leaves = [_leaf(step, i) for i, step in enumerate(steps)]
    calls = "\n".join(f"    static_leaf_{i}::entry_{i}(args);" for i in range(len(steps)))
    return _assemble(
        leaves,
        '\nextern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {\n'
        + calls
        + "\n}\n",
    )


@dataclass(frozen=True)
class StaticNode:
    step: StaticStep
    lane: str
    after: tuple[str, ...] = ()
    dependency_scope: str = "all_blocks"


def plan_static_region(nodes: tuple[StaticNode, ...]) -> tuple[StaticNode, ...]:
    """Validate a fork/join region and establish acyclic per-lane order."""
    by_name = {node.step.name: node for node in nodes}
    if not nodes or len(by_name) != len(nodes):
        raise ValueError("Region nodes must be nonempty and uniquely named")
    if any(node.lane not in ("aic", "aiv0", "aiv1") for node in nodes):
        raise ValueError("Unsupported execution lane")
    if any(node.dependency_scope not in ("all_blocks", "same_block") for node in nodes):
        raise ValueError("Dependency scope must be all_blocks or same_block")
    successors = {name: [] for name in by_name}
    pending = {}
    for node in nodes:
        if len(set(node.after)) != len(node.after):
            raise ValueError("Duplicate dependency")
        pending[node.step.name] = len(node.after)
        for dep in node.after:
            if dep not in by_name:
                raise ValueError(f"Unknown predecessor: {dep}")
            successors[dep].append(node.step.name)
    ready = [node.step.name for node in nodes if pending[node.step.name] == 0]
    ordered = []
    for name in ready:
        ordered.append(by_name[name])
        for consumer in successors[name]:
            pending[consumer] -= 1
            if pending[consumer] == 0:
                ready.append(consumer)
    if len(ordered) != len(nodes):
        raise ValueError("Static region has a dependency cycle")
    return tuple(ordered)


def render_static_region(
    nodes: tuple[StaticNode, ...],
    lane: str,
    *,
    state_arg: int,
    epoch_arg: int,
    max_wait_cycles: int = 1_000_000,
    trace_arg: Optional[int] = None,
) -> str:
    """Expand per-lane code, preserving all-block cross-lane dependency events.

    state_arg is an INT64 tensor with shape [node_count, grid, 16], zeroed once
    before first use. Each event has its own 128-byte line and a single writer.
    epoch_arg is a positive, strictly increasing scalar for each replay; the
    outer scheduler boundary fences all participants before state reuse.
    same_block dependencies require matching producer/consumer tile ownership;
    same-lane dependencies of that scope are implicit in instruction order.
    trace_arg optionally supplies [node_count, grid, 16] INT64 storage. Its first
    five fields are wait-entry, kernel-start, kernel-end, physical-core, epoch.
    No deferred/async leaf work or external consumers are permitted internally.
    Cross-block waits require a forward-progress proof: use unique resident-capable
    core positions with independently progressing predecessors, or sync_start
    admission. Ordinary dynamic MIX may dispatch waves onto the same core; a
    global join can otherwise wait on a queued peer.
    """
    ordered = plan_static_region(nodes)
    if max_wait_cycles <= 0:
        raise ValueError("A bounded wait budget is required")
    bindings = [state_arg, epoch_arg] + ([] if trace_arg is None else [trace_arg])
    if len(set(bindings)) != len(bindings) or any(i < 0 or i >= 48 for i in bindings):
        raise ValueError("Region state, epoch and trace bindings must be distinct user arguments")
    if any(i in node.step.arguments for node in ordered for i in bindings):
        raise ValueError("Control arguments must not alias leaf argument bindings")
    if lane not in ("aic", "aiv0", "aiv1") or state_arg == epoch_arg:
        raise ValueError("Invalid static region arguments")
    by_name = {node.step.name: i for i, node in enumerate(ordered)}
    leaves = []
    body = []
    for i, node in enumerate(ordered):
        if node.lane != lane:
            continue
        if node.step.block_count is not None:
            raise ValueError("Cross-core regions require a shared runtime grid")
        leaves.append(_leaf(node.step, i))
        if trace_arg is not None:
            body.append(
                f"    auto* trace_{i} = traces + ({i} * grid + block) * 16;\n    trace_{i}[0] = get_sys_cnt_aicore();"
            )
        for dep in node.after:
            producer = by_name[dep]
            if node.dependency_scope == "same_block" and ordered[producer].lane == lane:
                continue
            peer_begin, peer_end = ("block", "block + 1") if node.dependency_scope == "same_block" else ("0", "grid")
            body.append(f"""    for (int peer = {peer_begin}; peer < {peer_end}; ++peer) {{
        auto* event = events + ({producer} * grid + peer) * 16;
        const uint64_t wait_start = get_sys_cnt_aicore();
        do {{
            if (static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL ||
                get_sys_cnt_aicore() - wait_start > {max_wait_cycles}) {{
                auto ctx = get_async_ctx(args);
                pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
                pto2::detail::defer_flush(ctx);
                return;
            }}
            dcci((__gm__ void*)event, cache_line_t::SINGLE_CACHE_LINE);
        }} while (*event != epoch);
    }}""")
        if trace_arg is not None:
            body.append(f"    trace_{i}[1] = get_sys_cnt_aicore();")
        end_stamp = f"    trace_{i}[2] = get_sys_cnt_aicore();\n" if trace_arg is not None else ""
        body.append(f"""    static_leaf_{i}::entry_{i}(args);
    pipe_barrier(PIPE_ALL);
{end_stamp}
    auto* done_{i} = events + ({i} * grid + block) * 16;
    dsb(DSB_DDR);
    *done_{i} = epoch;
    dcci((__gm__ void*)done_{i}, SINGLE_CACHE_LINE, CACHELINE_OUT);""")
        if trace_arg is not None:
            body.append(
                f"    trace_{i}[3] = get_physical_core_id();\n    trace_{i}[4] = epoch;\n"
                f"    dsb(DSB_DDR);\n    dcci((__gm__ void*)trace_{i}, SINGLE_CACHE_LINE, CACHELINE_OUT);"
            )
    if not leaves:
        raise ValueError("The requested lane has no work")
    trace_binding = ""
    if trace_arg is not None:
        trace_binding = (
            f"    auto* trace_desc = reinterpret_cast<__gm__ Tensor*>(args[{trace_arg}]);\n"
            "    auto* traces = reinterpret_cast<__gm__ int64_t*>(trace_desc->buffer.addr)\n"
            "        + trace_desc->start_offset;\n"
        )
    return '#include "aicore/aicore.h"\n#include "async_kernel_api.h"\n' + _assemble(
        leaves,
        f"""\nextern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {{
    const int block = get_block_idx(args);
    const int grid = get_block_num(args);
    auto* descriptor = reinterpret_cast<__gm__ Tensor*>(args[{state_arg}]);
    auto* events = reinterpret_cast<__gm__ volatile int64_t*>(descriptor->buffer.addr) + descriptor->start_offset;
    const int64_t epoch = args[{epoch_arg}];
"""
        + trace_binding
        + "\n".join(body)
        + "\n}\n",
    )
