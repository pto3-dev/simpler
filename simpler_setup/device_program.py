# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile synchronous leaf invocations into a position-independent device plan.

The caller owns the union of tensor access directions and intermediate lifetimes
until the outer task completes. Each invocation returns with its pipelines drained;
no deferred completion may escape. A grid must fit unique resident cluster positions.
Epochs increase strictly between non-overlapping replays; buffers are zeroed once.
"""

import struct
from dataclasses import dataclass

LANES = ("aic", "aiv0", "aiv1")
MAGIC = 0x44505247


@dataclass(frozen=True)
class LeafSignature:
    func_id: int
    lane: str
    argument_count: int
    synchronous: bool = True


@dataclass(frozen=True)
class Invocation:
    name: str
    func_id: int
    lane: str
    owner: int
    logical_block: int
    logical_count: int
    arguments: tuple[int, ...]
    after: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeviceProgram:
    image: bytes
    event_names: tuple[str, ...]
    grid: int

    @property
    def frame_bytes(self):
        return self.grid * len(LANES) * 512

    @property
    def signal_bytes(self):
        return len(self.event_names) * 128


def _validated_streams(nodes, leaves, grid, root_arg_count, epoch_arg):
    names = {n.name: i for i, n in enumerate(nodes)}
    if not nodes or len(names) != len(nodes) or any(not n.name for n in nodes):
        raise ValueError("Invocation names must be nonempty and unique")
    streams = {(lane, owner): [] for lane in LANES for owner in range(grid)}
    dependencies = {}
    for n in nodes:
        leaf = leaves.get(n.func_id)
        if (
            leaf is None
            or not 0 <= n.func_id < 2**32
            or n.lane not in LANES
            or leaf.lane != ("aic" if n.lane == "aic" else "aiv")
            or not leaf.synchronous
        ):
            raise ValueError("Unregistered, incompatible or asynchronous leaf")
        if not 0 <= n.owner < grid or not 0 <= n.logical_block < n.logical_count < 2**31:
            raise ValueError("Invalid physical owner or logical block geometry")
        if len(n.arguments) != leaf.argument_count or len(n.arguments) > 48:
            raise ValueError("Leaf argument count mismatch")
        if any(not 4 <= i < root_arg_count or i == epoch_arg for i in n.arguments):
            raise ValueError("Leaf binding overlaps control or context arguments")
        if len(set(n.after)) != len(n.after) or any(a not in names for a in n.after):
            raise ValueError("Unknown or duplicate tile dependency")
        dependencies[n.name] = set(n.after)
        stream = streams[n.lane, n.owner]
        if stream:
            dependencies[n.name].add(stream[-1].name)
        stream.append(n)
    successors = {n.name: [] for n in nodes}
    degree = {name: len(deps) for name, deps in dependencies.items()}
    for name, deps in dependencies.items():
        for dep in deps:
            successors[dep].append(name)
    ready = [name for name, count in degree.items() if count == 0]
    for name in ready:
        for child in successors[name]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if len(ready) != len(nodes):
        raise ValueError("Tile dependencies and physical lane order contain a cycle")
    return names, streams


def compile_device_program(
    invocations,
    signatures,
    *,
    grid,
    available_clusters,
    root_arg_count,
    epoch_arg,
    max_wait_cycles=1_000_000,
    record_trace=False,
):
    """Validate tile dependencies plus physical lane order before serializing.

    Input order fixes the sequence on each (lane, owner). Cross-lane dependencies
    name individual invocations. Cycle detection includes implicit lane order,
    rejecting a wait on work queued behind the waiter on the same physical core.
    available_clusters comes from the actual launch, not the backend capacity.
    """
    nodes = tuple(invocations)
    signatures = tuple(signatures)
    leaves = {s.func_id: s for s in signatures}
    if len(leaves) != len(signatures):
        raise ValueError("Duplicate registered func_id")
    if not isinstance(grid, int) or not 0 < grid <= available_clusters:
        raise ValueError("Grid exceeds available resident cluster positions")
    if not 5 <= root_arg_count <= 48 or not 4 <= epoch_arg < root_arg_count:
        raise ValueError("Invalid root argument prefix or epoch binding")
    if not 0 < max_wait_cycles < 2**32:
        raise ValueError("Wait budget must fit uint32 and be positive")
    names, streams = _validated_streams(nodes, leaves, grid, root_arg_count, epoch_arg)
    order = [n for stream in streams.values() for n in stream]
    events = {n.name: i for i, n in enumerate(order)}
    ranges = []
    cursor = 0
    for stream in streams.values():
        ranges.extend((cursor, cursor + len(stream)))
        cursor += len(stream)
    bindings = []
    waits = []
    records = []
    for n in order:
        # Same-lane predecessors have completed through program order.
        deps = [a for a in n.after if (nodes[names[a]].lane, nodes[names[a]].owner) != (n.lane, n.owner)]
        records.extend(
            (
                n.func_id,
                LANES.index(n.lane),
                n.owner,
                n.logical_block,
                n.logical_count,
                len(bindings),
                len(n.arguments),
                len(waits),
                len(deps),
                events[n.name],
                0,
                0,
                0,
                0,
                0,
                0,
            )
        )
        bindings.extend(n.arguments)
        waits.extend(events[a] for a in deps)
    ro = 64
    so = (ro + 4 * len(ranges) + 63) // 64 * 64
    bo = so + 64 * len(order)
    wo = bo + 4 * len(bindings)
    size = wo + 4 * len(waits)
    if size >= 2**32:
        raise ValueError("Device image exceeds uint32 offsets")
    image = bytearray(size)
    header = (
        MAGIC,
        1,
        size,
        grid,
        len(order),
        len(bindings),
        len(waits),
        len(order),
        ro,
        so,
        bo,
        wo,
        epoch_arg,
        root_arg_count,
        max_wait_cycles,
        int(record_trace),
    )
    struct.pack_into("<16I", image, 0, *header)
    for offset, values in ((ro, ranges), (so, records), (bo, bindings), (wo, waits)):
        struct.pack_into("<" + "I" * len(values), image, offset, *values)
    return DeviceProgram(bytes(image), tuple(n.name for n in order), grid)
