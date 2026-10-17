# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export A1 AICPU observations without claiming AICore instruction timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .critical_path import RankGraph, analyze_rank


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value < 2**64:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**64)")
    return value


def _metadata(record: dict[str, Any]) -> tuple[int, list[int], list[dict[str, Any]], list[str], list[int]]:
    if record.get("schema_version") != 1 or record.get("timestamp_kind") != "aicpu_observed_envelopes":
        raise ValueError("requires version 1 AICPU-observed envelope schema")
    if record.get("status") != 1:
        raise ValueError("capture did not complete successfully")
    frequency = _integer(record.get("frequency"), "frequency", minimum=1)
    physical = record.get("raw_core_ids")
    if not isinstance(physical, list) or not 1 <= len(physical) <= 8:
        raise ValueError("requires one through eight physical core IDs")
    for core in physical:
        _integer(core, "physical core ID")
        if core >= 8:
            raise ValueError("physical core ID exceeds the A1 observation protocol")
    if len(set(physical)) != len(physical):
        raise ValueError("physical core IDs must be unique")
    tasks, labels, blocks = record.get("tasks"), record.get("labels"), record.get("task_blocks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("capture contains no tasks")
    if (
        not isinstance(labels, list)
        or not isinstance(blocks, list)
        or len(tasks) != len(labels)
        or len(tasks) != len(blocks)
    ):
        raise ValueError("task, label and block-count tables must match")
    if any(not isinstance(label, str) for label in labels):
        raise ValueError("task labels must be strings")
    for count in blocks:
        if not 1 <= _integer(count, "task block count") <= len(physical):
            raise ValueError("task block count exceeds the launched core grid")
    return frequency, physical, tasks, labels, blocks


def _validate_task(task: dict[str, Any], cores: int, previous: dict[str, Any] | None) -> None:
    if not isinstance(task, dict):
        raise ValueError("task records must be objects")
    for key in ("dispatch_begin", "dispatch_end", "released"):
        _integer(task.get(key), key)
    for key in ("received", "finished"):
        values = task.get(key)
        if not isinstance(values, list) or len(values) != cores:
            raise ValueError(f"{key} must contain one observation per core")
        for value in values:
            _integer(value, key)
    if not task["dispatch_begin"] <= task["dispatch_end"] <= min(task["received"]):
        raise ValueError("dispatch/receipt observations are not time-forward")
    if max(task["received"]) > task["released"] or task["released"] > min(task["finished"]):
        raise ValueError("receipt/release/finish observations are not time-forward")
    if previous is not None and (
        previous["released"] > task["dispatch_begin"] or max(previous["finished"]) > task["released"]
    ):
        raise ValueError("sequential dependency barrier is not time-forward")


def convert(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return Perfetto and an observed-envelope critical-path report."""
    frequency, physical, tasks, labels, blocks = _metadata(record)
    for index, task in enumerate(tasks):
        _validate_task(task, len(physical), tasks[index - 1] if index else None)
    origin = tasks[0]["dispatch_begin"]
    events: list[dict[str, Any]] = []

    def event(name: str, lane: int, start: int, end: int, task: int, **metadata: Any) -> None:
        events.append(
            {
                "ph": "X",
                "pid": 1,
                "tid": lane,
                "name": name,
                "ts": (start - origin) * 1e6 / frequency,
                "dur": (end - start) * 1e6 / frequency,
                "args": {"task": task, "timestamp_kind": "aicpu_observed_envelopes", **metadata},
            }
        )

    events.append(
        {
            "ph": "M",
            "pid": 1,
            "name": "process_name",
            "args": {"name": "A1 native observations (instrumented diagnostic replay)"},
        }
    )
    for core in physical:
        events.append(
            {
                "ph": "M",
                "pid": 1,
                "tid": core,
                "name": "thread_name",
                "args": {"name": f"Physical core {core}: observed envelopes"},
            }
        )
    events.append(
        {
            "ph": "M",
            "pid": 1,
            "tid": 100,
            "name": "thread_name",
            "args": {"name": "AICPU dispatch / early queue publication"},
        }
    )
    for index, task in enumerate(tasks):
        event(
            "dispatch" if index == 0 else "early_dispatch (queue publication)",
            100,
            task["dispatch_begin"],
            task["dispatch_end"],
            index,
        )
        for block, core in enumerate(physical):
            event("observed dependency wait", core, task["received"][block], task["released"], index)
            event(
                labels[index] if block < blocks[index] else "inactive block / barrier observation",
                core,
                task["released"],
                task["finished"][block],
                index,
                active_block=block < blocks[index],
                block_index=block,
            )
    ids = [str(index) for index in range(len(tasks))]
    start = {key: tasks[i]["released"] for i, key in enumerate(ids)}
    end = {key: max(tasks[i]["finished"]) for i, key in enumerate(ids)}
    predecessors = {key: [ids[i - 1]] if i else [] for i, key in enumerate(ids)}
    graph = RankGraph(
        frequency,
        start,
        end,
        {key: end[key] - start[key] for key in ids},
        dict(zip(ids, labels, strict=True)),
        dict(zip(ids, blocks, strict=True)),
        predecessors,
        {key: (ids[i - 1], end[ids[i - 1]]) for i, key in enumerate(ids) if i},
        origin,
        max(end.values()),
        len(ids) - 1,
        "A1 observed sequential task barriers",
    )
    result = analyze_rank(graph, tol=0)
    scale = 1000 / frequency
    report = {
        "timestamp_kind": "aicpu_observed_envelopes",
        "frequency": frequency,
        "tasks": len(tasks),
        "core_observations": len(tasks) * len(physical),
        "physical_core_ids": physical,
        "observed_span_ms": result.makespan * scale,
        "observed_envelope_ms": result.compute_total * scale,
        "observed_between_envelope_ms": result.stall_total * scale,
        "barrier_cpm_ms": result.cpm_len * scale,
        "barrier_cpm_tasks": result.cpm_ntasks,
        "barrier_cpm_path": result.cpm_path,
        "dispatch_total_us": sum(task["dispatch_end"] - task["dispatch_begin"] for task in tasks) * 1e6 / frequency,
        "early_queue_publications": len(tasks) - 1,
        "mean_publication_to_previous_completion_observation_us": (
            sum(max(tasks[i - 1]["finished"]) - tasks[i]["dispatch_end"] for i in range(1, len(tasks)))
            * 1e6
            / frequency
            / (len(tasks) - 1)
            if len(tasks) > 1
            else None
        ),
        "interpretation": "Envelopes include coherence, polling, barriers and collection overhead. "
        "The CPM describes the instrumented sequential barrier implementation, not "
        "a compute-only lower bound. Early publication does not prove core-side prefetch "
        "or instruction-level overlap. CANN task HWTS is a separate timing scope.",
    }
    return {"traceEvents": events, "displayTimeUnit": "ms", "a1_observation_report": report}, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    trace, report = convert(json.loads(args.capture.read_text()))
    args.output.write_text(json.dumps(trace) + "\n")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
