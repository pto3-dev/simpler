# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Causal-order and clock semantics of A1 observation export."""

import copy

import pytest

from simpler_setup.tools.a1_chip_swimlane import convert


def capture():
    return {
        "schema_version": 1,
        "timestamp_kind": "aicpu_observed_envelopes",
        "status": 1,
        "frequency": 1000000,
        "raw_core_ids": [5, 2],
        "labels": ["add", "mul"],
        "task_blocks": [2, 1],
        "tasks": [
            {
                "dispatch_begin": 1000000010,
                "dispatch_end": 1000000012,
                "released": 1000000020,
                "received": [1000000014, 1000000016],
                "finished": [1000000120, 1000000124],
            },
            {
                "dispatch_begin": 1000000021,
                "dispatch_end": 1000000023,
                "released": 1000000130,
                "received": [1000000125, 1000000127],
                "finished": [1000000180, 1000000134],
            },
        ],
    }


def test_physical_lanes_epoch_frequency_and_inactive_blocks():
    trace, report = convert(capture())
    events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    first = next(event for event in events if event["name"] == "add" and event["tid"] == 5)
    assert first["ts"] == 10 and first["dur"] == 100
    assert first["args"]["timestamp_kind"] == "aicpu_observed_envelopes"
    assert any(event["tid"] == 2 and event["args"].get("active_block") is False for event in events)
    assert report["observed_span_ms"] == pytest.approx(0.170)
    assert report["observed_envelope_ms"] == pytest.approx(0.154)
    assert report["observed_between_envelope_ms"] == pytest.approx(0.016)
    assert report["barrier_cpm_path"] == ["0", "1"]
    assert report["early_queue_publications"] == 1


def test_quantized_equal_timestamps_are_not_invented():
    data = capture()
    data["tasks"][0]["dispatch_end"] = data["tasks"][0]["dispatch_begin"]
    trace, _ = convert(data)
    event = next(event for event in trace["traceEvents"] if event.get("name") == "dispatch")
    assert event["dur"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("frequency", 0),
        ("frequency", True),
        ("frequency", "38400000"),
        ("status", 2),
        ("schema_version", 2),
        ("timestamp_kind", "aicore_instruction_timestamps"),
        ("raw_core_ids", [5, 5]),
        ("raw_core_ids", [8, 2]),
        ("task_blocks", [2, 3]),
    ],
)
def test_rejects_ambiguous_or_incomplete_captures(field, value):
    data = capture()
    data[field] = value
    with pytest.raises(ValueError):
        convert(data)


@pytest.mark.parametrize(
    "mutation",
    [
        "receive_before_dispatch",
        "release_before_receive",
        "finish_before_release",
        "dependency_before_finish",
        "missing_core",
        "overflow",
    ],
)
def test_rejects_noncausal_or_malformed_records(mutation):
    data = copy.deepcopy(capture())
    if mutation == "receive_before_dispatch":
        data["tasks"][0]["received"][0] = 0
    elif mutation == "release_before_receive":
        data["tasks"][0]["released"] = 0
    elif mutation == "finish_before_release":
        data["tasks"][0]["finished"][0] = 0
    elif mutation == "dependency_before_finish":
        data["tasks"][1]["released"] = 1000000123
    elif mutation == "missing_core":
        data["tasks"][0]["finished"].pop()
    else:
        data["tasks"][0]["dispatch_begin"] = 2**64
    with pytest.raises(ValueError):
        convert(data)
