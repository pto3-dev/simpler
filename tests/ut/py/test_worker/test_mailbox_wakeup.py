# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Idle mailbox work must release the GIL without losing a published request."""

import mmap
import multiprocessing
import struct
import time

import pytest
from _task_interface import _mailbox_notify_work, _mailbox_wait_i32_result_for_test
from simpler import worker


class Parked(Exception):
    pass


class PollingBudgetExceeded(Exception):
    pass


def test_idle_consumer_parks_with_a_publisher_wakeup_contract(monkeypatch):
    buf = memoryview(bytearray(worker.mailbox_size(1)))
    struct.pack_into("i", buf, worker.MAILBOX_OFF_WAKE_CAPABILITY, 1)
    addr = worker._buffer_field_addr(buf, 0)
    original = worker._mailbox_load_i32
    reads = 0

    def load(address):
        nonlocal reads
        reads += 1
        if reads > 20000:
            raise PollingBudgetExceeded
        return original(address)

    def park(*_args):
        raise Parked

    monkeypatch.setattr(worker, "_mailbox_load_i32", load)
    monkeypatch.setattr(worker, "_mailbox_wait_i32", park, raising=False)
    try:
        with pytest.raises(Parked):
            worker._run_mailbox_loop(
                buf, addr, handle_task=lambda _frame, _entry: (0, ""), handle_control=lambda _: (0, "")
            )
    finally:
        buf.release()


def test_notification_before_wait_cannot_be_lost():
    buf = memoryview(bytearray(worker.mailbox_size(1)))
    try:
        base = worker._buffer_field_addr(buf, 0)
        address = base + worker.MAILBOX_OFF_WAKE_SEQUENCE
        expected = worker._mailbox_load_i32(address)
        _mailbox_notify_work(base)
        result = _mailbox_wait_i32_result_for_test(address, expected, 0.5)
        assert result == "value_mismatch"
    finally:
        buf.release()


def _consumer_process(shared, connection):
    buf = memoryview(shared)
    original = worker._mailbox_wait_i32

    def park(*args):
        connection.send("park")
        return original(*args)

    worker._mailbox_wait_i32 = park

    def task(_frame, _entry):
        connection.send("task")
        return 0, ""

    def control(command):
        connection.send(("control", command))
        return 0, ""

    try:
        worker._run_mailbox_loop(
            buf,
            worker._buffer_field_addr(buf, 0),
            handle_task=task,
            handle_control=control,
            on_shutdown=lambda: connection.send("shutdown"),
        )
    finally:
        buf.release()
        connection.close()


def _receive(connection):
    assert connection.poll(3), "Mailbox consumer did not wake before watchdog"
    return connection.recv()


def _await_state(address, expected):
    deadline = time.monotonic() + 2
    while worker._mailbox_load_i32(address) != expected:
        if time.monotonic() > deadline:
            pytest.fail("Mailbox state not published before watchdog")


def test_cross_process_task_control_and_shutdown_wake_parked_consumer():
    shared = mmap.mmap(-1, worker.mailbox_size(1))
    buf = memoryview(shared)
    base = worker._buffer_field_addr(buf, 0)
    _mailbox_notify_work(base)
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    process = context.Process(target=_consumer_process, args=(shared, child))
    process.start()
    child.close()
    try:
        assert _receive(parent) == "park"
        task = base + worker.MAILBOX_FRAME_SIZE
        entry = base + worker.MAILBOX_RUN_ENTRY_TABLE_OFFSET
        worker._mailbox_store_u64(entry + worker._OFF_ENTRY_GENERATION, 3)
        struct.pack_into("=Q", buf, worker.MAILBOX_FRAME_SIZE + worker._OFF_FRAME_ENTRY_INDEX, 0)
        struct.pack_into("=Q", buf, worker.MAILBOX_FRAME_SIZE + worker._OFF_FRAME_ENTRY_GENERATION, 3)
        struct.pack_into("=Q", buf, worker.MAILBOX_FRAME_SIZE + worker._OFF_FRAME_GENERATION, 1)
        worker._mailbox_store_i32(task, worker._TASK_READY)
        _mailbox_notify_work(base)
        assert _receive(parent) == "task"
        _await_state(entry + worker._OFF_ENTRY_STATE, worker._TASK_DONE)
        assert _receive(parent) == "park"
        struct.pack_into("Q", buf, worker._OFF_CALLABLE, 1234)
        worker._mailbox_store_i32(base, worker._CONTROL_REQUEST)
        _mailbox_notify_work(base)
        assert _receive(parent) == ("control", 1234)
        _await_state(base, worker._CONTROL_DONE)
        assert _receive(parent) == "park"
        worker._request_child_shutdown(buf)
        assert _receive(parent) == "shutdown"
        process.join(timeout=3)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=3)
        parent.close()
        buf.release()
        shared.close()


def test_legacy_publisher_without_capability_keeps_parent_liveness(monkeypatch):
    buf = memoryview(bytearray(worker.mailbox_size(1)))
    calls = 0

    def parent():
        nonlocal calls
        calls += 1
        return 100 if calls == 1 else 101

    def no_park(*_args):
        pytest.fail("A legacy publisher cannot wake a blocked consumer")

    monkeypatch.setattr(worker.os, "getppid", parent)
    monkeypatch.setattr(worker, "_mailbox_wait_i32", no_park)
    try:
        worker._run_mailbox_loop(
            buf,
            worker._buffer_field_addr(buf, 0),
            handle_task=lambda _frame, _entry: (0, ""),
            handle_control=lambda _: (0, ""),
        )
        assert calls == 2
    finally:
        buf.release()
