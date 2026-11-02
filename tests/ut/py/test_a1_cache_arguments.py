# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A1 cache-offset transport without loading a driver or launching a kernel."""

import ctypes
import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest

from simpler_setup import A1L2CacheInfo, a1_single_launch


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    a1_single_launch.env_manager.ensure("ASCEND_HOME_PATH")
    toolchain = a1_single_launch.GxxToolchain()
    assert toolchain.ascend_home_path is not None
    root, sdk = Path(__file__).resolve().parents[3], Path(toolchain.ascend_home_path)
    common = root / "src/common"
    binary = tmp_path_factory.mktemp("a1-cache-probe") / "probe"
    source = Path(__file__).parent / "support/a1_cache_argument_probe.cpp"
    subprocess.run(
        [
            toolchain.cxx_path,
            "-std=c++17",
            "-O1",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            f"-I{root}",
            f"-I{sdk / 'include'}",
            f"-I{sdk / 'pkg_inc'}",
            f"-I{sdk / 'pkg_inc/runtime'}",
            f"-I{sdk / (platform.machine() + '-linux') / 'pkg_inc'}",
            f"-I{sdk / (platform.machine() + '-linux') / 'pkg_inc/runtime'}",
            f"-I{common}",
            f"-I{common / 'task_interface'}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    needed = subprocess.run(["readelf", "-d", str(binary)], check=True, capture_output=True, text=True).stdout
    assert "libascendcl" not in needed and "libruntime" not in needed
    return binary


@pytest.mark.parametrize("mode", ["queries", "slots", "bounds"])
def test_native_query_and_argument_contract(native_probe, mode):
    subprocess.run([str(native_probe), mode], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("offset,status", [(0, 0), (1 << 40, 0), (0, 507015)])
def test_cache_info_reads_session_state_and_preserves_status(monkeypatch, offset, status):
    class Native:
        reads = 0

        def simpler_a1_create_v2(self, _image, _size, _device, handle):
            handle._obj.value = 123
            return 0

        def simpler_a1_l2_cache_info(self, handle, result, query_status):
            assert handle.value == 123
            result._obj.value = offset
            query_status._obj.value = status
            self.reads += 1
            return 0

        def simpler_a1_close(self, _handle):
            return 0

    native = Native()
    monkeypatch.setattr(a1_single_launch, "_load_runtime", lambda: native)
    with a1_single_launch.A1SingleLaunch(b"\x7fELF", argument_abi_version=2) as runner:
        assert runner.argument_abi_version == 2
        first = runner.l2_cache_info
        assert isinstance(first, A1L2CacheInfo)
        assert first == (offset, status)
        assert runner.l2_cache_info == first
        assert native.reads == 2
        with pytest.raises(AttributeError):
            setattr(first, "alias_offset", 1)
    with pytest.raises(RuntimeError, match="closed"):
        _ = runner.l2_cache_info
    assert native.reads == 2


@pytest.mark.parametrize("version", [0, 3, True, "2", None])
def test_invalid_argument_abi_fails_before_runtime_load(monkeypatch, version):
    def unexpected_load():
        pytest.fail("invalid argument ABI must not load the runtime")

    monkeypatch.setattr(a1_single_launch, "_load_runtime", unexpected_load)
    with pytest.raises(ValueError, match="ABI version must be 1 or 2"):
        a1_single_launch.A1SingleLaunch(b"\x7fELF", argument_abi_version=version)


def test_default_argument_abi_keeps_legacy_creation(monkeypatch):
    class Native:
        def simpler_a1_create(self, _image, _size, _device, handle):
            handle._obj.value = 321
            return 0

        def simpler_a1_close(self, _handle):
            return 0

    monkeypatch.setattr(a1_single_launch, "_load_runtime", Native)
    with a1_single_launch.A1SingleLaunch(b"\x7fELF") as runner:
        assert runner.argument_abi_version == 1
        assert runner.max_argument_abi_version == 2
        with pytest.raises(RuntimeError, match="requires A1 argument ABI 2"):
            _ = runner.l2_cache_info


@pytest.mark.parametrize("version", [1, 2])
def test_runtime_compiles_and_checks_abi_without_driver_load(monkeypatch, version):
    calls = []

    class Symbol:
        def __init__(self, name):
            self.name = name

        def __call__(self):
            assert self.name == "simpler_a1_argument_abi_version"
            calls.append(self.name)
            return version

    class Library:
        def __getattr__(self, name):
            symbol: Any = Symbol(name)
            setattr(self, name, symbol)
            return symbol

    library = Library()

    def capture_library(path):
        assert Path(path).is_file()
        return library

    monkeypatch.setattr(ctypes, "CDLL", capture_library)
    a1_single_launch._load_runtime.cache_clear()
    try:
        if version == 2:
            assert a1_single_launch._load_runtime() is library
        else:
            with pytest.raises(RuntimeError, match="ABI version mismatch"):
                a1_single_launch._load_runtime()
        assert calls == ["simpler_a1_argument_abi_version"]
    finally:
        a1_single_launch._load_runtime.cache_clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
