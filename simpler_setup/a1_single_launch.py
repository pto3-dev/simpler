# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronous execution of one coupled-core entry on Ascend310P."""

import ctypes
import platform
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from simpler import env_manager

from .environment import PROJECT_ROOT
from .toolchain import GxxToolchain


class _HostTensorArg(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("dtype", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * 5),
    ]


class _ResidentArg(ctypes.Structure):
    _fields_ = [("tensor", _HostTensorArg), ("offset", ctypes.c_uint64)]


@lru_cache(maxsize=1)
def _load_runtime() -> Any:
    env_manager.ensure("ASCEND_HOME_PATH")
    toolchain = GxxToolchain()
    if toolchain.ascend_home_path is None:
        raise RuntimeError("ASCEND_HOME_PATH is required for A1 single launch")
    sdk = Path(toolchain.ascend_home_path)
    common = PROJECT_ROOT / "src" / "common"
    source = PROJECT_ROOT / "src" / "a1" / "platform" / "onboard" / "host" / "single_launch.cpp"
    with TemporaryDirectory(prefix="simpler_a1_") as directory:
        output = str(Path(directory) / "single_launch.so")
        cmd = [
            toolchain.cxx_path,
            *toolchain.get_compile_flags(),
            f"-I{sdk / 'include'}",
            f"-I{sdk / 'pkg_inc'}",
            f"-I{sdk / 'pkg_inc' / 'runtime'}",
            f"-I{sdk / (platform.machine() + '-linux') / 'pkg_inc'}",
            f"-I{sdk / (platform.machine() + '-linux') / 'pkg_inc' / 'runtime'}",
            f"-I{common}",
            f"-I{common / 'task_interface'}",
            str(source),
            str(common / "task_interface" / "assert_compat.cpp"),
            f"-L{sdk / 'lib64'}",
            f"-Wl,-rpath,{sdk / 'lib64'}",
            "-lascendcl",
            "-lruntime",
            "-ldl",
            "-pthread",
            "-o",
            output,
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode:
            raise RuntimeError(f"A1 host runtime compilation failed:\n{result.stderr}")
        library = ctypes.CDLL(output)
    library.simpler_a1_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.simpler_a1_create.restype = ctypes.c_int
    library.simpler_a1_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(_HostTensorArg), ctypes.c_uint32]
    library.simpler_a1_run.restype = ctypes.c_int
    library.simpler_a1_close.argtypes = [ctypes.c_void_p]
    library.simpler_a1_close.restype = ctypes.c_int
    library.simpler_a1_last_error.argtypes = []
    library.simpler_a1_last_error.restype = ctypes.c_char_p
    library.simpler_a1_tensor_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_HostTensorArg),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.simpler_a1_tensor_create.restype = ctypes.c_int
    library.simpler_a1_tensor_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ResidentArg),
        ctypes.POINTER(_HostTensorArg),
        ctypes.c_uint32,
    ]
    library.simpler_a1_tensor_copy.restype = ctypes.c_int
    library.simpler_a1_register.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.simpler_a1_register.restype = ctypes.c_int
    library.simpler_a1_run_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ResidentArg),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    library.simpler_a1_run_resident.restype = ctypes.c_int
    library.simpler_a1_prepare_sequence.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ResidentArg),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.simpler_a1_prepare_sequence.restype = ctypes.c_int
    library.simpler_a1_run_sequence.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    library.simpler_a1_run_sequence.restype = ctypes.c_int
    library.simpler_a1_prepare_graph.argtypes = library.simpler_a1_prepare_sequence.argtypes
    library.simpler_a1_prepare_graph.restype = ctypes.c_int
    library.simpler_a1_run_graph.argtypes = library.simpler_a1_run_sequence.argtypes
    library.simpler_a1_run_graph.restype = ctypes.c_int
    library.simpler_a1_rebuild_graph.argtypes = library.simpler_a1_run_sequence.argtypes
    library.simpler_a1_rebuild_graph.restype = ctypes.c_int
    return library


def _tensor_arg(tensor: Any) -> _HostTensorArg:
    import torch  # noqa: PLC0415

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("A1 single launch accepts torch tensors")
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("A1 single launch requires contiguous CPU tensors")
    if tensor.dtype not in (torch.float16, torch.float32):
        raise ValueError("A1 single launch requires FP16 or FP32 tensors")
    shape = list(tensor.shape)
    if not 1 <= len(shape) <= 5 or any(not 0 < size <= 0xFFFFFFFF for size in shape):
        raise ValueError("tensor shape exceeds the A1 kernel ABI")
    if tensor.numel() > 0xFFFFFFFF:
        raise ValueError("tensor element count exceeds the A1 kernel ABI")
    return _HostTensorArg(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        1 if tensor.dtype == torch.float16 else 0,
        len(shape),
        (ctypes.c_uint32 * 5)(*(shape + [0] * (5 - len(shape)))),
    )


class A1DeviceTensor:
    """Dense session-owned FP16/FP32 storage; explicit transfers only.

    Views retain their owner and may change shape without changing dtype.
    Session close invalidates all tensors and views. Individual storage is
    released when the owning session closes, after its stream drains.
    """

    def __init__(
        self,
        owner: "A1SingleLaunch",
        tensor_id: int,
        shape: tuple[int, ...],
        dtype: Any,
        capacity: int,
        offset: int = 0,
    ):
        self._owner = owner
        self._id = tensor_id
        self.shape = shape
        self.dtype = dtype
        self._capacity = capacity
        self._offset = offset

    def _argument(self) -> _ResidentArg:
        return _ResidentArg(_geometry(self.shape, self.dtype, self._id), self._offset)

    def view(self, shape: tuple[int, ...], offset_elements: int = 0) -> "A1DeviceTensor":
        """Create a dense view relative to this view, without copying data."""
        geometry = _geometry(tuple(shape), self.dtype, self._id)
        current = self._argument().tensor
        if isinstance(offset_elements, bool) or not isinstance(offset_elements, int) or offset_elements < 0:
            raise ValueError("view offset must be a nonnegative integer")
        element_bytes = 2 if geometry.dtype == 1 else 4
        offset = offset_elements * element_bytes
        if offset > current.bytes or geometry.bytes > current.bytes - offset:
            raise ValueError("resident tensor view exceeds its parent view")
        return A1DeviceTensor(self._owner, self._id, tuple(shape), self.dtype, self._capacity, self._offset + offset)

    def copy_from(self, tensor: Any) -> None:
        """Upload one matching contiguous CPU tensor into this view."""
        self._owner._copy_tensor(self, tensor, upload=True)

    def to_cpu(self) -> Any:
        """Download this view into a newly allocated CPU tensor."""
        import torch  # noqa: PLC0415

        tensor = torch.empty(self.shape, dtype=self.dtype)
        self._owner._copy_tensor(self, tensor, upload=False)
        return tensor


def _geometry(shape: tuple[int, ...], dtype: Any, address: int = 1) -> _HostTensorArg:
    import torch  # noqa: PLC0415

    if dtype not in (torch.float16, torch.float32):
        raise ValueError("resident tensors require FP16 or FP32")
    if not 1 <= len(shape) <= 5 or any(
        isinstance(x, bool) or not isinstance(x, int) or not 0 < x <= 0xFFFFFFFF for x in shape
    ):
        raise ValueError("resident tensor shape exceeds the A1 kernel ABI")
    elements = 1
    for size in shape:
        elements *= size
    if elements > 0xFFFFFFFF:
        raise ValueError("resident tensor element count exceeds the A1 kernel ABI")
    return _HostTensorArg(
        address,
        elements * (2 if dtype == torch.float16 else 4),
        1 if dtype == torch.float16 else 0,
        len(shape),
        (ctypes.c_uint32 * 5)(*shape),
    )


class A1SingleLaunch:
    """Run a single ``__global__`` kernel with a tensor-pointer-array ABI.

    ``kernel`` is the whole ELF object returned by
    ``KernelCompiler('a1').compile_single_launch``. The entry accepts one GM
    pointer to an array of tensor descriptor addresses. Arguments must match
    the kernel's static signature. Calls are synchronous and copy every
    contiguous CPU tensor back. Exact aliases share device storage; partially
    overlapping views are rejected. Each session owns its context and stream,
    without resetting the device or finalizing the process-wide ACL runtime.
    """

    def __init__(self, kernel: bytes, device_id: int = 0):
        if not isinstance(kernel, bytes) or not kernel.startswith(b"\x7fELF"):
            raise ValueError("kernel must be a CANN-loadable ELF object")
        if isinstance(device_id, bool) or not isinstance(device_id, int) or not 0 <= device_id <= 0x7FFFFFFF:
            raise ValueError("device_id must be a nonnegative int32")
        self._lock = threading.RLock()
        self._handle = ctypes.c_void_p()
        self._library = _load_runtime()
        image = ctypes.create_string_buffer(kernel)
        self._check(self._library.simpler_a1_create(image, len(kernel), device_id, ctypes.byref(self._handle)))

    def _check(self, status: int) -> None:
        if status:
            error = self._library.simpler_a1_last_error()
            raise RuntimeError(error.decode() if error else "A1 runtime operation failed")

    def __call__(self, *tensors: Any) -> None:
        """Run the entry and synchronously update all argument tensors."""
        if not 1 <= len(tensors) <= 64:
            raise ValueError("tensor argument count must be in 1..64")
        args = (_HostTensorArg * len(tensors))(*(_tensor_arg(tensor) for tensor in tensors))
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("single launch session is closed")
            self._check(self._library.simpler_a1_run(self._handle, args, len(args)))

    def empty(self, shape: tuple[int, ...], dtype: Any) -> A1DeviceTensor:
        """Allocate an uninitialized dense device tensor in this session."""
        return self._create_tensor(_geometry(tuple(shape), dtype), dtype, upload=False)

    def upload(self, tensor: Any) -> A1DeviceTensor:
        """Allocate and upload a contiguous CPU tensor once."""
        return self._create_tensor(_tensor_arg(tensor), tensor.dtype, upload=True)

    def _create_tensor(self, argument: _HostTensorArg, dtype: Any, *, upload: bool) -> A1DeviceTensor:
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("single launch session is closed")
            tensor_id = ctypes.c_uint64()
            self._check(
                self._library.simpler_a1_tensor_create(
                    self._handle, ctypes.byref(argument), int(upload), ctypes.byref(tensor_id)
                )
            )
            return A1DeviceTensor(self, tensor_id.value, tuple(argument.shape[: argument.rank]), dtype, argument.bytes)

    def _copy_tensor(self, device: A1DeviceTensor, host: Any, *, upload: bool) -> None:
        argument = _tensor_arg(host)
        if device._owner is not self or tuple(host.shape) != device.shape or host.dtype != device.dtype:
            raise ValueError("resident transfer geometry or owner mismatch")
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("single launch session is closed")
            resident = device._argument()
            self._check(
                self._library.simpler_a1_tensor_copy(
                    self._handle, ctypes.byref(resident), ctypes.byref(argument), int(upload)
                )
            )

    def register_kernel(self, kernel: bytes) -> int:
        """Register another whole ELF object in the same device context."""
        if not isinstance(kernel, bytes) or not kernel.startswith(b"\x7fELF"):
            raise ValueError("kernel must be a CANN-loadable ELF object")
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("single launch session is closed")
            image = ctypes.create_string_buffer(kernel)
            kernel_id = ctypes.c_uint32()
            self._check(self._library.simpler_a1_register(self._handle, image, len(kernel), ctypes.byref(kernel_id)))
            return kernel_id.value

    def _device_arguments(self, tensors: tuple[A1DeviceTensor, ...], kernel_id: int, blocks: int) -> Any:
        if not 1 <= len(tensors) <= 64 or any(
            not isinstance(t, A1DeviceTensor) or t._owner is not self for t in tensors
        ):
            raise ValueError("resident launch requires 1..64 tensors owned by this session")
        if isinstance(kernel_id, bool) or not isinstance(kernel_id, int) or not 0 <= kernel_id <= 0xFFFFFFFF:
            raise ValueError("kernel id must be a nonnegative uint32")
        if isinstance(blocks, bool) or not isinstance(blocks, int) or not 1 <= blocks <= 8:
            raise ValueError("A1 launch blocks must be in 1..8")
        args = (_ResidentArg * len(tensors))(*(t._argument() for t in tensors))
        return args

    def run_device(self, *tensors: A1DeviceTensor, kernel_id: int = 0, blocks: int = 1) -> None:
        """Run synchronously on resident tensors without activation transfers."""
        args = self._device_arguments(tensors, kernel_id, blocks)
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("single launch session is closed")
            self._check(self._library.simpler_a1_run_resident(self._handle, args, len(args), kernel_id, blocks))

    def prepare_sequence(self, calls: list[tuple[int, int, tuple[A1DeviceTensor, ...]]]) -> "A1PreparedSequence":
        """Bind 1..4096 ordered calls to stable resident tensors and registered images.

        Preparation uploads descriptors once. Invocation queues the whole sequence
        in stream order and synchronizes once, without allocation or activation copies.
        """
        return self._prepare_calls(calls, device_model=False)

    def prepare_graph(self, calls: list[tuple[int, int, tuple[A1DeviceTensor, ...]]]) -> "A1PreparedSequence":
        """Prepare a static ordered device model with one model submission per replay.

        The model retains immutable tensor bindings. Stream order establishes
        dependencies, including in-place reads/writes. No kernel is executed
        during preparation, and no per-task host launch occurs during replay.
        """
        return self._prepare_calls(calls, device_model=True)

    def _prepare_calls(
        self, calls: list[tuple[int, int, tuple[A1DeviceTensor, ...]]], *, device_model: bool
    ) -> "A1PreparedSequence":
        if not 1 <= len(calls) <= 4096:
            raise ValueError("resident sequence requires 1..4096 calls")
        arguments = [self._device_arguments(tensors, kernel_id, blocks) for kernel_id, blocks, tensors in calls]
        counts = (ctypes.c_uint32 * len(calls))(*(len(args) for args in arguments))
        ids = (ctypes.c_uint32 * len(calls))(*(call[0] for call in calls))
        blocks = (ctypes.c_uint32 * len(calls))(*(call[1] for call in calls))
        flat = (_ResidentArg * sum(counts))(*(arg for args in arguments for arg in args))
        sequence_id = ctypes.c_uint32()
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("single launch session is closed")
            prepare = (
                self._library.simpler_a1_prepare_graph if device_model else self._library.simpler_a1_prepare_sequence
            )
            self._check(prepare(self._handle, flat, counts, ids, blocks, len(calls), ctypes.byref(sequence_id)))
        return A1PreparedSequence(
            self, sequence_id.value, tuple(t for call in calls for t in call[2]), device_model=device_model
        )

    def close(self) -> None:
        """Drain and release session resources; a failed drain remains retryable."""
        with self._lock:
            if self._handle.value:
                self._check(self._library.simpler_a1_close(self._handle))
                self._handle.value = None

    def __enter__(self) -> "A1SingleLaunch":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self):
        if hasattr(self, "_library"):
            try:
                self.close()
            except RuntimeError:
                pass


class A1PreparedSequence:
    """Immutable device bindings retained by one session; calls remain synchronous."""

    def __init__(
        self,
        owner: A1SingleLaunch,
        sequence_id: int,
        tensors: tuple[A1DeviceTensor, ...],
        *,
        device_model: bool = False,
    ):
        self._owner = owner
        self._sequence_id = sequence_id
        self._tensors = tensors
        self._device_model = device_model

    def rebuild(self) -> None:
        """Recreate this device model with the current CANN profiling state.

        Call after starting or stopping profiling, outside measured replays.
        Tensor bindings and this graph handle remain unchanged. No arithmetic
        kernel executes; the session drains before replacing model resources.
        A failed old-model cleanup remains owned for retry or session close.
        """
        if not self._device_model:
            raise ValueError("rebuild requires a prepared device model")
        with self._owner._lock:
            if not self._owner._handle.value:
                raise RuntimeError("single launch session is closed")
            self._owner._check(self._owner._library.simpler_a1_rebuild_graph(self._owner._handle, self._sequence_id))

    def __call__(self) -> None:
        with self._owner._lock:
            if not self._owner._handle.value:
                raise RuntimeError("single launch session is closed")
            library = self._owner._library
            run = library.simpler_a1_run_graph if self._device_model else library.simpler_a1_run_sequence
            self._owner._check(run(self._owner._handle, self._sequence_id))
