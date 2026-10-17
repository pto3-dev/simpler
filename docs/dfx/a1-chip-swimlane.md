# A1 native chip observation protocol

Ascend310P (`dav-m200`) does not expose the AICore `SYS_CNT` instruction
through the supported CANN compiler. The A1 diagnostic collector uses the
AICPU architectural virtual counter (`cntvct_el0`) and reads its frequency
from `cntfrq_el0`. The verified 310P3 frequency is 38.4 MHz; it is not an
assumed 50 MHz A2/A3 frequency. Ordered reads use ISB around the counter.

## Producer contract

`src/a1/platform/include/chip_profile_wire.h` defines the versioned POD
layout. Allocate exactly `ChipProfileBytes(tasks, cores)`, zero it, then
set version, positive task count, one through eight cores, and a positive
timeout of at most one second in `timeout_us`. The host must validate these
values and the allocation before launching either entry.

`src/a1/platform/onboard/aicpu/chip_profile.cpp` builds a standalone AICPU
SO exporting the common loader's init/exec symbols. Its exec payload is one
uint64 GM address of this buffer. It runs concurrently with a persistent
AICore entry using `chip_profile_core.h`; it does not run model arithmetic.
The architecture-independent `LoadAicpuOp` bootstraps and launches this SO.

For block b and task t, the core protocol is:

1. Publish physical `get_coreid() + 1` in `hello[b]` once.
2. Wait for `commands[t] == 1`; publish `states[b] = 2*t + 1`.
3. Wait for `ready[t] == 1`; execute the task body.
4. Drain all pipelines; publish `states[b] = 2*t + 2`.

The AICPU observes receipt on every block before releasing a task. It
stages the next command immediately after release while the current task
can compute, then observes completion on every block before releasing
the next dependency. This deliberately imposes a sequential task barrier.
It records the real dispatch begin/end, dependency release, and per-block
receipt/completion observations. A completed task can be observed after
the core has already posted receipt for its successor; monotonic states
preserve that completion observation without inventing a core timestamp.

Each writer owns a separate 128-byte GM flag line. A1 DCCI takes two
arguments; AICPU uses cache clean/invalidate and DSB. Never place a
CPU-written token and a core-written status on the same cache line.
Status 1 means complete, 2 timeout/abort, and 3 invalid header. Both peers
have bounded polling; timeout publishes abort tokens. Drain both streams
before downloading or freeing any buffer.

## Interpretation and validation

These are **AICPU-observed envelopes**, including polling, coherence,
barrier and observer overhead. They are not AICore instruction start/end
timestamps or pure compute durations. Early staging is a measured queue
publication, not proof that a core prefetches a descriptor while busy.
Do not export these as the existing A2/A3 instruction-timestamp schema.
Use separate low-overhead HWTS/PMU captures for model speed comparisons.

On 310P3/CANN 9.1, five changing-input Add-to-Mul diagnostic runs retain
exact outputs and guards. A full 216-task, eight-core YOLO26n 128x128
diagnostic replay passes three rounds with all 31 saved arrays byte-equal
to the strictly validated model replay. The diagnostic host wall time is
8.02–8.41 ms and includes collection overhead; it is not a deployment
latency claim. The default static CANN model backend remains unchanged.
