"""Capacity, not intent — refusing to load weights without RAM to spare.

The gate this replaces asked a boolean about *intent* ("is this an integration
test?") and said nothing about *capacity*. A 421M model was loaded onto a host
reporting **1.6 GB available**; load average reached 78 on 12 cores, the swap
storm followed, and the OS restarted. Intent is not a resource measurement.

Unknown memory is reported as unknown and **skipped, visibly** — a guard that
guesses is worse than no guard. Everything it can read, it reads *before*
anything is loaded, and believes the number.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Free-RAM multiplier required over the resident estimate, applied in exactly
#: one place — ``check_memory``. Weights plus activations plus the framework is
#: not the file size, and this host has taken a swap storm for less.
DEFAULT_HEADROOM = 1.5


@dataclass(frozen=True)
class MemoryCheck:
    """``ok`` is the verdict; ``reason`` always names the numbers or the absence."""

    ok: bool
    available_gb: Optional[float]
    required_gb: float
    reason: str


def available_gb() -> Optional[float]:
    """Free RAM in GB, or ``None`` when it cannot be determined.

    ``psutil`` first because it is portable and accounts for reclaimable cache;
    ``/proc/meminfo`` ``MemAvailable`` second because it is the kernel's own
    estimate of what a new process can actually get. ``None`` means unknown —
    callers must not read it as zero.
    """
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 2 ** 30
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 2 ** 20
    except Exception:
        pass
    return None


def model_gb(path, activation_factor: float = 1.15) -> float:
    """Resident estimate for a weights file: size on disk plus its activations.

    Deliberately *not* the headroom — that is ``check_memory``'s job. Applying
    both here inflated a 40 GB file to 90 GB in the first dry run (40 × 1.15
    then × 1.5 would be 69), which is the kind of number an operator rightly
    stops trusting.
    """
    size = Path(path).stat().st_size / 2 ** 30
    return size * activation_factor


def check_memory(required_gb: float, headroom: float = DEFAULT_HEADROOM) -> MemoryCheck:
    """Refuse to load ``required_gb`` of weights without room to spare."""
    if required_gb < 0:
        raise ValueError(f"required_gb cannot be negative, got {required_gb}")
    avail = available_gb()
    need = required_gb * headroom
    if avail is None:
        return MemoryCheck(True, None, required_gb,
                           "available memory unknown — check skipped (a guess "
                           "would be worse than none)")
    if avail >= need:
        return MemoryCheck(True, avail, required_gb,
                           f"{avail:.1f} GB free >= {need:.1f} GB needed "
                           f"({required_gb:.1f} GB weights x {headroom} headroom)")
    return MemoryCheck(False, avail, required_gb,
                       f"{avail:.1f} GB free < {need:.1f} GB needed "
                       f"({required_gb:.1f} GB weights x {headroom} headroom) — "
                       f"refusing to load; the host was taken down doing exactly this")


#: Resident estimates for the encoder configurations we load. Text-only is the
#: 270M default; vision/audio add their towers (EG2: 130M backbone + 140M
#: embedder + 170M vision + 300M audio).
#:
#: These live here rather than in ``encoder`` on purpose: importing that module
#: pulls in torch, and the whole point of the gate is to run *before* a
#: heavyweight import. A guard that needs the thing it is guarding is useless.
ENCODER_GB = {"text": 1.1, "text+vision": 1.8, "text+vision+audio": 2.6}
