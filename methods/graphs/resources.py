"""Memory/disk guards and progress used by exact joining and Leiden."""
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import os
import resource
import shutil
import stat
import threading
import time

GIB = 1024**3


def available_memory_gib():
    value = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    return int(value.split()[1]) * 1024 / GIB

class ResourceLimit(RuntimeError):
    """Signal a controlled stop before a configured resource limit is crossed."""


def directory_bytes(path: Path) -> int:
    """Return the current byte size of all regular files below a directory."""
    total = 0
    for item in path.rglob("*"):
        try:
            info = item.stat()
        except FileNotFoundError:
            # Another writer may atomically rename a temporary shard mid-scan.
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def current_rss_bytes() -> int:
    """Return current resident memory in bytes using Linux procfs."""
    try:
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        # ru_maxrss is KiB on Linux and is a safe upper bound when procfs is absent.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def peak_rss_bytes() -> int:
    """Return peak resident memory in bytes for the current process."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


@dataclass
class ResourceMonitor:
    """Check prospective allocations, resident memory, output size, and free disk."""

    output_dir: Path
    memory_limit_bytes: int
    output_limit_bytes: int
    free_disk_reserve_bytes: int

    def check_memory(self, extra_bytes: int = 0, label: str = "operation") -> None:
        """Raise before an allocation would exceed the configured RSS ceiling."""
        projected = current_rss_bytes() + max(0, int(extra_bytes))
        if projected > self.memory_limit_bytes:
            raise ResourceLimit(
                f"{label} projects {projected / GIB:.3f} GiB RSS, above "
                f"{self.memory_limit_bytes / GIB:.3f} GiB"
            )

    def check_output(self, extra_bytes: int = 0, label: str = "output") -> None:
        """Raise before output growth crosses its limit or free-disk reserve."""
        used = directory_bytes(self.output_dir) if self.output_dir.exists() else 0
        if used + max(0, int(extra_bytes)) > self.output_limit_bytes:
            raise ResourceLimit(
                f"{label} would exceed the {self.output_limit_bytes / GIB:.3f} GiB "
                "output limit"
            )
        parent = self.output_dir if self.output_dir.exists() else self.output_dir.parent
        free = shutil.disk_usage(parent).free
        if free - max(0, int(extra_bytes)) < self.free_disk_reserve_bytes:
            raise ResourceLimit(f"{label} would cross the configured free-disk reserve")


class Progress:
    """Print compact, throttled progress records for long-running phases."""

    def __init__(self, quiet: bool = False, interval_seconds: float = 10.0):
        """Create a reporter from quiet and interval settings; output is thread-safe."""
        self.quiet = quiet
        self.interval = max(0.1, float(interval_seconds))
        self.started = time.perf_counter()
        self.started_cpu = time.process_time()
        self.last = 0.0
        self.last_cpu = self.started_cpu
        self.lock = threading.Lock()

    def show(self, phase: str, force: bool = False, **values) -> None:
        """Print one phase record when due; inputs are a label and scalar counters."""
        if self.quiet:
            return
        now = time.perf_counter()
        with self.lock:
            if not force and now - self.last < self.interval:
                return
            previous_wall = self.last or self.started
            cpu_now = time.process_time()
            cpu_percent = 100.0 * (cpu_now - self.last_cpu) / max(1e-9, now - previous_wall)
            self.last = now
            self.last_cpu = cpu_now
            fields = " ".join(f"{key}={value}" for key, value in values.items())
            if values.get("completed") and values.get("total"):
                elapsed = now - self.started
                eta = elapsed * (float(values["total"]) / float(values["completed"]) - 1.0)
                fields += f" eta_s={max(0.0, eta):.0f}"
            print(
                f"[{now - self.started:9.1f}s] {phase} {fields} "
                f"rss={current_rss_bytes() / GIB:.2f}GiB cpu~={cpu_percent:.0f}% cores~={cpu_percent / 100:.1f}",
                flush=True,
            )


class WorkingMemoryBudget:
    """Bound the sum of estimated worker-local allocations without wasting memory."""

    def __init__(self, limit_bytes: int):
        """Create a shared byte budget; output is a reservation context manager."""
        self.limit = max(1, int(limit_bytes))
        self.used = 0
        self.condition = threading.Condition()

    @contextmanager
    def reserve(self, amount: int):
        """Wait for and hold estimated bytes, yielding nothing, then release them."""
        amount = max(1, int(amount))
        if amount > self.limit:
            raise ResourceLimit(
                f"one worker needs {amount / GIB:.3f} GiB, above the "
                f"{self.limit / GIB:.3f} GiB working-memory limit"
            )
        with self.condition:
            while self.used + amount > self.limit:
                self.condition.wait(timeout=1.0)
            self.used += amount
        try:
            yield
        finally:
            with self.condition:
                self.used -= amount
                self.condition.notify_all()

