"""Packed supports → duplicate/minor states → exact candidates → signed edge shards."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import math
import json
import os
import threading
import time
import numpy as np
from numba import set_num_threads
from itertools import combinations_with_replacement
from methods.graphs.kernels import (_fill_ordered_tokens, _build_postings,
                                   _positive_matches_batch, _negative_matches_batch)
from methods.graphs.resources import (GIB, ResourceLimit, ResourceMonitor,
    Progress, WorkingMemoryBudget, directory_bytes, peak_rss_bytes)
from execution.cache import write_json

EDGE_FORMAT_VERSION = 2
DEFAULT_BUFFER_BYTES = 16 * 1024**2
GROUP_BATCH_SIZE = 262_144
DEFAULT_MIN_ONE_SUPPORT = 5
EPSILON = 1e-12

NODE_DTYPE = np.dtype(
    [
        ("feature_id", "<u4"),
        ("support", "<u2"),
        ("flipped", "?"),
        ("constant", "?"),
        ("eligible", "?"),
        ("exclusion_code", "u1"),
    ]
)
EDGE_DTYPE = np.dtype(
    [
        ("feature_j", "<u4"),
        ("feature_k", "<u4"),
        ("phi", "<f4"),
    ]
)
BYTE_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

def load_packed_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and validate packed bits, feature IDs, and metadata from a dataset."""
    with (dataset_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {"format", "version", "n", "p", "word_count", "bit_order"}
    if not required.issubset(metadata):
        raise ValueError("packed metadata is missing required fields")
    if metadata["format"] != "bernoulli-packed-uint64" or metadata["version"] != 1:
        raise ValueError("unsupported packed dataset format")
    n, p = int(metadata["n"]), int(metadata["p"])
    words = int(metadata["word_count"])
    if words != (n + 63) // 64 or not 0 < n <= np.iinfo(np.uint16).max:
        raise ValueError("invalid packed dimensions")
    bits = np.load(dataset_dir / "bits.npy", mmap_mode="r", allow_pickle=False)
    if bits.dtype != np.dtype("<u8") or bits.shape != (p, words):
        raise ValueError("bits.npy has the wrong dtype or shape")
    if n % 64:
        valid_mask = np.uint64((1 << (n % 64)) - 1)
        if np.any(np.bitwise_and(bits[:, -1], np.bitwise_not(valid_mask))):
            raise ValueError("nonzero padding bits found after the final sample")
    ids_path = dataset_dir / "feature_ids.npy"
    feature_ids = (
        np.load(ids_path, mmap_mode="r", allow_pickle=False)
        if ids_path.exists()
        else np.arange(p, dtype=np.uint32)
    )
    if feature_ids.dtype != np.dtype("<u4") or feature_ids.shape != (p,):
        raise ValueError("feature_ids.npy has the wrong dtype or shape")
    if np.unique(feature_ids).size != p:
        raise ValueError("feature IDs must be unique")
    return bits, feature_ids, metadata


def packed_supports(bits: np.ndarray, chunk_features: int | None = None) -> np.ndarray:
    """Count set bits per feature in bounded blocks and return uint16 supports."""
    if chunk_features is None:
        chunk_features = max(1, min(50_000, (32 * 1024**2) // bits.shape[1] // 8))
    supports = np.empty(bits.shape[0], dtype=np.uint16)
    for start in range(0, bits.shape[0], chunk_features):
        stop = min(bits.shape[0], start + chunk_features)
        byte_view = np.asarray(bits[start:stop]).view(np.uint8).reshape(stop - start, -1)
        supports[start:stop] = BYTE_POPCOUNT[byte_view].sum(axis=1, dtype=np.uint32)
    return supports


def positive_overlap_threshold(n: int, r: int, s: int, threshold: float) -> int:
    """Return a conservative integer overlap threshold for positive phi."""
    value = (r * s + threshold * math.sqrt(r * s * (n - r) * (n - s))) / n
    return int(math.ceil(value - EPSILON * max(1.0, abs(value))))


def negative_overlap_threshold(n: int, r: int, s: int, threshold: float) -> int:
    """Return a conservative maximum overlap for negative phi."""
    value = (r * s - threshold * math.sqrt(r * s * (n - r) * (n - s))) / n
    return int(math.floor(value + EPSILON * max(1.0, abs(value))))


def maximum_positive_phi(n: int, r: int, s: int) -> float:
    """Return the largest feasible positive phi for two minor supports."""
    small, large = min(r, s), max(r, s)
    return math.sqrt(small * (n - large) / (large * (n - small)))


def maximum_negative_abs_phi(n: int, r: int, s: int) -> float:
    """Return the largest feasible negative absolute phi for minor supports."""
    return math.sqrt(r * s / ((n - r) * (n - s)))


@dataclass
class PreparedJoin:
    """Minor-state groups used by both join branches.

    unique_bits is groups × ceil(samples/64). group_offsets indexes original
    pair rows in group_members; token_offsets indexes each group's sample tokens.
    """

    n: int
    nodes: np.ndarray
    unique_bits: np.ndarray
    group_support: np.ndarray
    group_offsets: np.ndarray
    group_members: np.ndarray
    token_offsets: np.ndarray
    tokens: np.ndarray
    bins: dict[int, np.ndarray]


def build_node_table(
    bits: np.ndarray,
    feature_ids: np.ndarray,
    n: int,
    supports: np.ndarray | None = None,
    min_one_support: int = DEFAULT_MIN_ONE_SUPPORT,
) -> np.ndarray:
    """Build compact node metadata from packed bits, IDs, and optional supports."""
    if supports is None:
        supports = packed_supports(bits)
    nodes = np.empty(bits.shape[0], dtype=NODE_DTYPE)
    nodes["feature_id"] = feature_ids
    nodes["support"] = supports
    nodes["flipped"] = supports > n // 2
    nodes["constant"] = (supports == 0) | (supports == n)
    nodes["eligible"] = (supports >= min_one_support) & (supports < n)
    nodes["exclusion_code"] = np.where(
        supports == n, 2, np.where(supports < min_one_support, 1, 0)
    ).astype(np.uint8)
    return nodes


def prepare_join_data(
    bits: np.ndarray,
    feature_ids: np.ndarray,
    n: int,
    monitor: ResourceMonitor,
    min_one_support: int = DEFAULT_MIN_ONE_SUPPORT,
) -> PreparedJoin:
    """Create set states, unique groups, exact support bins, and ordered tokens."""
    p, words = bits.shape
    supports = packed_supports(bits)
    eligible = (supports >= min_one_support) & (supports < n)
    flipped = supports > n // 2
    minor_supports = np.minimum(supports, n - supports).astype(np.uint16)
    monitor.check_memory(bits.nbytes + p * 16, "minor-state representation")
    set_bits = np.array(bits, dtype=np.uint64, copy=True)
    final_mask = (
        np.uint64((1 << (n % 64)) - 1)
        if n % 64
        else np.uint64(2**64 - 1)
    )

    # Complement dense columns and mask unused bits in the final word.
    for start in range(0, p, 50_000):
        stop = min(p, start + 50_000)
        selected = np.flatnonzero(flipped[start:stop]) + start
        if selected.size:
            set_bits[selected] = np.bitwise_not(set_bits[selected])
            set_bits[selected, -1] &= final_mask
    set_supports = minor_supports

    nodes = build_node_table(bits, feature_ids, n, supports, min_one_support)
    active = np.flatnonzero(eligible).astype(np.uint32)
    if active.size == 0:
        empty_u64 = np.empty((0, words), dtype=np.uint64)
        return PreparedJoin(
            n,
            nodes,
            empty_u64,
            np.empty(0, np.uint16),
            np.zeros(1, np.int64),
            np.empty(0, np.uint32),
            np.zeros(1, np.int64),
            np.empty(0, np.uint16),
            {},
        )

    monitor.check_memory(active.size * (words * 32 + 32), "unique-support grouping")
    unique_bits, inverse = np.unique(set_bits[active], axis=0, return_inverse=True)
    group_counts = np.bincount(inverse, minlength=unique_bits.shape[0])
    group_offsets = np.empty(unique_bits.shape[0] + 1, dtype=np.int64)
    group_offsets[0] = 0
    np.cumsum(group_counts, out=group_offsets[1:])
    member_order = np.argsort(inverse, kind="stable")
    group_members = active[member_order]
    first_members = group_members[group_offsets[:-1]]
    group_support = set_supports[first_members]

    # Count row frequencies with original multiplicity; any total token order is exact.
    row_frequency = np.zeros(n, dtype=np.int64)
    frequency_chunk = max(1, min(50_000, (64 * 1024**2) // max(1, words * 64)))
    for start in range(0, p, frequency_chunk):
        stop = min(p, start + frequency_chunk)
        monitor.check_memory(
            (stop - start) * words * (8 + 64), "row-frequency block"
        )
        active_chunk = set_bits[start:stop][eligible[start:stop]]
        if active_chunk.size:
            unpacked = np.unpackbits(active_chunk.view(np.uint8), axis=1, bitorder="little")
            row_frequency += unpacked[:, :n].sum(axis=0, dtype=np.int64)
    token_order = np.lexsort((np.arange(n), row_frequency)).astype(np.uint16)
    token_offsets = np.empty(unique_bits.shape[0] + 1, dtype=np.int64)
    token_offsets[0] = 0
    np.cumsum(group_support.astype(np.int64), out=token_offsets[1:])
    monitor.check_memory(int(token_offsets[-1]) * 2, "ordered support tokens")
    tokens = np.empty(token_offsets[-1], dtype=np.uint16)
    _fill_ordered_tokens(unique_bits, token_order, token_offsets, tokens)

    group_order = np.argsort(group_support, kind="stable")
    sorted_support = group_support[group_order]
    bins = {}
    for support in np.unique(sorted_support):
        left = np.searchsorted(sorted_support, support, side="left")
        right = np.searchsorted(sorted_support, support, side="right")
        bins[int(support)] = group_order[left:right].astype(np.uint32, copy=False)
    return PreparedJoin(
        n,
        nodes,
        unique_bits,
        group_support,
        group_offsets,
        group_members,
        token_offsets,
        tokens,
        bins,
    )


class EdgeWriter:
    """Expand matches into sorted compact shards using worker-local buffers."""

    def __init__(
        self,
        output_dir: Path,
        monitor: ResourceMonitor,
        nodes: np.ndarray,
    ):
        """Create a bounded edge buffer; inputs configure its schema and phi sign."""
        self.edge_dir = output_dir / "edges"
        self.edge_dir.mkdir()
        self.monitor = monitor
        self.nodes = nodes
        self.lock = threading.RLock()
        capacity = max(1, DEFAULT_BUFFER_BYTES // EDGE_DTYPE.itemsize)
        monitor.check_memory(capacity * EDGE_DTYPE.itemsize, "edge buffer")
        self.capacity = capacity
        self.buffers: dict[int, dict] = {}
        self.part = 0
        self.edge_count = 0
        self.edge_bytes = 0
        self.accepted_edges = 0
        self.base_output_bytes = directory_bytes(output_dir)
        self.expansion_chunk = min(capacity, 1_000_000)

    @property
    def used(self) -> int:
        """Return edges currently held across all worker-local buffers."""
        with self.lock:
            return sum(int(state["used"]) for state in self.buffers.values())

    def _state(self) -> dict:
        """Return the calling thread's lazily allocated edge buffer."""
        thread_id = threading.get_ident()
        with self.lock:
            state = self.buffers.get(thread_id)
            if state is None:
                self.monitor.check_memory(
                    self.capacity * EDGE_DTYPE.itemsize, "worker edge buffer"
                )
                state = {
                    "buffer": np.empty(self.capacity, dtype=EDGE_DTYPE),
                    "used": 0,
                }
                self.buffers[thread_id] = state
            return state

    def reserve(self, edge_count: int) -> None:
        """Atomically reserve projected output bytes for an edge batch."""
        with self.lock:
            projected_edges = self.accepted_edges + edge_count
            projected_parts = (projected_edges + self.capacity - 1) // self.capacity
            projected_bytes = (
                self.base_output_bytes
                + projected_edges * EDGE_DTYPE.itemsize
                + projected_parts * 256
                + 1024**2
            )
            if projected_bytes > self.monitor.output_limit_bytes:
                raise ResourceLimit(
                    "edge expansion would exceed the "
                    f"{self.monitor.output_limit_bytes / GIB:.3f} GiB output limit"
                )
            self.accepted_edges = projected_edges

    def _flush_state(self, state: dict) -> None:
        """Sort and atomically write one worker buffer."""
        used = int(state["used"])
        if not used:
            return
        values = state["buffer"][:used]
        order = np.lexsort((values["feature_k"], values["feature_j"]))
        sorted_values = values[order]
        # The disk check and write must be one transaction across search workers.
        with self.lock:
            self.monitor.check_output(used * EDGE_DTYPE.itemsize + 256, "edge shard")
            part = self.part
            final = self.edge_dir / f"part-{part:06d}.npy"
            temporary = final.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, sorted_values, allow_pickle=False)
            size = temporary.stat().st_size
            os.replace(temporary, final)
            self.part += 1
            self.edge_count += used
            self.edge_bytes += size
        state["used"] = 0


    def _append(
        self,
        left: np.ndarray,
        right: np.ndarray,
        phis: np.ndarray | float,
    ) -> None:
        """Restore original signs and buffer pair-row indices with scalar/vector phi."""
        state = self._state()
        cursor = 0
        while cursor < left.size:
            if state["used"] == self.capacity:
                self._flush_state(state)
            take = min(left.size - cursor, self.capacity - state["used"])
            source = slice(cursor, cursor + take)
            target = slice(state["used"], state["used"] + take)
            left_part, right_part = left[source], right[source]
            left_ids = self.nodes["feature_id"][left_part]
            right_ids = self.nodes["feature_id"][right_part]
            buffer = state["buffer"]
            buffer["feature_j"][target] = np.minimum(left_ids, right_ids)
            buffer["feature_k"][target] = np.maximum(left_ids, right_ids)
            # Dense-column complements reverse the sign when exactly one endpoint was flipped.
            signs = np.where(self.nodes["flipped"][left_part] == self.nodes["flipped"][right_part], 1., -1.)
            buffer["phi"][target] = (phis if np.ndim(phis) == 0 else phis[source]) * signs
            state["used"] += take
            cursor += take


    def add_group_clique(
        self,
        group: int,
        group_offsets: np.ndarray,
        group_members: np.ndarray,
    ) -> None:
        """Emit all pairs inside one duplicate-support group with absolute phi one."""
        members = group_members[group_offsets[group] : group_offsets[group + 1]]
        count = int(members.size) * (int(members.size) - 1) // 2
        self.reserve(count)
        for position in range(members.size - 1):
            remaining = members[position + 1 :]
            for start in range(0, remaining.size, self.expansion_chunk):
                right = remaining[start : start + self.expansion_chunk]
                left = np.full(right.size, members[position], dtype=np.uint32)
                self._append(left, right, 1.)


    def add_group_pairs(
        self,
        left_groups: np.ndarray,
        right_groups: np.ndarray,
        phis: np.ndarray,
        group_offsets: np.ndarray,
        group_members: np.ndarray,
    ) -> None:
        """Expand and append a batch of verified support-group relations."""
        if not (left_groups.size == right_groups.size == phis.size):
            raise ValueError("group relation arrays must have equal lengths")
        if not left_groups.size:
            return
        left_sizes = (group_offsets[left_groups + 1] - group_offsets[left_groups]).astype(
            np.uint64
        )
        right_sizes = (
            group_offsets[right_groups + 1] - group_offsets[right_groups]
        ).astype(np.uint64)
        counts = left_sizes * right_sizes
        cumulative = np.cumsum(counts, dtype=np.uint64)
        starts = np.r_[np.uint64(0), cumulative[:-1]]
        expanded = int(cumulative[-1])
        self.reserve(expanded)

        # Flatten each Cartesian group relation in bounded vectorized chunks.
        chunk_size = min(self.expansion_chunk, GROUP_BATCH_SIZE)
        for flat_start in range(0, expanded, chunk_size):
            flat_stop = min(expanded, flat_start + chunk_size)
            flat = np.arange(flat_start, flat_stop, dtype=np.uint64)
            relation = np.searchsorted(cumulative, flat, side="right")
            local = flat - starts[relation]
            relation_right_sizes = right_sizes[relation]
            left_positions = (
                group_offsets[left_groups[relation]]
                + (local // relation_right_sizes).astype(np.int64)
            )
            right_positions = (
                group_offsets[right_groups[relation]]
                + (local % relation_right_sizes).astype(np.int64)
            )
            left = group_members[left_positions]
            right = group_members[right_positions]
            self._append(left, right, phis[relation])

    def close(self) -> None:
        """Flush the final shard; the output is a complete sequence of edge parts."""
        for state in list(self.buffers.values()):
            self._flush_state(state)


def _collect_threaded_jobs(jobs: list[tuple], worker, workers: int,
                           phase: str, progress: Progress) -> list[dict]:
    """Run bounded support-bin jobs, cancelling unstarted work on failure."""
    results = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=phase) as pool:
        pending = {pool.submit(worker, *job) for job in jobs}
        try:
            while pending:
                done, pending = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
                results.extend(future.result() for future in done)
                progress.show(phase, completed=len(results), total=len(jobs),
                              active_workers=sum(future.running() for future in pending))
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return results


def _bin_job(prepared: PreparedJoin, threshold: float, writer: EdgeWriter,
             monitor: ResourceMonitor, budget: WorkingMemoryBudget, progress: Progress,
             r: int, s: int, *, positive: bool) -> dict:
    """Search one minor-support bin pair; share only buffering/guards between kernels."""
    n = prepared.n
    r_ids, s_ids = prepared.bins[r], prepared.bins[s]
    if positive:
        bound = positive_overlap_threshold(n, r, s, threshold)
        r_prefix, s_prefix = r - bound + 1, s - bound + 1
        index_first = r_ids.size * r_prefix <= s_ids.size * s_prefix
    else:
        bound = negative_overlap_threshold(n, r, s, threshold)
        r_prefix = s_prefix = 0
        index_first = r_ids.size <= s_ids.size
    index_ids, query_ids = (r_ids, s_ids) if index_first else (s_ids, r_ids)
    index_support, query_support = (r, s) if index_first else (s, r)
    index_prefix, query_prefix = (r_prefix, s_prefix) if index_first else (s_prefix, r_prefix)
    capacity = max(GROUP_BATCH_SIZE, int(index_ids.size))
    estimated = (int(index_ids.size) * (index_prefix * 6 + 44) + (n + 1) * 8 + capacity * 10
                 if positive else int(index_ids.size) * 8 + capacity * 10)
    phase = "positive" if positive else "negative"
    candidates = hits = group_edges = query_number = 0
    with budget.reserve(estimated):
        monitor.check_memory(estimated, f"{phase} bin pair ({r},{s})")
        if positive:
            posting_offsets, posting_ids, posting_positions = _build_postings(
                index_ids, prepared.token_offsets, prepared.tokens, index_prefix, n)
            marks = np.zeros(index_ids.size, dtype=np.int64)
            touched = np.empty(index_ids.size, dtype=np.uint32)
        match_local = np.empty(index_ids.size, dtype=np.uint32)
        match_overlap = np.empty(index_ids.size, dtype=np.uint16)
        batch_left = np.empty(capacity, dtype=np.uint32)
        batch_right = np.empty(capacity, dtype=np.uint32)
        batch_overlap = np.empty(capacity, dtype=np.uint16)
        while query_number < query_ids.size:
            if positive:
                processed, matches, checked, prefix_hits = _positive_matches_batch(
                    query_ids, query_number, query_support, query_prefix, index_support, bound, r == s,
                    index_ids, prepared.token_offsets, prepared.tokens, posting_offsets, posting_ids,
                    posting_positions, prepared.unique_bits, marks, touched, match_local, match_overlap,
                    batch_left, batch_right, batch_overlap)
                hits += prefix_hits
            else:
                processed, matches, checked = _negative_matches_batch(
                    query_ids, query_number, query_support, index_support, bound, r == s, index_ids,
                    prepared.unique_bits, match_local, match_overlap, batch_left, batch_right, batch_overlap)
            if processed == 0:
                raise RuntimeError(f"{phase} relation batch could not fit one query")
            candidates += checked
            # Verify the same signed-phi cutoff after the conservative overlap bound.
            overlaps = batch_overlap[:matches].astype(np.float64)
            denominator = math.sqrt(float(query_support) * index_support * (n - query_support) * (n - index_support))
            phis = (n * overlaps - query_support * index_support) / denominator
            keep = phis + EPSILON >= threshold if positive else phis - EPSILON <= -threshold
            writer.add_group_pairs(batch_left[:matches][keep], batch_right[:matches][keep], phis[keep],
                                   prepared.group_offsets, prepared.group_members)
            group_edges += int(keep.sum())
            query_number += processed
            if query_number % 1024 < processed:
                monitor.check_memory(label=f"{phase} join")
                progress.show(phase, support=f"{r},{s}", completed_queries=query_number,
                              total_queries=query_ids.size, candidates=candidates,
                              edges=writer.edge_count + writer.used)
    return (dict(candidates=candidates, prefix_hits=hits, group_edges=group_edges) if positive
            else dict(pairs_checked=candidates, group_edges=group_edges))


def _run_branch(prepared: PreparedJoin, threshold: float, writer: EdgeWriter,
                monitor: ResourceMonitor, workers: int, working_memory_bytes: int,
                progress: Progress | None, *, positive: bool) -> dict:
    """Schedule feasible support pairs; prefix and negative searches remain distinct."""
    progress = progress or Progress(quiet=True)
    counts = dict(bin_pairs=0, group_edges=0)
    counts.update(dict(prefix_hits=0, candidates=0) if positive else dict(pairs_checked=0))
    maximum = maximum_positive_phi if positive else maximum_negative_abs_phi
    overlap_bound = positive_overlap_threshold if positive else negative_overlap_threshold
    jobs = []
    for r, s in combinations_with_replacement(sorted(prepared.bins), 2):
        if maximum(prepared.n, r, s) >= threshold - EPSILON:
            bound = overlap_bound(prepared.n, r, s, threshold)
            feasible = bound <= min(r, s) if positive else bound >= 0
            if feasible:
                jobs.append((r, s))
    budget = WorkingMemoryBudget(working_memory_bytes)
    worker = lambda r, s: _bin_job(prepared, threshold, writer, monitor, budget, progress, r, s, positive=positive)
    # Compile each kernel on the first job before parallel threads enter it.
    results = [worker(*jobs[0])] if jobs else []
    results += _collect_threaded_jobs(jobs[1:], worker, workers, "positive" if positive else "negative", progress)
    for result in results:
        counts["bin_pairs"] += 1
        for name, value in result.items():
            counts[name] += value
    return counts


def run_positive_join(prepared: PreparedJoin, threshold: float, writer: EdgeWriter,
                      monitor: ResourceMonitor, workers: int = 1, working_memory_bytes: int = 40 * GIB,
                      progress: Progress | None = None) -> dict:
    """Emit duplicate cliques, then search positive minor-state correlations by PPJoin."""
    for group, size in enumerate(np.diff(prepared.group_offsets)):
        if size > 1:
            writer.add_group_clique(group, prepared.group_offsets, prepared.group_members)
    return _run_branch(prepared, threshold, writer, monitor, workers, working_memory_bytes, progress, positive=True)


def run_negative_join(prepared: PreparedJoin, threshold: float, writer: EdgeWriter,
                      monitor: ResourceMonitor, workers: int = 1, working_memory_bytes: int = 40 * GIB,
                      progress: Progress | None = None) -> dict:
    """Search only feasible strong negative minor-state correlations, with exact overlaps."""
    return _run_branch(prepared, threshold, writer, monitor, workers, working_memory_bytes, progress, positive=False)


def run_join(
    dataset_dir: Path, output_dir: Path, threshold: float,
    memory_limit_gib: float = 100., output_limit_gib: float = 100.,
    free_disk_reserve_gib: float = 10., workers: int = 4,
    working_memory_gib: float = 40., quiet: bool = False,
    progress_seconds: float = 10., min_one_support: int = DEFAULT_MIN_ONE_SUPPORT,
) -> dict:
    """Build signed exact edges for |phi|>=threshold; return completion/resource status.

    Preparation groups duplicate minor states. The positive PPJoin and feasible
    negative fallback share bounded buffers, never an all-pairs correlation matrix.
    """
    if not 0 < threshold <= 1 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and in (0, 1]")
    if memory_limit_gib <= 0 or output_limit_gib <= 0 or free_disk_reserve_gib < 0:
        raise ValueError("resource limits must be positive and disk reserve nonnegative")
    if workers < 1 or working_memory_gib <= 0 or not 1 <= min_one_support < 65536:
        raise ValueError("workers and working memory must be positive")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    working_memory_gib = min(working_memory_gib, memory_limit_gib)
    monitor = ResourceMonitor(output_dir, int(memory_limit_gib * GIB), int(output_limit_gib * GIB),
                              int(free_disk_reserve_gib * GIB))
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    progress = Progress(quiet, progress_seconds)
    set_num_threads(1)
    manifest = dict(status="running", method="exact", edge_weight_field="phi", threshold=float(threshold),
        memory_limit_gib=float(memory_limit_gib), output_limit_gib=float(output_limit_gib),
        free_disk_reserve_gib=float(free_disk_reserve_gib), dataset=str(dataset_dir), workers=int(workers),
        working_memory_gib=float(working_memory_gib), min_one_support=int(min_one_support),
        edge_format_version=EDGE_FORMAT_VERSION, edge_weight=dict(field="phi", transform="abs"))
    write_json(output_dir / "run.json", manifest)
    writer, phase_times = None, {}
    try:
        phase_started = time.perf_counter()
        progress.show("load", force=True, dataset=dataset_dir)
        bits, feature_ids, metadata = load_packed_dataset(dataset_dir)
        progress.show("preprocess", force=True, features=metadata["p"])
        prepared = prepare_join_data(bits, feature_ids, int(metadata["n"]), monitor, min_one_support=min_one_support)
        monitor.check_output(prepared.nodes.nbytes + 256, "node table")
        np.save(output_dir / "nodes.npy", prepared.nodes, allow_pickle=False)
        phase_times["preprocess"] = time.perf_counter() - phase_started
        writer = EdgeWriter(output_dir, monitor, prepared.nodes)
        progress.show("positive", force=True, workers=workers)
        phase_started = time.perf_counter()
        positive = run_positive_join(prepared, threshold, writer, monitor, workers, int(working_memory_gib * GIB), progress)
        phase_times["positive_search"] = time.perf_counter() - phase_started
        rho = float(prepared.group_support.max()) / prepared.n if prepared.group_support.size else 0.
        # This margin bound proves when the negative search can contribute no edges.
        needs_negative = rho > 0 and threshold <= rho / (1. - rho) + EPSILON
        phase_started = time.perf_counter()
        negative = (run_negative_join(prepared, threshold, writer, monitor, workers, int(working_memory_gib * GIB), progress)
                    if needs_negative else dict(bin_pairs=0, pairs_checked=0, group_edges=0))
        phase_times["negative_fallback"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        writer.close()
        phase_times["finalization"] = time.perf_counter() - phase_started
        manifest.update(status="complete", n=prepared.n, p=int(prepared.nodes.size),
            active_features=int(prepared.nodes["eligible"].sum()), constant_features=int(prepared.nodes["constant"].sum()),
            low_prevalence_features=int((prepared.nodes["exclusion_code"] == 1).sum()),
            unique_support_sets=int(prepared.unique_bits.shape[0]), rho_hat=rho,
            negative_branch_required=bool(needs_negative), positive=positive, negative=negative)
    except ResourceLimit as error:
        if writer is not None:
            try:
                writer.close()
            except ResourceLimit:
                # Free disk may change before even the remaining partial shard fits.
                pass
        manifest.update(status="aborted_resource_limit", reason=str(error))
    except Exception as error:
        manifest.update(status="failed", reason=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.update(edge_count=writer.edge_count if writer else 0,
            edge_shards=writer.part if writer else 0, edge_bytes=writer.edge_bytes if writer else 0,
            phase_wall_s=phase_times, wall_s=time.perf_counter() - started_wall,
            cpu_s=time.process_time() - started_cpu, peak_rss_bytes=peak_rss_bytes(),
            output_bytes=directory_bytes(output_dir))
        write_json(output_dir / "run.json", manifest)
        progress.show("finished", force=True, status=manifest["status"], edges=manifest["edge_count"],
                      wall_s=f"{manifest['wall_s']:.1f}")
    return manifest


def iter_edge_shards(output_dir: Path) -> Iterator[np.ndarray]:
    """Yield memory-mapped edge arrays from a completed or partial run."""
    for path in sorted((output_dir / "edges").glob("part-*.npy")):
        yield np.load(path, mmap_mode="r", allow_pickle=False)

