"""Compiled exact candidate search; orchestration lives in exact.py.

Positive candidates use prefix postings and conservative positional bounds.
Negative candidates use the feasible minor-support fallback. Both verify packed
bit overlaps before returning bounded batches; neither emits graph shards.
"""
import numpy as np
from numba import njit

@njit(cache=True, nogil=True)
def _popcount64(value):
    """Count set bits in one uint64 word inside compiled kernels."""
    count = 0
    while value:
        count += int(value & np.uint64(1))
        value >>= np.uint64(1)
    return count


@njit(cache=True, nogil=True)
def _overlap_words(left, right):
    """Return the intersection cardinality of two packed bit rows."""
    total = 0
    for word in range(left.shape[0]):
        total += _popcount64(left[word] & right[word])
    return total


@njit(cache=True, nogil=True)
def _fill_ordered_tokens(unique_bits, token_order, offsets, tokens):
    """Fill CSR token storage in a shared global row-frequency order."""
    for group in range(unique_bits.shape[0]):
        output = offsets[group]
        for rank in range(token_order.shape[0]):
            row = int(token_order[rank])
            if (unique_bits[group, row // 64] >> np.uint64(row % 64)) & np.uint64(1):
                tokens[output] = row
                output += 1


@njit(cache=True, nogil=True)
def _build_postings(group_ids, offsets, tokens, prefix_length, token_count):
    """Build token-to-local-record postings for one exact-support bin."""
    counts = np.zeros(token_count, dtype=np.int64)
    for local in range(group_ids.shape[0]):
        start = offsets[group_ids[local]]
        for position in range(prefix_length):
            counts[tokens[start + position]] += 1
    posting_offsets = np.empty(token_count + 1, dtype=np.int64)
    posting_offsets[0] = 0
    for token in range(token_count):
        posting_offsets[token + 1] = posting_offsets[token] + counts[token]
    posting_ids = np.empty(posting_offsets[-1], dtype=np.uint32)
    posting_positions = np.empty(posting_offsets[-1], dtype=np.uint16)
    cursor = posting_offsets[:-1].copy()
    for local in range(group_ids.shape[0]):
        start = offsets[group_ids[local]]
        for position in range(prefix_length):
            token = tokens[start + position]
            place = cursor[token]
            posting_ids[place] = local
            posting_positions[place] = position
            cursor[token] += 1
    return posting_offsets, posting_ids, posting_positions


@njit(cache=True, nogil=True)
def _positive_matches_batch(
    query_groups,
    query_start,
    query_support,
    query_prefix,
    index_support,
    required_overlap,
    self_join,
    index_groups,
    token_offsets,
    tokens,
    posting_offsets,
    posting_ids,
    posting_positions,
    unique_bits,
    marks,
    touched,
    match_local,
    match_overlap,
    output_left,
    output_right,
    output_overlap,
):
    """Process consecutive PPJoin queries into one bounded relation batch."""
    output_count = processed = candidate_count = prefix_hits = 0
    for query_number in range(query_start, query_groups.shape[0]):
        query_group = int(query_groups[query_number])
        touched_count = local_hits = 0
        query_offset = token_offsets[query_group]
        epoch = query_number + 1
        for query_position in range(query_prefix):
            token = tokens[query_offset + query_position]
            for place in range(posting_offsets[token], posting_offsets[token + 1]):
                local_hits += 1
                local = int(posting_ids[place])
                indexed_group = int(index_groups[local])
                if self_join and indexed_group >= query_group:
                    continue
                if marks[local] == epoch:
                    continue
                indexed_position = int(posting_positions[place])
                remaining_bound = 1 + min(
                    query_support - query_position - 1,
                    index_support - indexed_position - 1,
                )
                if remaining_bound < required_overlap:
                    continue
                marks[local] = epoch
                touched[touched_count] = local
                touched_count += 1

        matches = 0
        for item in range(touched_count):
            local = int(touched[item])
            overlap = _overlap_words(
                unique_bits[query_group], unique_bits[index_groups[local]]
            )
            if overlap >= required_overlap:
                match_local[matches] = local
                match_overlap[matches] = overlap
                matches += 1
        if output_count + matches > output_left.shape[0]:
            for item in range(touched_count):
                marks[int(touched[item])] = 0
            return processed, output_count, candidate_count, prefix_hits
        for match in range(matches):
            output_left[output_count] = query_group
            output_right[output_count] = index_groups[match_local[match]]
            output_overlap[output_count] = match_overlap[match]
            output_count += 1
        processed += 1
        candidate_count += touched_count
        prefix_hits += local_hits
    return processed, output_count, candidate_count, prefix_hits


@njit(cache=True, nogil=True)
def _negative_matches_batch(
    query_groups,
    query_start,
    query_support,
    index_support,
    upper_overlap,
    self_join,
    index_groups,
    unique_bits,
    match_local,
    match_overlap,
    output_left,
    output_right,
    output_overlap,
):
    """Process consecutive negative-fallback queries into one relation batch."""
    output_count = processed = checked_total = 0
    for query_number in range(query_start, query_groups.shape[0]):
        query_group = int(query_groups[query_number])
        matches = checked = 0
        for local in range(index_groups.shape[0]):
            indexed_group = int(index_groups[local])
            if self_join and indexed_group >= query_group:
                continue
            checked += 1
            overlap = _overlap_words(unique_bits[query_group], unique_bits[indexed_group])
            if overlap <= upper_overlap:
                match_local[matches] = local
                match_overlap[matches] = overlap
                matches += 1
        if output_count + matches > output_left.shape[0]:
            return processed, output_count, checked_total
        for match in range(matches):
            output_left[output_count] = query_group
            output_right[output_count] = index_groups[match_local[match]]
            output_overlap[output_count] = match_overlap[match]
            output_count += 1
        processed += 1
        checked_total += checked
    return processed, output_count, checked_total
