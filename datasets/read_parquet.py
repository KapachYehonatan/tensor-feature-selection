"""Extract one cohort from the public stratified table using the installed Arrow reader."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse


def extract(source, metadata_path, destination):
    metadata = pd.read_csv(metadata_path, dtype=str)
    if metadata.uuid.isna().any() or metadata.uuid.duplicated().any():
        raise ValueError('Invalid sample UUID mapping')
    sample_index = {value: i for i, value in enumerate(metadata.uuid)}
    table = pq.ParquetFile(source)
    columns = ['uuid', 'gene_family', 'rpk_abundance']
    if not set(columns) <= set(table.schema_arrow.names):
        raise ValueError('Unexpected species-stratified relative-abundance schema')
    uuid_column = table.schema_arrow.names.index('uuid')
    genes, values, rows, samples, seen = {}, [], [], [], set()
    keys = list(sample_index)
    for group in range(table.num_row_groups):
        statistics = table.metadata.row_group(group).column(uuid_column).statistics
        if statistics is not None and statistics.has_min_max:
            if not any(statistics.min <= key <= statistics.max for key in keys):
                continue
        for batch in table.iter_batches(row_groups=[group], columns=columns, batch_size=100_000, use_threads=False):
            chosen = batch.filter(pc.is_in(batch.column('uuid'), value_set=pc.unique(metadata.uuid.tolist())))
            if not len(chosen):
                continue
            frame = chosen.to_pandas()
            valid = frame.gene_family.notna() & frame.gene_family.str.contains('|', regex=False)
            frame = frame.loc[valid]
            frame = frame.loc[~frame.gene_family.str.endswith('|unclassified')]
            if not len(frame):
                continue
            v = frame.rpk_abundance.to_numpy(dtype=float)
            if not np.isfinite(v).all() or np.any(v < 0):
                raise ValueError('Invalid supplied relative abundance')
            seen.update(frame.uuid)
            rows.append(np.fromiter((genes.setdefault(g, len(genes)) for g in frame.gene_family), np.int32))
            samples.append(frame.uuid.map(sample_index).to_numpy(dtype=np.int32))
            values.append(v)
        if group % 100 == 0:
            print(f'JoS_2022: Parquet row group {group + 1}/{table.num_row_groups}', flush=True)
    if seen != set(sample_index) or not values:
        raise ValueError('Selected cohort samples missing from abundance table')
    matrix = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(samples))),
                               shape=(len(genes), len(metadata))).tocsr()
    if matrix.nnz != sum(map(len, values)):
        raise ValueError('Duplicate sample–pair entries in the public table')
    output = Path(destination)
    sparse.save_npz(output / 'values.npz', matrix)
    (output / 'features.txt').write_text(''.join(g + '\n' for g in genes))
    print(f'JoS_2022: extracted {len(metadata)} samples and {len(genes)} classified pairs', flush=True)


if __name__ == '__main__':
    extract(*sys.argv[1:])
