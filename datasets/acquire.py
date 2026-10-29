"""Acquire the eight approved cohorts and export unchanged sparse supplied values."""
from pathlib import Path
import hashlib
import gzip
import json
import subprocess
import tempfile
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from execution import cache
from datasets.catalog import NEW_DATASETS
from datasets.loading import load
from datasets.prepare import prepare, prepare_matrix, pair_mapping

SOURCE = Path('/home/kapachy/Microbium/projects/cMD_downloads/multilabel')
R = Path('/home/kapachy/.venvs/taxsea-r/bin/Rscript')
PARQUET_PYTHON = Path('/home/kapachy/miniforge3/envs/yamas_shotgun/bin/python')
DOWNLOADS = cache.CACHE / 'downloads'
CMD_FILES = 'https://raw.githubusercontent.com/waldronlab/curatedMetagenomicData/afd13adf72a4cda3c60ce60e7d5f20ef04bd5f8e/inst/extdata/metadata.csv'
CMD_METADATA = 'https://raw.githubusercontent.com/waldronlab/curatedMetagenomicDataCuration/ffea85122e43c7c047451636f84dd823ba52c077/inst/extdata/sampleMetadata.csv'
HUB = 'https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/'
HF = 'https://huggingface.co/datasets/waldronlab/metagenomics_mac/resolve/781465e08de1bf565fb9c3b3a4bbcf1c9e13cb7e/'
PD_METADATA = 'https://raw.githubusercontent.com/ASAP-MAC/parkinsonsMetagenomicData/46cc48621cd88ed802b75b6209ce419c118399b6/data/sampleMetadata.rda'
# Source target column, negative/positive labels, and approved negative/positive counts.
COHORTS = {
    'YachidaS_2019': ('disease', 'healthy', 'CRC', 246, 258),
    'MetaCardis_2020_a': ('disease', 'healthy', 'T2D', 230, 537),
    'QinN_2014': ('study_condition', 'control', 'cirrhosis', 114, 123),
    'KarlssonFH_2013': ('study_condition', 'control', 'T2D', 43, 53),
    'NagySzakalD_2017': ('study_condition', 'control', 'ME/CFS', 50, 50),
    'FengQ_2015': ('study_condition', 'control', 'CRC', 61, 46),
    'YeZ_2018': ('study_condition', 'control', 'BD', 45, 20),
    'JoS_2022': ('control', 'Study Control', 'Case', 74, 82),
}


@lru_cache(None)
def conversion_versions():
    r = subprocess.check_output([str(R), '--version'], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    arrow = subprocess.check_output([str(PARQUET_PYTHON), '-c',
        'import json,pyarrow,numpy,scipy;print(json.dumps(dict(pyarrow=pyarrow.__version__,numpy=numpy.__version__,scipy=scipy.__version__)))'], text=True)
    return dict(R=r, parquet=json.loads(arrow))


def download(url, path):
    """Resume interrupted transfers; validate completed downloads before reuse."""
    path = Path(path)
    receipt = path.with_name(path.name + '.download.json')
    if path.exists():
        saved = json.loads(receipt.read_text())
        if saved['url'] != url or saved['sha256'] != cache.file_hash(path):
            raise ValueError(f'Changed downloaded source: {path}')
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    print(f'Downloading {url} -> {path}', flush=True)
    subprocess.run(['curl', '--fail', '--location', '--continue-at', '-', '--retry', '5',
        '--retry-all-errors', '--retry-delay', '5', '--connect-timeout', '30', '--silent', '--show-error',
        '--output', str(partial), url], check=True)
    saved = dict(url=url, sha256=cache.file_hash(partial), bytes=partial.stat().st_size)
    cache.write_json(receipt, saved)
    partial.replace(path)
    print(f'Download complete: {path.name}; {saved["bytes"]:,} bytes; sha256={saved["sha256"]}', flush=True)
    return saved


def read_sparse_rds(path, directory):
    extractor = Path(__file__).with_name('read_cmd.R')
    signature, record = cache.identity('rds_conversion', {'rds': cache.file_hash(path)}, {}, files=(extractor,))
    if not cache.complete(directory, signature):
        subprocess.run([str(R), str(extractor), str(path), str(directory)], check=True)
        cache.write_json(directory / 'protocol.json', record)
        cache.seal(directory, signature, [p for p in directory.iterdir() if p.is_file() and p.name != 'complete.json'])
    shape = tuple(map(int, (directory / 'shape.txt').read_text().split()))
    values = np.memmap(directory / 'x.bin', dtype='<f8', mode='r')
    rows = np.memmap(directory / 'i.bin', dtype='<i4', mode='r')
    if (directory / 'class.txt').read_text().strip() == 'dgCMatrix':
        pointers = np.memmap(directory / 'p.bin', dtype='<i4', mode='r')
        matrix = sparse.csc_matrix((values, rows, pointers), shape=shape)
    else:
        columns = np.memmap(directory / 'j.bin', dtype='<i4', mode='r')
        matrix = sparse.coo_matrix((values, (rows, columns)), shape=shape)
    return matrix, (directory / 'features.txt').read_text().splitlines(), (directory / 'samples.txt').read_text().splitlines()


def check_counts(samples, cohort):
    if samples.sample_id.duplicated().any() or samples.subject_id.isna().any() or samples.subject_id.duplicated().any():
        raise ValueError('Cohort must have unique samples and participants')
    expected = COHORTS[cohort][3:]
    actual = tuple(int((samples.y == y).sum()) for y in (0, 1))
    if actual != expected:
        raise ValueError(f'{cohort}: labels changed: expected {expected}, found {actual}')


def metacardis_source(directory):
    """Apply the approved exclusion of samples without classified-pair abundance."""
    name = 'MetaCardis_2020_a'
    raw = mmread(directory / f'{name}_gene_families.mtx', spmatrix=True).tocsr()
    features = (directory / f'{name}_genes.txt').read_text().splitlines()
    ids = (directory / f'{name}_samples.txt').read_text().splitlines()
    metadata = pd.read_csv(directory / f'{name}_metadata.tsv', sep='\t', low_memory=False).set_index('sample_id').loc[ids]
    classified = np.fromiter(('|' in f and f.split('|', 1)[1] != 'unclassified' for f in features), dtype=bool)
    totals = np.asarray(raw[classified].sum(axis=0)).ravel()
    target = metadata.disease.isin(['healthy', 'T2D']).to_numpy()
    excluded = target & (totals == 0)
    observed = metadata.iloc[np.flatnonzero(excluded)].disease.value_counts().to_dict()
    if observed != {'T2D': 12, 'healthy': 4}:
        raise ValueError(f'MetaCardis zero-classified sample counts changed: {observed}')
    keep = np.flatnonzero(target & ~excluded)
    omitted = metadata.iloc[np.flatnonzero(excluded)].reset_index()[['sample_id', 'disease']]
    print(f'MetaCardis: excluding {len(omitted)} approved zero-classified samples; keeping {len(keep)}.', flush=True)
    return raw[:, keep], features, [ids[i] for i in keep], metadata.iloc[keep].reset_index(), omitted


def prepare_one(name, metadata, resources):
    target, negative, positive, _, _ = COHORTS[name]
    folder = DOWNLOADS / name
    folder.mkdir(parents=True, exist_ok=True)
    if name in {'YachidaS_2019', 'MetaCardis_2020_a'}:
        directory = SOURCE / name
        sources = {p.name: cache.file_hash(p) for p in directory.iterdir() if p.is_file()}
        kind = 'local_cmd'
    elif name == 'JoS_2022':
        sources = {
            filename: download(HF + filename, DOWNLOADS / filename)
            for filename in ('genefamilies_relab_stratified_uuid.parquet',)
        }
        sources['metadata'] = download(PD_METADATA, DOWNLOADS / 'parkinsons_sampleMetadata.rda')
        kind = 'parquet'
    else:
        selected = resources[resources.Title.astype(str).str.contains(name, regex=False)
                             & resources.Title.astype(str).str.contains('gene.families', case=False)].copy()
        if selected.empty:
            raise ValueError(f'No public gene-family resource for {name}')
        selected = selected.sort_values('RDataPath').iloc[-1]
        sources = {'rds': download(HUB + selected.RDataPath, folder / 'gene_families.rds'),
                   'metadata': cache.file_hash(DOWNLOADS / 'sampleMetadata.csv')}
        kind = 'rds'
    key, record = cache.identity('input_export', sources,
        dict(cohort=name, target_column=target, negative=negative, positive=positive,
             expected_counts=COHORTS[name][3:], normalization='none', transformation='none',
             source_units='relative abundance; preserved as supplied', conversion_software=conversion_versions(),
             **(dict(exclusion='zero classified-pair abundance; user approved 2026-10-28',
                     excluded_negative=4, excluded_positive=12) if name == 'MetaCardis_2020_a' else {}),
             target_note='study_condition is authoritative; one detailed disease entry lists hepatitis only' if name == 'QinN_2014' else ''),
        (prepare, prepare_matrix, pair_mapping, check_counts)
        + ((read_sparse_rds,) if kind == 'rds' else ())
        + ((metacardis_source,) if name == 'MetaCardis_2020_a' else ()),
        files=((Path(__file__).with_name('read_cmd.R'),) if kind == 'rds' else
               (Path(__file__).with_name('read_parquet.py'),) if kind == 'parquet' else ()),
        software=('numpy', 'scipy', 'pandas'))
    destination = cache.CACHE / 'inputs' / name
    if destination.exists():
        saved = json.loads((destination / 'protocol.json').read_text())
        saved_key = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
        if saved != record:
            # The first exports fingerprinted the R reader even when it was unused.
            compatible = dict(saved, code={k: v for k, v in saved['code'].items() if k in record['code']},
                              files={k: v for k, v in saved['files'].items() if k in record['files']})
            if kind == 'rds' or compatible != record:
                raise ValueError(f'Existing input export is incompatible: {destination}')
            cache.write_json(destination / 'compatibility.json', dict(
                original_signature=saved_key, compatible_signature=key,
                reason='Removed unused converter dependencies; all source, target, used-code and software fingerprints match.'))
        if not cache.complete(destination, saved_key):
            raise ValueError(f'Incomplete input export: {destination}')
        check_counts(pd.read_csv(destination / 'samples.csv'), name)
        print(f'{name}: compatible input export already complete', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import_', dir=destination.parent) as temporary:
        output = Path(temporary) / name
        if name == 'MetaCardis_2020_a':
            matrix, features, ids, source_metadata, omitted = metacardis_source(directory)
            prepare_matrix(name, matrix, features, ids, source_metadata, source_study=name,
                target_column=target, negative_label=negative, positive_label=positive, destination=output)
            omitted.to_csv(output / 'excluded_samples.csv', index=False)
        elif kind == 'local_cmd':
            prepare(name, directory, name, target_column=target, negative_label=negative,
                    positive_label=positive, destination=output)
        elif kind == 'rds':
            matrix, features, ids = read_sparse_rds(folder / 'gene_families.rds', folder / 'slots')
            prepare_matrix(name, matrix, features, ids, metadata[metadata.study_name.eq(name)].copy(),
                source_study=name, target_column=target, negative_label=negative,
                positive_label=positive, destination=output)
        else:
            subprocess.run([str(R), '-e',
                'args<-commandArgs(TRUE);e<-new.env();load(args[1],e);'
                'x<-as.data.frame(e$sampleMetadata);x<-x[x$study_name=="JoS_2022",];'
                'write.csv(x[,intersect(c("sample_id","subject_id","uuid","control","BioSample","NCBI_accession"),names(x))],args[2],row.names=FALSE)',
                str(DOWNLOADS / 'parkinsons_sampleMetadata.rda'), str(folder / 'metadata.csv')], check=True)
            subprocess.run([str(PARQUET_PYTHON), str(Path(__file__).with_name('read_parquet.py')),
                str(DOWNLOADS / 'genefamilies_relab_stratified_uuid.parquet'),
                str(folder / 'metadata.csv'), str(folder)], check=True)
            matrix = sparse.load_npz(folder / 'values.npz')
            samples = pd.read_csv(folder / 'metadata.csv', dtype=str)
            prepare_matrix(name, matrix, (folder / 'features.txt').read_text().splitlines(),
                samples.sample_id.tolist(), samples, source_study=name, target_column=target,
                negative_label=negative, positive_label=positive, destination=output)
        samples = pd.read_csv(output / 'samples.csv')
        check_counts(samples, name)
        cache.write_json(output / 'protocol.json', record)
        cache.seal(output, key, [p for p in output.iterdir() if p.is_file()])
        output.replace(destination)
    data = load(name)
    print(f'{name}: READY samples={len(data.samples)}, pairs={data.values.shape[1]}, '
          f'genes={len(np.unique(data.pair_gene))}, bacteria={len(np.unique(data.pair_bacteria))}', flush=True)


def prepare_all():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    download(CMD_METADATA, DOWNLOADS / 'sampleMetadata.csv')
    download(CMD_FILES, DOWNLOADS / 'resources.csv')
    metadata = pd.read_csv(DOWNLOADS / 'sampleMetadata.csv', low_memory=False)
    resources = pd.read_csv(DOWNLOADS / 'resources.csv')
    failures = []
    for name in NEW_DATASETS:
        print(f'{name}: acquiring/preparing approved cohort', flush=True)
        cache.write_json(DOWNLOADS / 'status.json', dict(dataset=name, stage='preparing'))
        try:
            prepare_one(name, metadata, resources)
        except Exception as error:
            import traceback
            traceback.print_exc()
            failures.append(dict(dataset=name, error=f'{type(error).__name__}: {error}'))
            pd.DataFrame(failures).to_csv(DOWNLOADS / 'failures.csv', index=False)
    if not failures:
        (DOWNLOADS / 'failures.csv').unlink(missing_ok=True)
    audit_overlap(metadata)
    cache.write_json(DOWNLOADS / 'status.json', dict(dataset=None, stage='incomplete' if failures else 'complete'))
    return failures


def audit_overlap(metadata):
    """Record shared sequencing accessions; cohorts are evaluated separately, never pooled."""
    records = []
    for directory in (cache.CACHE / 'inputs').iterdir():
        if not (directory / 'samples.csv').exists():
            continue
        samples = pd.read_csv(directory / 'samples.csv', dtype=str)
        if directory.name == 'JoS_2022':
            source = pd.read_csv(DOWNLOADS / 'JoS_2022/metadata.csv', dtype=str)
        else:
            source = metadata.loc[metadata.study_name.eq(directory.name.removesuffix('_unique'))]
        source = source.loc[source.sample_id.isin(samples.sample_id)]
        for _, row in source.iterrows():
            tokens = {str(row.sample_id)}
            for field in ('NCBI_accession', 'BioSample'):
                if field in row and pd.notna(row[field]):
                    tokens.update(str(row[field]).replace(',', ';').split(';'))
            for token in tokens:
                token = token.strip()
                if token.startswith(('SAMN', 'SAMEA', 'SAMD', 'SRR', 'ERR', 'DRR')):
                    records.append(dict(dataset=directory.name, sample_id=row.sample_id, accession=token))
    frame = pd.DataFrame(records, columns=['dataset', 'sample_id', 'accession']).drop_duplicates()
    shared = frame.groupby('accession').dataset.transform('nunique') > 1
    overlap = frame.loc[shared].sort_values(['accession', 'dataset'])
    overlap.to_csv(DOWNLOADS / 'cohort_overlap.csv', index=False)
    print(f'Cross-cohort accession audit: {overlap.accession.nunique()} shared accessions; cohorts remain separate.', flush=True)


if __name__ == '__main__':
    prepare_all()
