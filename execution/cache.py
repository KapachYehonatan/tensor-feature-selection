"""Stage-local, strict checkpoints shared by experiments."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import inspect
import json
import platform
from importlib.metadata import version

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / "cache"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sparse_hash(matrix, values=True):
    matrix = matrix.tocsr()
    return dict(shape=matrix.shape, indices=array_hash(matrix.indices), indptr=array_hash(matrix.indptr),
                **(dict(data=array_hash(matrix.data)) if values else {}))


def identity(stage, inputs, settings, functions=(), files=(), software=()):
    configuration = asdict(settings) if is_dataclass(settings) else settings
    record = dict(stage=stage, inputs=inputs, settings=configuration,
                  code={f"{f.__module__}.{f.__name__}": hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                        for f in functions},
                  files={str(p): file_hash(p) for p in files},
                  software=dict(python=platform.python_version(), **{name: version(name) for name in software}))
    record = json.loads(json.dumps(record, sort_keys=True))
    signature = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return signature, record


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def metric_protocol(experiment):
    modules = {"prediction": "prediction.py", "correlation": "correlation.py", "shuffled_fdp": "shuffled_fdp.py"}
    files = tuple(Path(__file__).resolve().parents[1] / "metrics" / modules[m]
                  for m in experiment.metrics if m in modules)
    software = ("numpy", "scipy", "pandas", "scikit-learn") if "prediction" in experiment.metrics else ("numpy", "scipy")
    return identity("metrics", {}, dict(metrics=experiment.metrics, evaluation=asdict(experiment.evaluation)),
                    (metric_protocol,), files, software)


def complete(directory, signature):
    marker = Path(directory) / "complete.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if saved["signature"] != signature:
        raise ValueError(f"Incompatible stage: {directory}")
    for name, size, modified in saved["files"]:
        path = Path(directory) / name
        if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != (size, modified):
            raise ValueError(f"Changed stage artifact: {path}")
    return True


def stage_path(dataset, scope, stage, signature, record):
    if any(not str(value) or str(value).startswith(".") or "/" in str(value) or "\\" in str(value)
           for value in (dataset, scope, stage, signature)):
        raise ValueError("Invalid stage path component")
    directory = CACHE / "stages" / dataset / scope / stage / signature
    protocol = directory / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != record:
        raise ValueError(f"Changed stage protocol: {protocol}")
    if not protocol.exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Unidentified stage directory: {directory}")
        write_json(protocol, record)
    return directory


def seal(directory, signature, files):
    directory = Path(directory)
    write_json(directory / "complete.json", dict(signature=signature, files=[
        (str(p.relative_to(directory)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]))
