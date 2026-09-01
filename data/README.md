# Data provenance

Source datasets are read directly and never modified:

`/home/kapachy/Microbium/projects/cMD_downloads`

Each study contains a Matrix Market coordinate matrix, ordered feature and
sample text files, and clinical metadata. Many studies also contain previously
cleaned SciPy sparse matrices and earlier experiment outputs. The primary
prevalence analysis starts from the raw Matrix Market data because the existing
cleaned files appear to use a stricter 15% prevalence threshold.

The preprocessing script records the selected source files, dimensions, target
mapping, and filter settings in its output summary. No source file is copied
into this directory.

