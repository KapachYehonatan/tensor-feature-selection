# Microbiome gene-family datasets

Source documentation for human stool shotgun-metagenomic datasets download from cMD. These are processed abundance tables,
not raw sequencing reads.

Each study's four files are stored in `cMD_downloads/<study>/` alongside this
README. Download receipts and original source containers are kept privately in
`cMD_downloads/.sources/`. The data folder is ignored by Git.

## Sources and upstream processing

**cMD** means [curatedMetagenomicData](https://waldronlab.io/curatedMetagenomicData/articles/curatedMetagenomicData.html),
downloaded through Bioconductor ExperimentHub; clinical metadata come from its
manually curated sample metadata. The source reprocessed sequencing data using
MetaPhlAn3 and HUMAnN3 with UniRef90 gene families. Its `gene_families` resources
are assembled from HUMAnN relative-abundance tables: gene-length adjustment and
sample abundance normalization have already occurred upstream. These are not
integer read counts. See the [source pipeline](https://waldronlab.io/curatedMetagenomicData/articles/our-pipeline.html)
and [gene-family table assembly](https://github.com/waldronlab/curatedMetagenomicDataPipeline/blob/main/R/read_dir.R).

**ASAP-MAC** means [parkinsonsMetagenomicData](https://asap-mac.github.io/parkinsonsMetagenomicData/).
JoS_2022 abundances were downloaded from the public
[metagenomics_mac repository](https://huggingface.co/datasets/waldronlab/metagenomics_mac/tree/781465e08de1bf565fb9c3b3a4bbcf1c9e13cb7e),
with metadata from the ASAP-MAC package. Its standardized pipeline includes
quality checks, quality/host-read filtering, and HUMAnN functional profiling.
The downloaded table is species-stratified
`genefamilies_relab_stratified_uuid.parquet`. The value column is named
`rpk_abundance` despite the file's `relab` designation; the supplied values
have not been converted to another unit.

## What the four files contain

| File | Contents |
|---|---|
| `<study>_gene_families.mtx` | Sparse abundance matrix: **features in rows, samples in columns**. Unlisted entries are zero. |
| `<study>_genes.txt` | Feature IDs. A classified feature such as `UniRef90_…|g__…s__…` represents a gene family attributed to a species. |
| `<study>_samples.txt` | Sample IDs.|
| `<study>_metadata.tsv` | Tab-separated sample annotations, including identifiers, clinical conditions and other available study variables; matched to the sample IDs. |

## Datasets

Counts describe samples in the source tables, not necessarily independent
participants.

| Study | Download source | Source samples | Clinical conditions available |
|---|---|---:|---|
| FengQ_2015 | cMD | 154 | Colorectal cancer, adenoma and healthy controls. |
| HMP_2019_ibdmdb | cMD | 1,628 | Inflammatory bowel disease and healthy controls; longitudinal sampling. |
| JieZ_2017 | cMD | 385 | Atherosclerotic cardiovascular disease and healthy controls. |
| JoS_2022 | ASAP-MAC | 156 | Parkinson's disease and study controls. |
| KarlssonFH_2013 | cMD | 145 | Type 2 diabetes, impaired glucose tolerance and healthy controls. |
| LiJ_2017 | cMD | 196 | Hypertension, pre-hypertension and healthy controls. |
| MetaCardis_2020_a | cMD | 1,831 | Diabetes, metabolic syndrome, impaired glucose tolerance, coronary artery disease, heart failure, combinations of these conditions and healthy controls. |
| NagySzakalD_2017 | cMD | 100 | Myalgic encephalomyelitis/chronic fatigue syndrome and healthy controls. |
| NielsenHB_2014 | cMD | 396 | Inflammatory bowel disease and healthy controls; includes repeated samples. |
| QinJ_2012 | cMD | 363 | Type 2 diabetes and healthy controls. |
| QinN_2014 | cMD | 237 | Cirrhosis and healthy controls. |
| RubelMA_2020 | cMD | 175 | Soil-transmitted helminth infection and healthy controls. |
| WirbelJ_2018 | cMD | 125 | Colorectal cancer and healthy controls. |
| YachidaS_2019 | cMD | 616 | Colorectal cancer, adenoma, previous carcinoma surgery, few-polyps category and healthy controls. |
| YeZ_2018 | cMD | 65 | Behçet's disease and healthy controls. |
| ZhuF_2020 | cMD | 171 | Schizophrenia and healthy controls. |
