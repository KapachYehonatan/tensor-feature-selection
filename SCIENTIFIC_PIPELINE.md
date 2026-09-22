# Scientific pipeline

This is the living scientific description of the current real-data pipeline.
It should change whenever a preprocessing rule, transformation, clustering
rule, selection procedure, model, or evaluation design changes.

## 1. Input and cohort definition

For sample $i$, the input contains nonnegative abundances for observed
gene–bacterium pairs $e=(j,k)$:

$$
X\in\mathbb R_+^{n\times m},\qquad e\mapsto (b_e,g_e).
$$

This sparse matrix represents the observed entries of the conceptual tensor;
the dense bacteria × gene tensor is never formed. Preprocessing keeps only
features named `gene|bacterium`, removes missing or `unclassified` bacteria,
and discards non-pair aggregate rows. It retains only healthy samples and the
configured disease class, encoded as $Y=0$ and $Y=1$. Pair columns absent
from every retained sample are removed. A sample with no remaining classified
abundance causes the analysis to stop rather than being imputed.

`HMP_2019_ibdmdb_unique` and `NielsenHB_2014_unique` are cross-sectional
versions of longitudinal sources. They retain one stool sample collected at
day zero per participant. If several day-zero samples exist, the one with the
greatest read count is retained; sample ID breaks an exact tie.

The prevalence count reported during preprocessing is only a whole-dataset
preview. It does not determine the features used for prediction.

## 2. Holdout and prevalence filtering

Evaluation uses one stratified 80/20 train/test split with seed 42. All rules
below—including prevalence filtering, clustering, feature selection, scaling,
and residualization—are fitted using training samples only.

A pair $e$ is retained when its training detection support satisfies

$$
\sum_{i\in\mathcal T}\mathbf{1}(X_{ie}>0)
\geq \max\!\left(5,\left\lceil0.05|\mathcal T|\right\rceil\right).
$$

The same retained columns are then used in the test set. Filtering may remove
rare biological signals. Each sample is closed to unit retained mass:

$$
\widetilde X_{ie}=
\frac{X_{ie}}{\sum_{h\in F_{\mathcal T}}X_{ih}}.
$$

Consequently, all later features are relative to the mass left after
training-derived filtering; changing the filter can change every normalized
value.

## 3. Bacteria and gene co-clustering

The initial bacteria and gene marginals are

$$
B_{ij}=\sum_{e:b_e=j}\widetilde X_{ie},\qquad
G_{ik}=\sum_{e:g_e=k}\widetilde X_{ie}.
$$

Bacteria are first clustered from the sample profiles $\sqrt{B_{\cdot j}}$.
The algorithm then alternates between dimensions. Given bacteria clusters
$c_B$, gene $k$ has the contextual profile

$$
P^G_{(i,r),k}=
\sqrt{\sum_{e:g_e=k,\ c_B(b_e)=r}\widetilde X_{ie}}.
$$

Given gene clusters $c_G$, bacterium $j$ analogously has

$$
P^B_{(i,c),j}=
\sqrt{\sum_{e:b_e=j,\ c_G(g_e)=c}\widetilde X_{ie}}.
$$

Thus samples are preserved, while the opposite biological dimension supplies
the context. For two profiles $u,v$, let

$$
S_u=\{a:u_a>0\},\qquad
J(u,v)=\frac{|S_u\cap S_v|}{|S_u\cup S_v|}.
$$

Under the predictive pipeline defaults, a feature may join a representative
only if

$$
J(u,v)\geq0.80
\quad\text{and}\quad
\operatorname{corr}(u,v)\geq0.93.
$$

Pearson correlation includes implicit zeros. Bray–Curtis similarity,
$2\sum_a\min(u_a,v_a)/(\sum_a u_a+\sum_a v_a)$, is available as a
sensitivity alternative. The candidate search is exact with respect to the
Jaccard rule; it is only a computational filter and does not approximate the
reported similarities.

No cluster count is chosen. Features are processed from lowest support to
highest and assigned to the qualifying representative with greatest abundance
similarity, then greatest Jaccard similarity. Every member therefore passes
both thresholds directly against its representative; similarity chains are
not used. This deterministic ordering can affect the resulting partition.
Alternation stops when both label vectors are unchanged or after four rounds.
There is no global co-clustering objective or guarantee of convergence before
that limit.

The cluster-level marginal features are sums of their member marginals:

$$
C^B_{ir}=\sum_{j:c_B(j)=r}B_{ij},\qquad
C^G_{ic}=\sum_{k:c_G(k)=c}G_{ik}.
$$

The standalone script in `clustering/` is descriptive and uses the complete
dataset, correlation threshold 0.95, and up to four rounds. Its clusters are
not reused for prediction; feature selection relearns clusters from its
training split using the rules above.

## 4. Marginal feature selection

For every bacteria-cluster and gene-cluster feature, the pipeline applies a
two-sided asymptotic Mann–Whitney test to $\sqrt{C}$, comparing $Y=0$ with
$Y=1$. Zeros remain observations. Its rank-based effect summary is

$$
\mathrm{AUC}=\frac{U}{n_1n_0},\qquad
\mathrm{effect}=2\,\mathrm{AUC}-1.
$$

Benjamini–Hochberg adjustment is performed separately for bacteria clusters
and gene clusters. Every feature with adjusted $q\leq0.10$ is retained; there
is no fixed feature-count cap. When results are reported as individual bacteria
or genes, selecting a cluster selects all its members.

The BH values describe these marginal screens; they are not post-selection
p-values for the final model.

## 5. Conditional pair selection

Selected marginal features are not deleted from the tensor. Instead, the
pipeline asks whether pairs add predictive information beyond marginal and
co-cluster-block structure.

First, three-fold out-of-fold predictions from the class-balanced L2 logistic
model described below give the training outcome residual

$$
r_i^Y=Y_i-\widehat{P}_{-f(i)}(Y_i=1).
$$

Each retained pair belongs to the learned block
$h(e)=(c_B(b_e),c_G(g_e))$. Define

$$
V_{ie}=\sqrt{\widetilde X_{ie}},\qquad
Z_{ih}=\sum_{e:h(e)=h}V_{ie}.
$$

Every pair receives the partial correlation

$$
\left|\operatorname{corr}(V_{\cdot e},r^Y\mid Z_{\cdot h(e)})\right|.
$$

The usual partial-correlation $t$ approximation provides a $p$-value for every
prevalence-retained pair. Benjamini–Hochberg adjustment is applied once across
the complete pair family, and every pair with $q\leq0.10$ is selected. There is
no per-block candidate cap or L1 selection stage. Pair BH correction is
separate from the bacteria- and gene-cluster BH families.

This is adjustment for one learned block aggregate plus an out-of-fold
marginal outcome model. It is not a copula decomposition or a formal test of
conditional independence. The selected raw square-root pair values are used
for prediction alongside the marginal clusters; null and label-permutation
experiments assess false discoveries empirically.

## 6. Prediction and interpretation

Separate class-balanced L2 logistic models with $C=1$ compare:

- selected bacteria-cluster features;
- selected gene-cluster features;
- both selected marginal levels;
- selected partial-BH pairs alone, when any are selected;
- selected marginals plus selected partial-BH pairs;
- an intercept-only reference.

Scaling is learned on the training set; sparse features are scaled but not
mean-centered. Performance is measured once on the untouched holdout using
AUROC and average precision. The scientific evidence for pair-level value is
improvement of marginals plus pairs over marginals alone, supported by pair-only
behavior and negative controls. A single fixed split does not establish
selection stability, calibrated probabilities, external validity, or causal
effects.

## 7. Synthetic data generation

Synthetic tensors have balanced, randomly permuted bacteria clusters
$B_1,\ldots,B_R$ and gene clusters $G_1,\ldots,G_C$. Every block receives a
prevalence

$$
\rho_{rc}\in\{0.01,0.05,0.10,0.20,0.40\}
$$

with respective proportions $(0.20,0.40,0.15,0.20,0.05)$. For each sample and
block,

$$
H_{irc}\sim\operatorname{Bernoulli}(\rho_{rc}),\qquad
a_{ir}\sim N(0,\sigma_B^2),\qquad g_{ic}\sim N(0,\sigma_G^2).
$$

An all-zero sample has its block detections redrawn. For bacterium $j\in B_r$
and gene $k\in G_c$, the noisy abundance is

$$
X_{ijk}=H_{irc}D_{ijk}
\exp(\mu+a_{ir}+g_{ic}+\epsilon_{ijk}),
$$

where $D_{ijk}\sim\operatorname{Bernoulli}(1-d)$ and
$\epsilon_{ijk}\sim N(0,\sigma_\epsilon^2)$. Current defaults are $d=0.05$,
$\sigma_B=\sigma_G=0.6$, and $\sigma_\epsilon=0.2$, producing approximately
91% zeros overall but substantially different prevalences among blocks. The
`exact_blocks` scenario uses the same block detections and latent effects with
$d=0$ and $\sigma_\epsilon=0$; `noisy_blocks` uses the noisy tensor. Neither has
a target.

All four target scenarios reuse exactly the same noisy tensor. After closing
each complete synthetic sample to unit mass, true bacteria- and gene-cluster
features are constructed exactly as in the analysis and square-root
transformed. Planted pair predictors are residualized against their true
bacteria-cluster × gene-cluster block as in Section 5. Every selected target
predictor is standardized over the generated samples.

Signal pairs use distinct bacteria and genes, lie outside the marginal-signal
clusters, pass the full-data prevalence rule, and come from sufficiently common
blocks. Their signs are balanced. The logistic score is

$$
\eta_i=\alpha+
\sum_{r\in\mathcal S_B}\beta_r Z(C^B_{ir})+
\sum_{c\in\mathcal S_G}\gamma_c Z(C^G_{ic})+
\sum_{e\in\mathcal S_P}\delta_e Z(R_{ie}),
$$

followed by

$$
p_i=\operatorname{sigmoid}(\eta_i),\qquad
Y_i\sim\operatorname{Bernoulli}(p_i).
$$

The intercept $\alpha$ is chosen so that the mean $p_i$ is 0.5; realized class
balance is not forced. The same uniform random draws generate all target
scenarios, making their outcomes matched comparisons:

- `null_target`: $p_i=0.5$, with no predictive features;
- `marginal_signals`: bacteria- and gene-cluster terms only;
- `pair_signals`: residual-pair terms only;
- `full_signals`: both marginal and residual-pair terms.

The current main preset uses $p=50$, $q=10{,}000$, $R=10$, $C=100$, four pair
signals, sample sizes 150, 250, 350, and 500, and seven independent seeds. Its
pair coefficients are deliberately strong ($|\delta_e|=5$), so it is first a
recoverability check rather than a realistic effect-size claim.

## 8. Synthetic evaluation

Each target dataset is passed to the same public fixed-holdout pipeline used
for real data; the algorithm is not reimplemented in the evaluator. The two
target-free scenarios run clustering on a reproducible random training split.
Results measure:

- cluster agreement with truth using ARI and NMI, recovered cluster count,
  active-entity coverage, and the fraction of active entities left as
  singletons;
- exact sparse candidate-search recall against exhaustive representative
  search for all bacteria and a truth-stratified subset of at most 500 genes;
- training-filter retention of planted features;
- exact feature recovery separately for bacteria, genes, and pairs;
- held-out AUROC and average precision for every prediction model;
- oracle AUROC and average precision from the probabilities that generated
  $Y$;
- sparsity, pair-prevalence quantiles, runtime, and peak process memory.

For any level, with selected set $S$ and planted set $T$,

$$
\operatorname{precision}=\frac{|S\cap T|}{|S|},\qquad
\operatorname{recall}=\frac{|S\cap T|}{|T|},\qquad
\operatorname{FDP}=\frac{|S\setminus T|}{|S|}.
$$

When no feature is selected, FDP is recorded as zero and precision is
undefined. Bacteria and gene truth contains every member of a planted cluster;
pair recovery requires the exact planted $(j,k)$ coordinate, with no credit for
a correlated neighbor. A planted feature removed by training prevalence
filtering counts as a false negative, while its separate retention rate shows
where the loss occurred. Seed-level results remain visible and are summarized
by their mean and standard deviation; averaging FDP across seeds is an
empirical estimate of false discovery rate, not a formal guarantee.

## 9. Top-K method comparison

The comparison reuses the training-only holdout split, prevalence filter,
closure, square-root transformation, and prediction model. New reports compare
the joint pair-plus-bacteria family, bacteria averages, gene averages,
co-clustered marginals plus partial-BH pairs, the
Bernoulli-correlation-join, and the Jaccard join at

$$
K\in\{5,10,20,35,50,60,75,85,100\}.
$$

Collapsed bacteria and gene features are averages of their retained pair
columns. The joint method concatenates transformed pairs and collapsed
bacteria, then applies one BH correction across that combined family. Every
method retains only features with training-set $q\leq0.10$ and
ranks them by adjusted $q$, raw $p$, feature type, and stable identifier. The
marginal co-clustering method applies BH separately to bacteria and gene
clusters. Its extension keeps those exact marginal features and q-values, then
adds pairs passing the separate all-pair partial-correlation BH test. Every
result records bacteria, gene, pair, and total model columns.

Held-out AUROC, precision, recall, and F1 use the current class-balanced L2
logistic model; binary metrics threshold its probability at 0.5. RSP is
reported for all six displayed BH-defined methods:

$$
\operatorname{RSP}=\frac{0.05RP-SP}{0.05RP+SP},
$$

where $RP$ is the number of $q\leq0.05$ discoveries under real training labels
and $SP$ is their mean count over five fixed label permutations. Synthetic
comparisons use `full_signals` only and average seeds within sample size.

Corrected synthetic discoveries are also scored within each method's
post-filtering hypothesis family:

$$
\operatorname{FDP}=\frac{FP}{TP+FP},\qquad
\operatorname{FPR}=\frac{FP}{N},\qquad
\operatorname{power}=\frac{TP}{P}.
$$

Individual bacteria and genes are positive when they belong to a planted
marginal-signal cluster; pairs require the exact planted coordinate. A
recovered co-cluster is positive when it contains at least one planted marginal
member. Empty selections have FDP and FPR zero.

### Bernoulli-correlation-join

This real-data-only method treats every retained bacteria--gene pair as an
ordinary feature; bacteria and gene identities are not used to form clusters.
On the training samples it binarizes detection,

$$
Z_{ie}=\mathbf 1(X_{ie}>0),
$$

and the external exact join finds every edge satisfying

$$
|\widehat\phi_{ef}|\geq \tau,
\qquad
\widehat\phi_{ef}=
\frac{n a_{ef}-s_es_f}
{\sqrt{s_es_f(n-s_e)(n-s_f)}}.
$$

The requested threshold is $\tau=0.70$. Before Leiden, the graph is limited to
500 million edges, independently of the 125 GiB process limit and the
partition-table memory estimate. Larger graphs are filtered to the smallest
threshold in $\{0.75,0.80,\ldots,1.00\}$ that satisfies the cap. If an actual
Leiden allocation still fails, the next threshold is tried. This filtering is
exact: stored verified edges are streamed and only edges below the higher
threshold are removed, without recomputing similarities. The effective
threshold is recorded for every dataset.

Thus the join is exactly Pearson correlation between standardized Bernoulli
columns, computed from supports and intersections without forming a correlation
matrix. Weighted Leiden communities define feature clusters. An all-one
training column has undefined correlation and remains a singleton; an all-zero
column after prevalence filtering is an error.

The learned membership is applied to the original, nonbinary relative
abundances. For cluster $G_h$ the tested feature is

$$
V_{ih}=\sqrt{\frac{1}{|G_h|}\sum_{e\in G_h}\widetilde X_{ie}}.
$$

Mann--Whitney tests and one BH correction are then applied across recovered
clusters. Features with $q\leq0.10$ are ranked by adjusted p-value, raw p-value,
and stable cluster ID. All filtering, standardization, graph construction, and
clustering use training samples only.

### Jaccard join

This real-data-only baseline uses the same retained pair detection columns and
the external exact PPJoin implementation, but skips the minor-state and
standardized-correlation reduction. It connects two pair features when

$$
J(S_e,S_f)=\frac{|S_e\cap S_f|}{|S_e\cup S_f|}\geq \tau,
\qquad S_e=\{i:X_{ie}>0\}.
$$

It starts at $\tau=0.70$, and weighted Leiden uses the exact Jaccard values.
Memberships are applied to the original relative abundances with the same mean
aggregation, square-root transform, Mann--Whitney tests, BH correction, Top-K
prediction, resource guard, and exact threshold fallback. The only scientific
difference between the two join methods is the binary similarity used to build
the graph.
