# E-Value Methods for Sparse Feature Selection

## Problem and assumptions

We observe a feature matrix $X\in\mathbb{R}^{n\times p}$ and binary labels $Y\in\{0,1\}^n$, typically with $p\gg n$. The features are sparse: many entries represent **structural absence**, while observed (present) entries contain abundance measurements that are **already log-transformed**. No additional log transformation is applied.

For each feature $j$, both methods test the **marginal independence** hypothesis

$$
H_{0,j}: X_j\perp Y.
$$

Assumptions:

- Samples $(X_i,Y_i)$ are independent and identically distributed (i.i.d.). Features may be arbitrarily dependent **across columns**.
- No parametric assumption is made about either the feature distribution or the distribution of abundances among present samples.
- The structural-zero/presence indicator is known. If log transformation can turn a present abundance into a numerical zero, presence must come from the original zero mask, **not** the transformed value.
- Test parameters and calibration choices are fixed in advance, or mixed using predetermined weights; they must not be chosen to maximize the observed association without a validity-preserving procedure.

### E-values and e-BH

An e-value $E_j$ is a nonnegative statistic satisfying

$$
\mathbb{E}_{H_{0,j}}[E_j]\leq 1.
$$

Given $p$ valid e-values, sort them $E_{(1)}\geq\cdots\geq E_{(p)}$ and define

$$
k^*=\max\left\{k\in\{1,\ldots,p\}: E_{(k)}\geq\frac{p}{\alpha k}\right\},
$$

with $k^*=0$ if the set is empty. **e-BH** rejects the $k^*$ hypotheses with the largest e-values. It controls the false discovery rate (FDR) at level $\alpha$ under **arbitrary dependence among valid e-values**.

---

## Method 1 — Mann–Whitney p-values → e-values → e-BH

**Idea:** Use a standard nonparametric marginal association test on the complete feature, including its zeros, and convert the resulting p-value to an e-value.

**Procedure for each feature $j$:**

1. Compare $X_j$ between $Y=0$ and $Y=1$ using a **two-sided Mann–Whitney U test**, including all structural zeros.
2. Obtain a valid p-value $P_j$. Because sparse features have many ties, use an appropriately tie-aware test; **label-permutation calibration** is preferable when finite-sample validity is important. A standard asymptotic implementation provides only approximate p-values.
3. Convert the p-value into an e-value with a prespecified $\gamma\in(0,1)$ (e.g., $\gamma=0.5$):

   $$
   \boxed{E_j=(1-\gamma)P_j^{-\gamma}.}
   $$

4. Apply e-BH to all $p$ e-values.

**Why is the conversion valid?** A valid p-value is *super-uniform* under the null: $\Pr(P_j\leq t)\leq t$. Consequently,

$$
\mathbb{E}_{H_{0,j}}[(1-\gamma)P_j^{-\gamma}]\leq
(1-\gamma)\int_0^1 u^{-\gamma}\,du=1.
$$

**Interpretation and limitation:** This is a simple baseline, but Mann–Whitney principally detects shifts in relative ordering, and the p-to-e conversion may be **very conservative** when $p$ is large. The test is valid under equality of the full feature distributions; it is not a general-purpose test with high power against every possible difference between distributions.

---

## Method 2 — Two-part (hurdle) e-values → e-BH

**Idea:** Test two distinct sources of marginal association: (i) whether a feature is *present* and (ii) its *abundance conditional on presence*. Combine valid e-values from these two components.

Let $Z_{ij}\in\{0,1\}$ indicate whether feature $j$ is structurally present in sample $i$. Whenever $Z_{ij}=1$, use the recorded, already-log-transformed value $X_{ij}$ without further transformation.

### Part A — Presence e-value

Define

$$
n_1=\sum_{i=1}^n Y_i,\qquad
s_j=\sum_{i=1}^n Z_{ij},\qquad
K_j=\sum_{i=1}^n Y_iZ_{ij}.
$$

Under $H_{0,j}$, conditioning on the class count $n_1$ and feature prevalence $s_j$ gives

$$
K_j\sim\operatorname{Hypergeometric}(n,n_1,s_j).
$$

For a fixed $\lambda>0$, define direction-specific e-values

$$
E_j^Z(t)=\frac{\exp(tK_j)}{
\mathbb{E}_{K\sim\operatorname{HG}(n,n_1,s_j)}[\exp(tK)]},
\qquad t\in\{\lambda,-\lambda\}.
$$

Use the two-sided mixture

$$
\boxed{E_j^Z=\tfrac12[E_j^Z(\lambda)+E_j^Z(-\lambda)].}
$$

The hypergeometric expectation is a finite sum over its exact support. Conditional on the margins, each direction-specific e-value has null expectation 1; the mixture is therefore also valid.

### Part B — Abundance e-value among present samples

For the $s_j$ samples with $Z_{ij}=1$, let $k_1$ and $k_0$ be the numbers with outcomes 1 and 0. When both are positive, define the observed mean difference

$$
T_j^A=
\frac{1}{k_1}\sum_{i:Y_i=1,\,Z_{ij}=1}X_{ij}
-\frac{1}{k_0}\sum_{i:Y_i=0,\,Z_{ij}=1}X_{ij}.
$$

For a fixed $\eta>0$, form the permutation-calibrated e-value

$$
E_j^A(t)=\frac{\exp(tT_j^A)}{
\mathbb{E}_{\pi}[\exp(tT_{j,\pi}^A)]},
\qquad t\in\{\eta,-\eta\},
$$

where $\pi$ uniformly reallocates the **observed present abundances** to the $k_1$ and $k_0$ label groups, preserving group sizes. Define

$$
\boxed{E_j^A=\tfrac12[E_j^A(\eta)+E_j^A(-\eta)].}
$$

Under the null, abundance values among present samples are exchangeable with respect to the labels. Thus $E_j^A$ has **conditional null expectation 1**, given the labels, presence indicators, and unordered present abundances. No distributional model for abundance is required.

If $k_0=0$ or $k_1=0$, set $E_j^A=1$: there is no within-present-sample class comparison to make.

**Exact denominator (optional computational shortcut).** Write $a_1,\ldots,a_s$ for the $s=s_j$ present abundances and let $C$ be the observed indices among them having $Y=1$. Set $\theta=t(1/k_1+1/k_0)$. Then

$$
E_j^A(t)=
\frac{\binom{s}{k_1}\exp\left(\theta\sum_{i\in C}a_i\right)}{
\displaystyle\sum_{\substack{S\subseteq\{1,\ldots,s\}\\|S|=k_1}}
\exp\left(\theta\sum_{i\in S}a_i\right)}.
$$

The denominator is an elementary symmetric polynomial and can be evaluated by dynamic programming, avoiding enumeration of all permutations. Use numerically stable computation.

### Combine the two components

The primary construction is the **product**:

$$
\boxed{E_j=E_j^Z E_j^A.}
$$

Although arbitrary e-values cannot generally be multiplied, **this particular product is valid** under $X_j\perp Y$ because the abundance component has conditional expectation 1 given $(Y,Z_j)$ (and the unordered present abundances). In particular,

$$
\mathbb{E}_{H_{0,j}}[E_j^Z E_j^A]
=\mathbb{E}_{H_{0,j}}\!\left[E_j^Z\,
\mathbb{E}_{H_{0,j}}[E_j^A\mid Y,Z_j]\right]
=\mathbb{E}_{H_{0,j}}[E_j^Z]\leq1.
$$

An alternative, always-valid combination under the joint null, is the **arithmetic mean**:

$$
E_j^{\mathrm{mean}}=\tfrac12(E_j^Z+E_j^A).
$$

Finally, apply **e-BH** to the combined e-values across all $p$ features.

---

## Comparison and scope

| Property | Method 1: Mann–Whitney → e-BH | Method 2: Two-part → e-BH |
|---|---|---|
| Feature information | Whole transformed feature, zeros included | Presence and observed abundance separately |
| Main statistic | Rank ordering | Presence count + difference in mean present abundance |
| Distributional model | None | None |
| Calibration | p-to-e from valid p-values | Exact hypergeometric and permutation normalizers |
| Cross-feature dependence | Arbitrary | Arbitrary |
| Main concern | Conservative p-to-e conversion; many ties | Power depends on $\lambda,\eta$ and support size |

**Statistical scope:** Both methods test *marginal* independence $X_j\perp Y$, not unique predictive contribution or conditional independence $Y\perp X_j\mid X_{-j}$. Finite-sample FDR control follows when the individual e-values are valid; it does **not** follow merely from naming a statistic an e-value. Strong dependence between features is permitted, but dependence **between samples** or outcome-aware tuning may invalidate the constructions.
