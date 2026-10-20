# SHOE: honest scores and conditional evidence

SHOE selects marginally associated raw bacterium–gene predictors using an
honest split, conditional e-values and the native e-BH rule at alpha=.10.

## Frozen score learning

A label-independent seeded half-split separates score training from evidence.
The training half defines retained pairs, constant exclusion and zero/low/high
bins from supplied values, without normalization or an abundance transformation.
Positive bins are separated by the
training positive median; collapse the split if no larger value exists.

For each bin, learn q=(class-1 count+1)/(bin size+2). Its logit is finite without
clipping. Subtract the zero-bin logit to obtain contrasts r1 and r2. No evidence
or validation labels enter these scores or the candidate family.

## Conditional evidence

For one feature, let m be the evidence sample count, k the evidence class-1
count, n0,n1,n2 its bin sizes, and u1,u2 the observed class-1 counts in the
positive bins. Calculate

\[
E=\frac{\exp(u_1r_1+u_2r_2)}{
\displaystyle\sum_{a,b}
\frac{\binom{n_1}{a}\binom{n_2}{b}\binom{n_0}{k-a-b}}{\binom{m}{k}}
\exp(ar_1+br_2)}.
\]

Sum over feasible allocations only. This is one calculation over the **entire
evidence half**, conditioning on its total class count. The implementation
enumerates bin-count allocations, not individual labelings, and uses log-space
calculations. Features absent throughout evidence receive E=0.

Under the marginal null, independent exchangeable rows make the k positive
labels uniform over the feature's evidence observations, conditional on the
training quantities, this feature's column and k. The denominator is the exact
conditional null mean of the numerator; hence E has null expectation one
(or at most one under the absent-feature convention). Cross-feature independence
is not required. This argument does not condition on the entire multivariate
evidence matrix. Invalid labels, dependence between participants, or a different
null interpretation can invalidate it; numerical tests are not proofs.

## Native selection

For the complete eligible family of d features, use uniform weights 1/d.
Order E decreasingly, breaking ties by raw pair ID. Take the largest index j
with E_(j) >= d/(alpha*j), and select the first j features; select none if no
index qualifies. This is the native [e-BH procedure](https://sas.uwaterloo.ca/~wang/papers/2021Wang-Ramdas-JRSSB.pdf),
which controls FDR under arbitrary feature dependence when its null e-values
are valid. It is not p-value BH/BY correction.

Only the **complete native set** has this statement. Arbitrary Top-K rankings
and truncations do not automatically inherit it. The target concerns association
on the supplied feature scale, not causality or conditional effects.
The honest split reduces available learning and evidence
samples, so small datasets and rare signals can have low power.

The worst-case evidence cost is O(d*m^2) time with O(m^2) allocation tables;
feature batches bound temporary score arrays. Inputs remain sparse, with no
complete dense sample-by-feature matrix.
