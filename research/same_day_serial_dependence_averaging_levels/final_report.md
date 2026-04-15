# Comparing `P(success | prev success)` and `P(success | prev fail)` Across Averaging Levels

Date: 2026-04-15

Analysis script: [conditional_probability_levels.py](./conditional_probability_levels.py)  
Result file: [results/conditional_probability_levels.json](./results/conditional_probability_levels.json)

## 1. Research Question

This project studies why, within a same-day sequence of first review outcomes from different cards, one often observes:

- `P(success | prev success) > P(success | prev fail)`

If interpreted too literally, this pattern could be taken as evidence that the outcome of one card directly affects the next card. That would be in tension with the conditional-independence assumptions commonly used by FSRS-style models.

The main question here is whether the observed gap primarily reflects:

- direct same-day cross-card serial dependence

or instead:

- mixture over heterogeneous users and heterogeneous user-days

An important limitation should be stated up front:

- this analysis uses stored parquet row order
- the current dataset snapshot does not expose a detected explicit review-order column
- therefore the results are about stored-order adjacency, not independently validated wall-clock review order

## 2. Statistics Reported

This analysis reports only three averaging levels:

1. `pooled`  
   All qualifying adjacent pairs are pooled together.
2. `equal_user_mean`  
   The conditional probability is computed separately for each user, then averaged equally across users.
3. `equal_user_day_mean`  
   The conditional probability is computed separately for each retained `user-day`, then averaged equally across `user-day` units.

These three levels answer different descriptive questions:

- `pooled`: what is seen when every qualifying pair receives equal weight
- `equal_user_mean`: what changes when large users no longer dominate by pair count
- `equal_user_day_mean`: what changes when weighting is pushed down to the `user-day` level

Two cautions matter:

- `equal_user_day_mean` is equal-weight over retained `user-day` units, not nested equal-weight over users and then days
- for the mean-based levels, the `prev success` and `prev fail` conditional means are computed over the units where each conditional is defined, so the reported `gap` is generally a separate-support gap

To address that second issue, the analysis also reports:

- `common_support_gap`

which is the mean unit-level gap computed only on units where both conditionals are defined.

In addition, two restricted-subset re-aggregations are reported:

- `pooled_on_common_support_user_days`
- `equal_user_mean_on_common_support_user_days`

These first restrict the data to `user-day` units where both contexts appear, and then recompute pooled or equal-user summaries on that restricted subset.

## 3. Sequence Definitions

### Main Target: `same_day_first_of_day_review`

This is the main target of interest.

- Keep only rows with `rating in {1, 2, 3, 4}`.
- Keep only rows with `0 < duration < 1,200,000`.
- For each `user-day-card`, keep the first stored-order occurrence of that card within the day.
- Restrict to `state == Review`.
- Inside that `state == Review` subsequence, keep same-day adjacent pairs only.

Under this definition, the two events in a retained pair come from two different cards.

### Reference Target: `raw_long_term`

This is a reference sequence used for comparison.

- Keep only rows with `rating in {1, 2, 3, 4}`.
- Keep only rows with `0 < duration < 1,200,000`.
- Restrict to the subsequence with `elapsed_days > 0`.
- Form adjacent pairs inside that subsequence.
- When no explicit order column is available, adjacency follows stored parquet row order.

This is not the main scientific target, but it is useful for checking whether the qualitative pattern is specific to the main target.

## 4. Main Results

### 4.1 Main Target: `same_day_first_of_day_review`

| Level | `P(success | prev success)` | `P(success | prev fail)` | Gap |
| --- | ---: | ---: | ---: |
| `pooled` | `0.898358` | `0.680715` | `0.217643` |
| `equal_user_mean` | `0.884342` | `0.704360` | `0.179981` |
| `equal_user_day_mean` | `0.876780` | `0.818717` | `0.058063` |

The `Gap` column above is a separate-support gap:

- `mean(P(success | prev success)) - mean(P(success | prev fail))`

where the two means may be taken over different unit sets.

Common-support diagnostics:

- `equal_user_mean`: `common_support_gap = 0.179427`, with `9,939` users on common support
- `equal_user_day_mean`: `common_support_gap = 0.011314`, with `2,715,178` `user-day` units on common support

Re-aggregating only on common-support `user-day` units:

- `pooled_on_common_support_user_days`: `P(success | prev success) = 0.882378`, `P(success | prev fail) = 0.685349`, gap `= 0.197029`
- `equal_user_mean_on_common_support_user_days`: `P(success | prev success) = 0.865950`, `P(success | prev fail) = 0.712337`, gap `= 0.153613`

Separate-support gap shrinkage:

- `pooled -> equal_user_mean`: `0.037662`
- `equal_user_mean -> equal_user_day_mean`: `0.121918`

Support:

- qualifying pairs: `375,615,926`
- users with at least one retained pair: `9,987`
- `user-day` units with at least one retained pair: `3,722,477`
- within `equal_user_day_mean`:
  - `prev success` defined on `3,691,520` `user-day` units
  - `prev fail` defined on `2,746,135` `user-day` units
  - both conditionals defined on `2,715,178` `user-day` units

### 4.2 Reference Target: `raw_long_term`

| Level | `P(success | prev success)` | `P(success | prev fail)` | Gap |
| --- | ---: | ---: | ---: |
| `pooled` | `0.895908` | `0.651259` | `0.244649` |
| `equal_user_mean` | `0.881320` | `0.674056` | `0.207264` |
| `equal_user_day_mean` | `0.864873` | `0.789761` | `0.075112` |

Common-support diagnostics:

- `equal_user_mean`: `common_support_gap = 0.206955`, with `9,974` users on common support
- `equal_user_day_mean`: `common_support_gap = 0.029955`, with `3,067,538` `user-day` units on common support

Re-aggregating only on common-support `user-day` units:

- `pooled_on_common_support_user_days`: `P(success | prev success) = 0.880828`, `P(success | prev fail) = 0.654464`, gap `= 0.226364`
- `equal_user_mean_on_common_support_user_days`: `P(success | prev success) = 0.866073`, `P(success | prev fail) = 0.677363`, gap `= 0.188710`

Separate-support gap shrinkage:

- `pooled -> equal_user_mean`: `0.037385`
- `equal_user_mean -> equal_user_day_mean`: `0.132152`

The same qualitative pattern appears in the reference target as well.

## 5. Interpretation

The main empirical pattern is:

1. Moving from `pooled` to `equal_user_mean` reduces the gap somewhat.
2. Moving from `equal_user_mean` to `equal_user_day_mean` reduces it much more.
3. Restricting the sample to common-support `user-day` units does reduce the gap, but does not make it small by itself.
4. On the main target, even after that restriction, the gap is still `0.197029` under pooled re-aggregation and `0.153613` under equal-user re-aggregation.
5. The gap becomes very small only at the equal-user-day common-support level: `0.011314`.

This supports the following interpretation:

- cross-user heterogeneity does matter
- restricting to common-support days alone does not remove most of the apparent effect
- day-level composition / heterogeneity matters more, especially once weighting is pushed all the way down to the `user-day` level
- a large pooled conditional-probability gap does not by itself imply strong direct same-day card-to-card dependence

The claim should be calibrated carefully. The current evidence supports:

- day-level composition / heterogeneity is a major contributor to the observed gap

It does not support the stronger claim that:

- all direct serial dependence has been ruled out

Intuitively, if some `user-day` units are high-success days and others are low-success days, then:

- high-success days are more likely to generate `prev success`
- low-success days are more likely to generate `prev fail`
- subsequent events on those same days inherit those different day-level success baselines

Mixing those day types together can mechanically create a large pooled gap even without strong direct causal influence from one card outcome to the next.

## 6. Why the Older Pooled/Weighted Summaries Were Misleading

The earlier descriptive summaries were misleading for two reasons:

1. `pooled` gives more weight to large users, high-activity days, and high-frequency contexts.
2. review-count-weighted user summaries still do not answer the question at the level of a typical user or a typical retained `user-day`.

As a result, those summaries mainly answer:

- what happens after mixing all eligible observations together

rather than:

- what happens for equal-weight users
- what happens for equal-weight retained `user-day` units

For this problem, the mixed answer is exactly the one most vulnerable to heterogeneity bias.

## 7. Most Defensible Conclusion

For the main target `same_day_first_of_day_review`:

- the pooled gap is `0.217643`
- the equal-user gap is `0.179981`
- on the subset of common-support `user-day` units, the pooled gap is still `0.197029`
- on the subset of common-support `user-day` units, the equal-user gap is still `0.153613`
- the equal-user-day separate-support gap is `0.058063`
- the equal-user-day common-support gap is only `0.011314`

The most defensible reading of the current data is therefore:

1. The large observed gap is not explained by cross-user differences alone.
2. It is also not strong evidence, by itself, of strong direct same-day card-to-card dependence.
3. Restricting the sample to common-support `user-day` units reduces the gap only modestly if one continues to use pooled or equal-user aggregation.
4. Reweighting at the `user-day` level is what greatly reduces the gap.
5. Restricting to common-support `user-day` units at the `user-day` level reduces the residual gap even further.

So the current evidence is much more consistent with:

- day-level heterogeneity / composition as the main source of the apparent effect

than with:

- a large, pervasive, direct same-day causal effect from one card outcome to the next

## 8. Limitations

This report identifies where the gap shrinks when the averaging level changes. It does not identify the specific mechanism behind day-level heterogeneity.

With the current revlog data alone, this analysis cannot separately identify the contributions of:

- fatigue or attention fluctuation
- changes in deck mix or difficulty mix
- workload and pacing differences
- time-of-day or external-context effects
- latent day-level memory-state variation

There are also important structural limitations:

- `equal_user_mean` also has support mismatch, although the numerical difference is small on the main target: `0.179981 -> 0.179427`
- `equal_user_day_mean` has substantial support mismatch
- on the main target, `prev success` is defined on `3,691,520` retained `user-day` units but `prev fail` only on `2,746,135`
- therefore the separate-support day-level gap `0.058063` is not the same estimand as the common-support day-level gap `0.011314`
- the restricted re-aggregations `pooled_on_common_support_user_days` and `equal_user_mean_on_common_support_user_days` answer an intermediate question: what happens after restricting the support, but before equal-weighting `user-day` units

Two robustness checks were not added in this round:

- minimum-denominator threshold sensitivity for equal-user and equal-user-day means
- bootstrap intervals or other uncertainty quantification

Therefore the current conclusion should be stated as:

- under the present definitions and weighting schemes, the largest shrinkage occurs when moving from user-level averaging to `user-day`-level averaging

and not as:

- the exact day-level causal effect has been identified

## 9. Conclusion

The corrected conclusion can be stated more rigorously as follows:

- The large conditional-probability gap is highly sensitive to the averaging level.
- Moving from `pooled` to `equal_user_mean` to `equal_user_day_mean` substantially reduces the separate-support gap.
- On the main target, restricting to common-support `user-day` units alone still leaves a large gap under pooled (`0.197029`) and equal-user (`0.153613`) aggregation.
- On the main target, once comparison is pushed all the way to the equal-user-day common-support level, the residual gap is only `0.011314`.

Accordingly, the current evidence supports:

- most of the apparent sequence effect is generated by user/day-level heterogeneity and composition

more than it supports:

- a strong, general, direct same-day cross-card serial dependence effect

## 10. Reproduction

```bash
uv run python research/same_day_serial_dependence_averaging_levels/conditional_probability_levels.py --max-workers 8
```
