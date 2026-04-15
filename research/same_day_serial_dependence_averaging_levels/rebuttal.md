# Rebuttal To 5 Reviewer Agents

Date: 2026-04-15

Reviewer agents:

- Hume
- Zeno
- Bacon
- Maxwell
- Kierkegaard

## Overall Response

The five reviewers were highly consistent. The main issue was not that the core script computed the wrong quantities. The main issue was that the earlier draft of the write-up interpreted the estimands too aggressively, especially by:

1. treating stored-order adjacency too much like verified chronological adjacency
2. treating `equal_user_day_mean.gap` too much like a residual day-level gap on a common set of `user-day` units
3. not making the support mismatch in `equal_user_day_mean` sufficiently prominent

Those criticisms were valid. The script outputs and the report were revised accordingly.

## Changes Accepted And Implemented

### 1. The sequence definition was explicitly downgraded to stored row order

Reviewer concern:

- The current dataset snapshot does not expose a detected review-order column, so “adjacent”, “previous”, and “next” should not be interpreted unconditionally as true wall-clock review order.

Response:

- [final_report.md](./final_report.md) now states this limitation near the beginning.
- [results/conditional_probability_levels.json](./results/conditional_probability_levels.json) continues to expose `review_order_column: null` in the metadata.

Current position:

- The result is a stored-order descriptive result.
- To upgrade it into a claim about true same-day temporal serial dependence, one must first validate that parquet row order matches actual review chronology.

### 2. `common_support_gap` was restored as a first-class output

Reviewer concern:

- `equal_user_day_mean.gap` is a separate-support gap:
  - `mean(P(success | prev success)) - mean(P(success | prev fail))`
- It is not the average unit-level gap over the same set of `user-day` units.

Response:

- [conditional_probability_levels.py](./conditional_probability_levels.py) now explicitly outputs:
  - `common_support_gap`
  - `both_defined_units`
- These diagnostics are now included in [results/conditional_probability_levels.json](./results/conditional_probability_levels.json).
- They are also reported directly in [final_report.md](./final_report.md).

Updated key numbers for the main target `same_day_first_of_day_review`:

- `equal_user_day_mean.gap = 0.058063`
- `equal_user_day_mean.common_support_gap = 0.011314`
- `both_defined_units = 2,715,178`

Interpretation:

- `0.058063` should be interpreted only as a separate-support gap after reweighting to the `user-day` level.
- The quantity that is closer to a residual same-support day-level gap is `0.011314`.

### 3. The sequence definitions were aligned with the actual implementation

Reviewer concern:

- `raw_long_term` is not “adjacent pairs in the full user sequence”; it is adjacency inside the `elapsed_days > 0` subsequence.
- `same_day_first_of_day_review` is also formed after filtering, and adjacency is taken inside the `state == Review` subsequence.

Response:

- [final_report.md](./final_report.md) and [results/conditional_probability_levels.json](./results/conditional_probability_levels.json) were updated so that the written sequence definitions now match the code exactly.
- The definitions now explicitly include:
  - `rating in {1, 2, 3, 4}`
  - `0 < duration < 1,200,000`
  - adjacency inside the `elapsed_days > 0` subsequence
  - adjacency inside the `state == Review` subsequence

### 4. The causal and decomposition language was tightened

Reviewer concern:

- The earlier draft described the `pooled -> equal_user_mean -> equal_user_day_mean` changes too much like a formal heterogeneity decomposition or like the remaining gap after fully controlling day mixing.

Response:

- The strongest claims in [final_report.md](./final_report.md) were rewritten more conservatively:
  - these are comparisons across different weighting/support estimands
  - they show that the result is highly sensitive to the averaging level
  - they support the view that day-level composition / heterogeneity matters a great deal
  - they do not imply that a day-level causal effect has been identified

## Reasonable Requests Not Added In This Round

The reviewers also asked for several robustness extensions that were reasonable, but were not added in this round:

1. minimum-denominator threshold sensitivity  
   for example, requiring `prev success` and `prev fail` denominators to be at least 5 or 10
2. bootstrap intervals or other uncertainty quantification
3. independent validation of whether parquet row order matches true review chronology
4. sensitivity to alternative adjacency definitions  
   for example, adjacency in the full first-of-day stream rather than only the `state == Review` subsequence

These items were not presented as if they had already been completed. They remain clearly identified as next-step work.

## Position On Each Reviewer

### Hume

Accepted:

- `equal_user_day_mean.gap` should not be read as a typical-user-day residual gap
- differences between estimands should not be overinterpreted as a strict decomposition

### Zeno

Accepted:

- the common-support diagnostics needed to be restored to the formal output
- `equal_user_mean` also has support mismatch, although the numerical impact is small

### Bacon

Accepted:

- the sequence definitions needed to match the code exactly
- the stored-order caveat needed to appear near the front of the write-up

### Maxwell

Accepted:

- omitting `both_defined_units` made the day-level gap hard to interpret
- `0.058063` and `0.011314` needed to be presented side by side

### Kierkegaard

Accepted:

- the earlier draft was not yet publication-ready in its claim calibration
- the current version is better framed as a descriptive correction result, not as a final causal account of day-level mechanisms

## Final Position

After the reviewer round and rebuttal, the current version is strong enough to support the following relatively strict and publishable conclusion:

- The older pooled and weighted summaries of `P(success | prev success)` and `P(success | prev fail)` were materially misleading.
- When the averaging level is changed from `pooled` to `equal_user_mean` to `equal_user_day_mean`, the reported results change substantially.
- On the main target, once the analysis is restricted to common-support `user-day` units, the remaining day-level gap is only `0.011314`.

Accordingly, this version is suitable as a descriptive correction note that reinterprets the earlier conditional-probability gap.

However, it is not yet sufficient to support stronger claims such as:

- the specific mechanism behind day-level heterogeneity has been identified
- or true chronological direct serial dependence has been shown to be weak

Those stronger claims would require a further round of work on order validation, denominator-threshold sensitivity, and uncertainty quantification.
