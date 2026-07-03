# Reflection

**Strategy.** One metered tool call per event (5 event types → 180 credits vs. a
220 budget, so cost_overage stays 0 on a practice-length stream). Each handler
turns the tool result into a verdict with generalizable logic, never a
run-specific answer table:

- **Structural faults** (contract schema-hash mismatch / type violation,
  orphaned lineage output with `downstream_count == 0`, missing upstream edges)
  are exact and deterministic — caught with zero false positives.
- **Missing upstream** is learned online: I track each job's observed upstream
  edge set in `ctx.state` and alert a run that drops an edge previously seen for
  that job, rather than hardcoding the expected set.
- **Numeric signals** use the published baselines. Two-sided signals
  (`row_count`, `mean_amount`) are flagged on a z-score against the mean/σ
  reconstructed from the mean±3σ bounds, at 2.75σ instead of the full 3σ, so
  subtle deviations still trip. One-sided ceilings (`null_rate`, `staleness`,
  `freshness`, `lineage_duration`, `centroid_shift`) flag above the ceiling.

**Which fault types were hardest to catch, and why?** The subtle-tier
`ai_infra` faults sitting right at normal variance. On the public stream an
`embedding_drift` at centroid 0.040 was indistinguishable from clean events
reaching 0.039 — no threshold separates them without paying false positives, so
I left the centroid cut at the published 3σ ceiling and accept that miss.
Feature skew was the opposite lesson: the published `feature_mean_shift_sigma_max`
(0.41) sits *below* the real clean tail (clean reaches ~0.47σ) while genuine
skews are all ≥1.8σ — so the published cut false-alarmed on clean events. I
moved the cut into the wide empty gap (0.6), which removed every false positive
and kept every real skew. Similarly I lowered the corpus-age cut from 49.8 to 46
(clean tops out ~41d) to catch subtle staleness near 48d with margin.

**What would you change about the cost/coverage tradeoff with another pass?**
The budget only bites on longer streams: at ~160 events, one-call-per-event
reaches 240 > 220 (a ~1.8-point overage penalty). I chose *not* to throttle,
because at the observed fault rate (~24%) the expected value of each additional
call (≈+0.31) still exceeds its marginal overage cost (≈+0.18) — skipping events
to save credits loses more score than it saves. If the private stream were much
longer or the fault rate much lower, I'd add budget-aware gating that spends
remaining credits on the highest-base-rate event types (feature/embedding) and
drops the cheap, low-yield ones first. Given the class imbalance, the whole
design deliberately leans toward recall: a caught fault is worth ~4× a false
alarm, so every threshold is set to the recall-favoring side of the tradeoff.
