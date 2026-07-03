"""
Data Siege defense.

One metered tool call per event (5 event types → 180 credits against a 220
budget, so cost_overage stays 0). Detection is generalizable statistical /
structural logic, not a run-specific answer table:

  * numeric two-sided signals (row_count, mean_amount) use a z-score against the
    mean/sigma reconstructed from the published mean±3σ baseline bounds, flagged
    a touch inside 3σ so subtle deviations still trip;
  * one-sided "high is bad" ceilings (null_rate, staleness, freshness,
    lineage duration, embedding centroid shift, doc age, feature mean-shift
    sigma) flag above the published ceiling;
  * structural faults (contract schema/type violations, orphaned lineage
    outputs, missing upstream edges) are exact and deterministic.

See ../README.md for the interface and ../RULES.md for what's legal.
"""
from api import Verdict

# How far inside the published 3σ bound a two-sided numeric signal may drift
# before we alert. TP is weighted ~4x an FP in the score given the clean/faulty
# class balance, so pulling in slightly from 3σ (to catch subtle-tier faults)
# is worth the small precision risk. Practice clean data stays under ~2.5σ.
Z = 2.75

# Recalibrated ceilings from the observed clean spread (the published baselines
# are calibrated to a different clean stream and don't perfectly fit every phase):
#   * clean feature mean-shift reaches ~0.47σ while real skews are all >=1.8σ, so
#     the published 0.41 cut sits inside the clean tail and false-alarms. A cut in
#     the wide empty gap catches every real skew with zero clean false positives.
#   * clean corpus age tops out ~41d, so a cut below the published 49.8 catches
#     subtle staleness (~48d) while keeping a safe margin over clean.
FEATURE_SIGMA_CUT = 0.6
CORPUS_AGE_CUT = 46.0


def _err(res):
    return not isinstance(res, dict) or "error" in res


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    res = ctx.tools.batch_profile(payload["batch_id"])
    if _err(res):
        return Verdict(alert=False, pillar="checks", reason="no profile")
    b = ctx.baseline
    reasons = []

    # two-sided: reconstruct mean/sigma from mean±3σ bounds, flag inside 3σ
    row_mid = (b["row_count_min"] + b["row_count_max"]) / 2
    row_sd = (b["row_count_max"] - b["row_count_min"]) / 6 or 1e-9
    if abs(res["row_count"] - row_mid) / row_sd > Z:
        reasons.append("volume")

    amt_mid = (b["mean_amount_min"] + b["mean_amount_max"]) / 2
    amt_sd = (b["mean_amount_max"] - b["mean_amount_min"]) / 6 or 1e-9
    if abs(res["mean_amount"] - amt_mid) / amt_sd > Z:
        reasons.append("distribution")

    # one-sided ceilings
    if res["null_rate"].get("customer_id", 0.0) > b["null_rate_max"]:
        reasons.append("null_rate")
    if res["staleness_min"] > b["staleness_min_max"]:
        reasons.append("staleness")

    return Verdict(alert=bool(reasons), pillar="checks", reason=",".join(reasons))


def check_contract_checkpoint(payload, ctx):
    res = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if _err(res):
        return Verdict(alert=False, pillar="contracts", reason="no diff")
    reasons = list(res.get("violations", []))  # schema_hash_mismatch / type_violation
    if res.get("freshness_delay_min", 0.0) > ctx.baseline["freshness_delay_max_min"]:
        reasons.append("freshness_sla")
    return Verdict(alert=bool(reasons), pillar="contracts", reason=",".join(reasons))


def check_lineage_run(payload, ctx):
    res = ctx.tools.lineage_graph_slice(payload["run_id"])
    if _err(res):
        return Verdict(alert=False, pillar="lineage", reason="no slice")
    reasons = []

    if res.get("actual_downstream_count", 1) == 0:
        reasons.append("orphan_output")
    if res.get("duration_ms", 0.0) > ctx.baseline["lineage_duration_ms_max"]:
        reasons.append("runtime_anomaly")

    # missing_upstream: learn each job's upstream edge set across the stream and
    # flag a run missing an edge previously seen for that same job.
    job = payload.get("job", "?")
    cur = set(res.get("actual_upstream") or [])
    ref_by_job = ctx.state.setdefault("lineage_up_ref", {})
    ref = ref_by_job.get(job, set())
    if ref - cur:
        reasons.append("missing_upstream")
    ref_by_job[job] = ref | cur

    return Verdict(alert=bool(reasons), pillar="lineage", reason=",".join(reasons))


def check_feature_materialization(payload, ctx):
    res = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if _err(res):
        return Verdict(alert=False, pillar="ai_infra", reason="no drift")
    skew = res.get("mean_shift_sigma", 0.0) > FEATURE_SIGMA_CUT
    return Verdict(alert=bool(skew), pillar="ai_infra", reason="feature_skew" if skew else "")


def check_embedding_batch(payload, ctx):
    res = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if _err(res):
        return Verdict(alert=False, pillar="ai_infra", reason="no drift")
    b = ctx.baseline
    reasons = []
    if res.get("centroid_shift", 0.0) > b["embedding_centroid_shift_max"]:
        reasons.append("embedding_drift")
    if res.get("avg_doc_age_days", 0.0) > CORPUS_AGE_CUT:
        reasons.append("corpus_staleness")
    return Verdict(alert=bool(reasons), pillar="ai_infra", reason=",".join(reasons))
