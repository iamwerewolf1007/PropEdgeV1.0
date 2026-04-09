"""
PropEdge V1.0 — feature_engineering.py
========================================
Computes the 20 V1.0 new features from the existing ev dict and play dict.

Called from:
  - batch_predict.py  : score_elite(ev, play) -> build_v1_features(ev, play)
  - model_trainer.py  : build_training_matrix() -> build_v1_features(ev, play)

The 20 features here are the top-performing new signals identified in the
PropEdge ML research study (33,577 graded plays, holdout Feb-Apr 2026).
They lifted AUC from ~0.61 to 0.800 and Tier A HR from 92% to 96.9%.

Input:
  ev   — dict built by build_ev() in batch_predict.py (82 ELITE_FEATURES)
  play — raw play dict from season JSON or today.json (optional extras)

All values default to safe neutrals if the source field is missing.
No exceptions are raised — always returns a complete 20-feature dict.
"""
from __future__ import annotations
import math


def _f(v, default: float = 0.0) -> float:
    """Safe float with nan/inf guard."""
    try:
        x = float(v) if v is not None else default
        return default if (math.isnan(x) or math.isinf(x)) else x
    except (TypeError, ValueError):
        return default


def _american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    elif odds > 0:
        return 100.0 / (odds + 100.0)
    return 0.5  # zero odds treated as even


def build_v1_features(ev: dict, play: dict) -> dict:
    """
    Compute all 20 V1.0 new features.

    Parameters
    ----------
    ev   : dict  — output of build_ev() — contains the 82 ELITE_FEATURES
    play : dict  — raw play dict from today.json / season JSON
                   used for odds, usage, min_line, max_line, h2h_conf, etc.

    Returns
    -------
    dict with keys matching V1_NEW_FEATURES in config.py
    """

    # ── Pull values from ev (always available) ──────────────────────────────
    L30     = _f(ev.get("L30",   ev.get("l30",   0)))
    L10     = _f(ev.get("L10",   ev.get("l10",   L30)))
    L5      = _f(ev.get("L5",    ev.get("l5",    L10)))
    L3      = _f(ev.get("L3",    ev.get("l3",    L5)))
    std10   = _f(ev.get("std10"), 5.0)
    hr10    = _f(ev.get("hr10"), 0.5)
    hr30    = _f(ev.get("hr30"), 0.5)
    line    = _f(ev.get("line"), 15.0)
    line_in_q  = _f(ev.get("line_in_q"))          # 1 if line within Q25-Q75
    gap_mean   = _f(ev.get("gap_mean_real"))       # signed mean gap (+ = model over)
    q_range    = _f(ev.get("q_range"), 6.0)       # Q75 - Q25 width
    defP       = _f(ev.get("defP_dynamic"), 15.0) # DVP rank (1=toughest, 30=easiest)
    pace       = _f(ev.get("pace_rank"),   15.0)
    rest_days  = _f(ev.get("rest_days"),   2.0)
    min_l10    = _f(ev.get("minL10", ev.get("min_l10", 28.0)))
    prob_v12   = _f(ev.get("prob_v12"), 0.5)
    v12_conv   = _f(ev.get("v12_clf_conv"))        # |prob_v12 - 0.5| * 2
    h2h_avg_gap = _f(ev.get("h2h_avg_gap"))       # H2H average - line
    dir_v92    = _f(ev.get("dir_v92"),  1.0)       # +1=over, -1=under
    dir_v12    = 1.0 if prob_v12 >= 0.5 else -1.0
    dir_v14    = _f(ev.get("dir_v14"), 1.0)
    season_prog = _f(ev.get("season_progress",
                             play.get("seasonProgress", 0.5)), 0.5)

    # ── Pull values from play dict (optional extras) ─────────────────────────
    over_odds   = _f(play.get("over_odds",   play.get("overOdds",  -110.0)), -110.0)
    under_odds  = _f(play.get("under_odds",  play.get("underOdds", -110.0)), -110.0)
    min_line    = _f(play.get("min_line",    play.get("minLine",   line)),   line)
    max_line    = _f(play.get("max_line",    play.get("maxLine",   line)),   line)
    usage_l10   = _f(play.get("usage_l10",  play.get("usageL10",   0.22)),  0.22)
    h2h_conf    = _f(play.get("h2hConfidence", play.get("h2h_conf",
                               ev.get("h2h_conf", 0.5))),                   0.5)

    # ── 1. Double-confirmation: line in IQR AND model has large gap ──────────
    v1_inqband_x_gap = line_in_q * gap_mean

    # ── 2. Hot player vs weak defence ────────────────────────────────────────
    momentum = L5 - L30
    v1_momentum_x_dvp = momentum * (31.0 - defP)

    # ── 3. Volatile player + line placed away from their average ─────────────
    line_vs_avg_z = (line - L10) / max(std10, 0.01)
    v1_std10_x_line_z = std10 * abs(line_vs_avg_z)

    # ── 4. High usage in fast game ───────────────────────────────────────────
    v1_usage_x_pace = usage_l10 * (31.0 - pace)

    # ── 5. Rested + hot player ───────────────────────────────────────────────
    form_hot = 1.0 if (L3 > L5 and L5 > L10) else 0.0
    v1_rest_x_form = rest_days * form_hot

    # ── 6. Recent hit rate × model conviction ────────────────────────────────
    v1_hr10_x_conv = hr10 * v12_conv

    # ── 7. Quantile width relative to line ───────────────────────────────────
    v1_q_width_pct = q_range / max(line, 0.01)

    # ── 8. Prediction band tightness (complement) ────────────────────────────
    v1_q_coverage = max(0.0, 1.0 - q_range / 20.0)

    # ── 9. Model vs market disagreement (edge signal) ────────────────────────
    implied_over  = _american_to_prob(over_odds)
    v1_model_vs_market = prob_v12 - implied_over

    # ── 10. Vig (juice) indicator ─────────────────────────────────────────────
    implied_under = _american_to_prob(under_odds)
    v1_juice_total = implied_over + implied_under

    # ── 11. Bookmaker agreement (line sharpness) ─────────────────────────────
    v1_line_sharpness = min_line / max(max_line, 0.01)

    # ── 12-13. Form state ─────────────────────────────────────────────────────
    v1_form_hot  = form_hot
    v1_form_cold = 1.0 if (L3 < L5 and L5 < L10) else 0.0

    # ── 14. Recency-weighted hit rate ─────────────────────────────────────────
    v1_recency_hr = (3.0 * hr10 + hr30) / 4.0

    # ── 15. Starter indicator (role security) ────────────────────────────────
    v1_is_starter = 1.0 if min_l10 >= 28.0 else 0.0

    # ── 16. Gap magnitude ────────────────────────────────────────────────────
    v1_gap_abs = abs(gap_mean)

    # ── 17. Gap squared (non-linear response) ────────────────────────────────
    v1_pred_gap_sq = gap_mean ** 2

    # ── 18. All sub-model gaps in same direction ──────────────────────────────
    v1_all_gaps_same_dir = 1.0 if (
        dir_v92 == dir_v12 == dir_v14
    ) else 0.0

    # ── 19. Non-linear season effect ─────────────────────────────────────────
    v1_season_sq = season_prog ** 2

    # ── 20. Confidence-weighted H2H signal ───────────────────────────────────
    v1_h2h_weighted = h2h_avg_gap * h2h_conf

    return {
        "v1_inqband_x_gap":    v1_inqband_x_gap,
        "v1_momentum_x_dvp":   v1_momentum_x_dvp,
        "v1_std10_x_line_z":   v1_std10_x_line_z,
        "v1_usage_x_pace":     v1_usage_x_pace,
        "v1_rest_x_form":      v1_rest_x_form,
        "v1_hr10_x_conv":      v1_hr10_x_conv,
        "v1_q_width_pct":      v1_q_width_pct,
        "v1_q_coverage":       v1_q_coverage,
        "v1_model_vs_market":  v1_model_vs_market,
        "v1_juice_total":      v1_juice_total,
        "v1_line_sharpness":   v1_line_sharpness,
        "v1_form_hot":         v1_form_hot,
        "v1_form_cold":        v1_form_cold,
        "v1_recency_hr":       v1_recency_hr,
        "v1_is_starter":       v1_is_starter,
        "v1_gap_abs":          v1_gap_abs,
        "v1_pred_gap_sq":      v1_pred_gap_sq,
        "v1_all_gaps_same_dir":v1_all_gaps_same_dir,
        "v1_season_sq":        v1_season_sq,
        "v1_h2h_weighted":     v1_h2h_weighted,
    }
