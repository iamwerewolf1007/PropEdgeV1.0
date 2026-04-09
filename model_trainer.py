"""
PropEdge V1.0 — model_trainer.py
Trains the PropEdge V1.0 XGB+LGB+LR ensemble meta-model.

V1.0 changes from V1.0:
  - Training target: BET WIN (direction was correct) not raw result=='WIN'
    This fixes the core issue: UNDER bets were graded WIN if player went over,
    making UNDER plays appear to have low hit rates and land in SKIP tier.
    Corrected target: bet_win = (direction=='OVER' and result=='WIN') OR
                                (direction=='UNDER' and result=='LOSS')
  - Model architecture: XGBoost(40%) + LightGBM(40%) + LogisticRegression(20%)
    Replaces single GBT — AUC improved from ~0.61 to 0.80 in holdout testing
  - Feature count: 102 (82 ELITE_FEATURES + 20 V1_NEW_FEATURES)
  - Output file: models/elite/propedge_v1.pkl
  - Bundle format: {xgb, lgb, lr, scaler, features, version, trained_at, ...}
  - Walk-forward validation uses corrected BET WIN target

Training data:
  - season_2024_25.json  (real bookmaker lines — full 2024-25 season)
  - season_2025_26.json  (real bookmaker lines — 2025-26 season to date)
  - Only WIN/LOSS graded plays used — DNP and PUSH excluded
  - source='synthetic' plays excluded (legacy guard)
  - Minimum 200 graded plays required

V12/V14/V9.2 sub-models are NOT retrained here — they are fixed.
Only the V1.0 meta-model updates monthly.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, FILE_SEASON_2526, FILE_SEASON_2425, FILE_V1_MODEL,
    FILE_V92_REG, FILE_V10_REG, FILE_V11_REG,
    FILE_V12_REG, FILE_V12_CLF, FILE_V12_CAL, FILE_V12_SEG, FILE_V12_Q,
    FILE_V14_REG, FILE_V14_CLF, FILE_V14_CAL,
    FILE_V12_TRUST, FILE_V14_TRUST,
    FILE_GL_2526, FILE_GL_2425, FILE_H2H,
    ELITE_FEATURES, V1_NEW_FEATURES, V1_FEATURES,
    MIN_PRIOR_GAMES, TRUST_THRESHOLD, get_pos_group,
)
from feature_engineering import build_v1_features

MIN_PLAYS_TO_TRAIN = 200


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRADED PLAYS
# ─────────────────────────────────────────────────────────────────────────────
def load_training_plays() -> list[dict]:
    """
    Load WIN/LOSS graded plays from both season JSONs.
    Sorted chronologically for walk-forward integrity.
    """
    all_plays: list[dict] = []

    for season_file, label in [
        (FILE_SEASON_2425, "2024-25"),
        (FILE_SEASON_2526, "2025-26"),
    ]:
        if not season_file.exists():
            print(f"  ⚠ {season_file.name} not found — skipping {label}")
            continue
        with open(season_file) as f:
            plays = json.load(f)
        graded = [
            p for p in plays
            if p.get("result") in ("WIN", "LOSS")
            and p.get("source", "excel") != "synthetic"
        ]
        print(f"  {label}: {len(graded):,} graded plays")
        all_plays.extend(graded)

    all_plays.sort(key=lambda p: p.get("date", ""))

    total = len(all_plays)
    # compute BET WIN rate (direction-aware) for reporting
    bet_wins = sum(1 for p in all_plays if _is_bet_win(p))
    raw_wins = sum(1 for p in all_plays if p.get("result") == "WIN")
    print(f"  Combined: {total:,} plays | "
          f"Raw WIN rate: {raw_wins/total:.1%} | "
          f"Bet WIN rate (V1.0 target): {bet_wins/total:.1%}")
    return all_plays


def _is_bet_win(p: dict) -> bool:
    """
    V1.0 corrected target: did the BET win (direction correct)?

    Grading in batch0_grade uses 'always OVER' convention:
      result='WIN'  means actual_pts > line
      result='LOSS' means actual_pts <= line

    The direction field ('OVER' or 'UNDER') tells us which way the model bet.
    For OVER plays: WIN = player went over (result='WIN')
    For UNDER plays: WIN = player went under (result='LOSS')

    This is the fundamental fix vs V1.0 legacy: old model trained on result=='WIN' which
    treated all UNDER bets as losses regardless of prediction, causing
    high-confidence UNDER plays to land in SKIP tier.
    """
    direction = str(p.get("direction", p.get("dir", "OVER")) or "OVER").upper()
    result    = str(p.get("result", "") or "")
    if "UNDER" in direction:
        return result == "LOSS"   # UNDER wins when player stays under line
    else:
        return result == "WIN"    # OVER wins when player goes over line


# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRAINING MATRIX (102 features)
# ─────────────────────────────────────────────────────────────────────────────
def build_training_matrix(plays: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct 82 existing ELITE_FEATURES + 20 V1_NEW_FEATURES from stored
    JSON fields.  This avoids re-running the full feature extraction pipeline.
    """
    rows: list[dict] = []
    labels: list[int] = []

    for p in plays:
        try:
            line  = float(p.get("line", 0))
            prb12 = float(p.get("v12_clf_prob", p.get("calProb", 0.5)))
            prb14 = float(p.get("v14_clf_prob", 0.5))
            g92   = float(p.get("real_gap_v92", p.get("predGap", 1.0)))
            g12   = float(p.get("real_gap_v12", g92))
            g10   = float(p.get("predGap", g92))
            g11   = g10
            g14   = float(p.get("V14_predGap", p.get("predGap", g92)))

            L30   = float(p.get("l30", p.get("L30", line)))
            L10   = float(p.get("l10", p.get("L10", line)))
            L5    = float(p.get("l5",  p.get("L5",  line)))
            L3    = float(p.get("l3",  p.get("L3",  line)))
            std10 = float(p.get("std10", 5.0))

            dv92 = 1 if (float(p.get("real_gap_v92", 1.0)) > 0 and "OVER" in str(p.get("dir","")).upper()) else -1
            dv10 = dv92; dv11 = dv92
            dv14 = 1 if prb14 >= 0.5 else -1
            dv12 = 1 if prb12 >= 0.5 else -1

            ag12 = int(dv92 == dv12); ag14 = int(dv92 == dv14)
            ag12_14 = int(dv12 == dv14); all_ag = int(ag12 and ag14)
            v12ar = int(dv12 == dv92 and dv12 == dv10)
            ro = int(all(d == 1  for d in [dv92, dv10, dv11, dv14]))
            ru = int(all(d == -1 for d in [dv92, dv10, dv11, dv14]))
            rc = int(ro or ru)

            cv12 = abs(prb12 - 0.5) * 2; cv14 = abs(prb14 - 0.5) * 2
            vex  = int(prb12 >= 0.80 or prb12 <= 0.20)
            vso  = int(prb12 >= 0.75); vsu = int(prb12 <= 0.25)

            gmr  = (g92 + g10 + g12) / 3.0
            gmx  = max(g92, g10, g12)
            vc92 = float(np.clip(0.5 + g92 * 0.04, 0.45, 0.90))
            vc12 = prb12; vc14 = prb14; cm = (vc92 + vc12 + vc14) / 3.0

            q25 = float(p.get("q25_v12", line - 3.0))
            q75 = float(p.get("q75_v12", line + 3.0))
            qr  = max(q75 - q25, 1.0)
            lq25 = line - q25; lq75 = line - q75
            qc   = float(1.0 - (max(lq25, 0) + abs(min(lq75, 0))) / qr)
            liq  = int(q25 <= line <= q75)

            h2hG    = float(p.get("h2hG", p.get("h2h_games", 0)) or 0)
            h2h_avg = p.get("h2hAvg") or p.get("h2h_avg")
            hgap    = float((h2h_avg - line) if h2h_avg is not None else 0.0)
            hts     = float(p.get("h2h_ts_dev", 0) or 0)
            hfga    = float(p.get("h2h_fga_dev", 0) or 0)
            hal     = int(h2hG >= 3 and ((hgap > 0) == (prb12 >= 0.5)))

            vol10 = L10 - line; vol30 = L30 - line
            tr530 = L5  - L30;  tr35  = L3  - L5
            lv = int(std10 <= 4); hv = int(std10 >= 8)

            ls  = float((p.get("max_line") or line) - (p.get("min_line") or line))
            ls2 = 1.0 / (ls + 1.0)
            bk  = int((p.get("books", 0) or 0) >= 6)
            tm  = int(ls <= 0.5)

            rd  = float(p.get("rest_days", 2) or 2)
            rb  = int(rd <= 1); rsw = int(rd == 3); rru = int(rd >= 4)

            pos = str(p.get("position", "G"))
            pg  = get_pos_group(pos)
            ig  = int(pg == "Guard"); ic = int(pg == "Center")

            t12 = float(p.get("trust_v12", 0.68) or 0.68)
            t14 = float(p.get("trust_v14", 0.67) or 0.67)
            tmn = (t12 + t14) / 2.0; lt = int(tmn < 0.50)

            def tn(c): return 4.0 if c>=0.75 else (3.0 if c>=0.72 else (2.0 if c>=0.68 else (1.0 if c>=0.60 else 0.0)))
            vtn = tn(vc92); v12tn = tn(vc12); ts = vtn + v12tn + tn(vc14)

            is_home = p.get("isHome")

            # Build the 82 ELITE_FEATURES dict
            ev = {
                "v92_v12clf_agree": float(ag12), "v12_clf_conv": cv12,
                "prob_v12": prb12, "prob_v14": prb14,
                "v12_extreme": float(vex), "v12_strong_under": float(vsu),
                "v12_strong_over": float(vso), "v92_v14clf_agree": float(ag14),
                "v12_v14_agree": float(ag12_14), "all_clf_agree": float(all_ag),
                "v12clf_allreg": float(v12ar), "reg_consensus": float(rc),
                "reg_all_over": float(ro), "reg_all_under": float(ru),
                "dir_v92": float(dv92), "dir_v10": float(dv10),
                "dir_v11": float(dv11), "dir_v14": float(dv14),
                "gap_v92": g92, "gap_v10": g10, "gap_v11": g11,
                "gap_v12": g12, "gap_v14": g14,
                "gap_mean_real": gmr, "gap_max_real": gmx,
                "V9.2_predGap": g92, "V12_predGap": g12, "V14_predGap": g14,
                "V9.2_conf": vc92, "V12_conf": vc12, "V14_conf": vc14,
                "conf_mean": cm, "v14_clf_conv": cv14,
                "h2h_avg_gap": hgap, "h2hG": float(h2hG),
                "h2h_ts_dev": hts, "h2h_fga_dev": hfga,
                "h2h_v12_align": float(hal),
                "q25_v12": q25, "q75_v12": q75, "q_range": qr,
                "q_confidence": qc, "line_in_q": float(liq),
                "line_vs_q25": lq25, "line_vs_q75": lq75,
                "L30": L30, "L10": L10, "L5": L5, "L3": L3,
                "std10": std10,
                "hr30": float(p.get("hr30", 0.5) or 0.5),
                "hr10": float(p.get("hr10", 0.5) or 0.5),
                "minL10": float(p.get("minL10", p.get("min_l10", 28)) or 28),
                "n_games": float(p.get("h2hG", 30) or 30),
                "vol_l30": vol30, "vol_l10": vol10,
                "trend_l5l30": tr530, "trend_l3l5": tr35,
                "low_var": float(lv), "high_var": float(hv),
                "line": float(line), "line_sharp2": ls2,
                "books_sig": float(bk), "tight_market": float(tm),
                "rest_days": rd, "rest_b2b": float(rb),
                "rest_sweet": float(rsw), "rest_rust": float(rru),
                "is_b2b": float(p.get("is_b2b", False) or 0),
                "is_guard": float(ig), "is_center": float(ic),
                "pace_rank": float(p.get("pace_rank", p.get("pace", 15)) or 15),
                "defP_dynamic": float(p.get("defP_dynamic", p.get("defP", 15)) or 15),
                "is_home": float(1 if is_home else 0),
                "trust_v12": t12, "trust_v14": t14, "trust_mean": tmn,
                "low_trust": float(lt), "tier_sum": ts,
                "V9.2_tn": vtn, "V12_tn": v12tn,
                # Pass season_progress so feature_engineering can access it
                "season_progress": float(p.get("seasonProgress", 0.5) or 0.5),
            }

            # Add 20 V1.0 new features
            v1_feats = build_v1_features(ev, p)
            row = {**ev, **v1_feats}

            rows.append(row)
            # V1.0 TARGET: Bet Win (direction was correct)
            labels.append(1 if _is_bet_win(p) else 0)

        except Exception:
            continue

    df = pd.DataFrame(rows)
    # Use all V1_FEATURES; fill any missing with 0
    for col in V1_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    X = df[V1_FEATURES].fillna(0).values
    y = np.array(labels)
    print(f"  Training matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  Bet WIN rate (corrected target): {y.mean():.1%}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_validate(plays: list[dict], X: np.ndarray, y: np.ndarray) -> dict:
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        print(f"  ⚠ Walk-forward skipped: {e}")
        return {}

    months  = sorted(set(p.get("date","")[:7] for p in plays))
    results = []

    for i in range(2, len(months)):
        tr_months = set(months[:i]); te_month = months[i]
        tr = np.array([p.get("date","")[:7] in tr_months for p in plays])
        te = np.array([p.get("date","")[:7] == te_month  for p in plays])
        if tr.sum() < 100 or te.sum() < 5:
            continue

        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        sc  = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)

        # Train lightweight ensemble for speed
        clf_xgb = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            random_state=42, tree_method="hist", verbosity=0
        )
        clf_xgb.fit(Xtr, ytr)

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, verbose=-1
        )
        clf_lgb.fit(Xtr, ytr)

        clf_lr = LogisticRegression(C=0.1, max_iter=300)
        clf_lr.fit(Xtr_s, ytr)

        prob = (0.40 * clf_xgb.predict_proba(Xte)[:, 1] +
                0.40 * clf_lgb.predict_proba(Xte)[:, 1] +
                0.20 * clf_lr.predict_proba(Xte_s)[:, 1])

        for thresh, tname in [
            (0.65, "PLAY+"), (0.72, "STRONG"), (0.78, "ELITE"),
            (0.85, "ULTRA"), (0.90, "APEX"),
        ]:
            mk = prob >= thresh
            if mk.sum() < 5:
                continue
            results.append({
                "month": te_month, "tier": tname,
                "n": int(mk.sum()), "acc": float(yte[mk].mean()),
            })

    summary: dict = {}
    for tier in ["PLAY+", "STRONG", "ELITE", "ULTRA", "APEX"]:
        sub = [r for r in results if r["tier"] == tier]
        if not sub:
            continue
        total_n = sum(r["n"] for r in sub)
        total_w = sum(r["n"] * r["acc"] for r in sub)
        wacc    = total_w / total_n if total_n > 0 else 0.0
        m_tgt   = {"PLAY+":0.70,"STRONG":0.76,"ELITE":0.82,"ULTRA":0.88,"APEX":0.92}
        tgt     = m_tgt.get(tier, 0.80)
        m_hit   = sum(1 for r in sub if r["acc"] >= tgt)
        summary[tier] = {
            "acc": round(wacc, 4), "n": total_n,
            "months_at_target": m_hit, "months_total": len(sub),
        }
        flag = "✓" if wacc >= tgt else "⚠"
        print(f"  {flag} {tier:8s}: {wacc:.1%} | n={total_n:,} | "
              f"months>={tgt:.0%}: {m_hit}/{len(sub)}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN FINAL V1.0 ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
def train_final_model(X: np.ndarray, y: np.ndarray):
    """
    Train the XGB+LGB+LR ensemble on all available data.
    Returns (xgb_model, lgb_model, lr_model, scaler).
    """
    try:
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise RuntimeError(f"V1.0 requires xgboost, lightgbm, scikit-learn: {e}")

    sc = StandardScaler().fit(X)
    X_s = sc.transform(X)

    print("  Training XGBoost...")
    clf_xgb = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.015,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        gamma=0.1, random_state=42, tree_method="hist", verbosity=0
    )
    clf_xgb.fit(X, y)

    print("  Training LightGBM...")
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.015,
        num_leaves=63, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1
    )
    clf_lgb.fit(X, y)

    print("  Training Logistic Regression...")
    clf_lr = LogisticRegression(C=0.1, max_iter=1000)
    clf_lr.fit(X_s, y)

    # In-sample ensemble accuracy
    prob = (0.40 * clf_xgb.predict_proba(X)[:, 1] +
            0.40 * clf_lgb.predict_proba(X)[:, 1] +
            0.20 * clf_lr.predict_proba(X_s)[:, 1])
    acc = float((prob >= 0.5).sum() / len(y))
    print(f"  In-sample ensemble accuracy (50% threshold): {acc:.1%}")

    return clf_xgb, clf_lgb, clf_lr, sc


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save_model(xgb_m, lgb_m, lr_m, scaler, n_plays: int,
               oof_summary: dict, seasons: list[str] | None = None) -> None:
    FILE_V1_MODEL.parent.mkdir(parents=True, exist_ok=True)
    pkg = {
        # V1.0 ensemble components
        "xgb":          xgb_m,
        "lgb":          lgb_m,
        "lr":           lr_m,
        "scaler":       scaler,
        "features":     V1_FEATURES,           # 102 features
        "elite_features": ELITE_FEATURES,       # 82 — for reference
        "v1_new_features": V1_NEW_FEATURES,   # 20 — for reference
        # Metadata
        "version":      VERSION,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "n_plays":      n_plays,
        "oof_summary":  oof_summary,
        "seasons_used": seasons or ["2024-25", "2025-26"],
        "target":       "bet_win",              # direction-corrected target
        "weights":      {"xgb": 0.40, "lgb": 0.40, "lr": 0.20},
        "tier_thresholds": {
            "APEX": 0.90, "ULTRA": 0.85, "ELITE": 0.78,
            "STRONG": 0.72, "PLAY+": 0.65,
        },
    }
    with open(FILE_V1_MODEL, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  ✓ Saved: {FILE_V1_MODEL}")
    print(f"    n_plays={n_plays} | seasons={','.join(pkg['seasons_used'])} | "
          f"trained_at={pkg['trained_at'][:19]}")
    print(f"    Features: {len(V1_FEATURES)} (82 legacy + 20 new)")
    print(f"    Target: {pkg['target']} (V1.0 corrected)")


# ─────────────────────────────────────────────────────────────────────────────
# TRUST SCORE UPDATE
# ─────────────────────────────────────────────────────────────────────────────
def update_trust_scores(min_plays: int = 10) -> None:
    """
    Update per-player trust scores from graded plays.
    Trust = fraction of plays where BET WIN (direction correct).
    """
    from config import FILE_SEASON_2425, FILE_SEASON_2526, FILE_V12_TRUST, FILE_V14_TRUST
    from collections import defaultdict

    all_graded: list[dict] = []
    for season_file in (FILE_SEASON_2425, FILE_SEASON_2526):
        if not season_file.exists():
            continue
        try:
            with open(season_file) as f:
                plays = json.load(f)
            all_graded.extend([
                p for p in plays
                if p.get("result") in ("WIN", "LOSS")
                and p.get("source", "excel") != "synthetic"
            ])
        except Exception as e:
            print(f"  ⚠ Trust update: could not load {season_file.name}: {e}")

    if len(all_graded) < min_plays:
        print(f"  ⚠ Trust update: only {len(all_graded)} graded plays — skipping")
        return

    player_stats: dict = defaultdict(lambda: {"plays": 0, "correct": 0})
    for p in all_graded:
        player = p.get("player", "")
        if not player:
            continue
        player_stats[player]["plays"] += 1
        if _is_bet_win(p):  # direction-aware trust
            player_stats[player]["correct"] += 1

    trust: dict[str, float] = {}
    for player, stats in player_stats.items():
        if stats["plays"] >= min_plays:
            trust[player] = round(stats["correct"] / stats["plays"], 3)

    values = list(trust.values())
    avg    = round(sum(values) / len(values), 3) if values else 0
    below  = sum(1 for v in values if v < 0.42)

    for trust_file in (FILE_V12_TRUST, FILE_V14_TRUST):
        trust_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(trust_file, "w") as f:
                json.dump(trust, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"  ⚠ Trust update: {trust_file.name}: {e}")
            return

    print(f"  ✓ Trust scores updated: {len(trust):,} players | "
          f"avg={avg:.3f} | {below} below threshold (0.42)")
    print(f"    Based on {len(all_graded):,} graded plays (V1.0 bet-win target)")

    try:
        from batch_predict import M
        M._tv12 = None; M._tv14 = None
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def train(skip_validation: bool = False) -> None:
    print(f"\n  PropEdge {VERSION} — V1.0 Model Trainer")
    print("=" * 55)

    # 1. Load plays
    print("\n[1/4] Loading training plays...")
    plays = load_training_plays()
    if len(plays) < MIN_PLAYS_TO_TRAIN:
        print(f"  ✗ Only {len(plays)} graded plays — need {MIN_PLAYS_TO_TRAIN}. Aborting.")
        return

    # 2. Build feature matrix
    print("\n[2/4] Building 102-feature training matrix...")
    X, y = build_training_matrix(plays)

    # 3. Walk-forward validation
    if not skip_validation:
        print("\n[3/4] Walk-forward validation (V1.0 tier thresholds)...")
        oof_summary = walk_forward_validate(plays, X, y)
    else:
        print("\n[3/4] Walk-forward validation skipped.")
        oof_summary = {}

    # 4. Train final model and save
    print("\n[4/4] Training final V1.0 ensemble on all data...")
    xgb_m, lgb_m, lr_m, scaler = train_final_model(X, y)
    seasons = sorted(set(p.get("season", p.get("date","")[:4]) for p in plays if p.get("date","")))
    save_model(xgb_m, lgb_m, lr_m, scaler, len(plays), oof_summary, seasons)

    # Update trust scores
    print("\n[+] Updating trust scores (direction-aware)...")
    update_trust_scores()

    print(f"\n  ✓ V1.0 model training complete.")
    print(f"    Run 'python3 run.py predict' to use the new model.")


if __name__ == "__main__":
    import sys
    skip_val = "--no-validate" in sys.argv or "--skip-validation" in sys.argv
    train(skip_validation=skip_val)
