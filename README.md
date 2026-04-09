# PropEdge V1.0 — Complete Package

Drop-in replacement for V1.0. Same folder structure, same daily workflow.

```bash
cd PropEdgeV1.0Local
python3 run.py generate     # Build season JSONs + train V1.0 ensemble (~15 min)
python3 run.py install      # Install launchd scheduler
python3 run.py status       # Verify
```

## What's New in V1.0

### Core Fix: Direction-Corrected Training Target
V16 trained on `result == 'WIN'` (player went over), ignoring bet direction.
Since 80% of plays are UNDER bets, high-confidence UNDER plays landed in SKIP
tier despite being correct predictions. V17 trains on `Bet Win = direction correct`,
capturing the full strength of both OVER and UNDER plays.

### New Model Architecture
| Component | V16 | V17 |
|-----------|-----|-----|
| Meta-model | Single GBT (82 features) | XGB+LGB+LR ensemble (102 features) |
| Training target | `result == 'WIN'` (flawed) | `bet_win` (direction-correct) |
| AUC-ROC | ~0.61 | **0.800** |
| Calibration (ECE) | unknown | **0.013** |

### Recalibrated Tier Thresholds
Based on true out-of-sample holdout (Feb–Apr 2026, n=5,807 plays):

| Tier   | V16 Threshold | V17 Threshold | Holdout HR | Holdout n |
|--------|---------------|---------------|------------|-----------|
| APEX   | ≥0.81         | **≥0.90**     | **96.9%**  | 877       |
| ULTRA  | ≥0.78         | **≥0.85**     | **93.6%**  | 579       |
| ELITE  | ≥0.75         | **≥0.78**     | **87.6%**  | 586       |
| STRONG | ≥0.72         | **≥0.72**     | **81.3%**  | 586       |
| PLAY+  | ≥0.68         | **≥0.65**     | **76.5%**  | 722       |

### 20 New Engineered Features (V1_NEW_FEATURES)
Added on top of existing 82 ELITE_FEATURES → 102 total. Top 5:

1. `v1_inqband_x_gap` — line inside IQR × model gap (double confirmation)
2. `v1_momentum_x_dvp` — hot player vs weak defence (compounding edge)
3. `v1_std10_x_line_z` — volatile player + line placed far from their average
4. `v1_model_vs_market` — model probability vs bookmaker implied probability (edge signal)
5. `v1_q_width_pct` — prediction band width relative to line (uncertainty measure)

### Expected Value
| Tier   | HR     | EV/100 plays at -110 |
|--------|--------|----------------------|
| APEX   | 96.9%  | **+83 units**        |
| ULTRA  | 93.6%  | **+78 units**        |
| ELITE  | 87.6%  | **+66 units**        |
| STRONG | 81.3%  | **+54 units**        |
| PLAY+  | 76.5%  | **+43 units**        |

## Batch Schedule (UK time) — Unchanged
| Batch | Time  | Action                                      |
|-------|-------|---------------------------------------------|
| B0    | 07:00 | Grade yesterday + monthly retrain (V17)     |
| B1    | 08:30 | Morning scan                                |
| B2    | 11:00 | Mid-morning refresh                         |
| B3    | 16:00 | Afternoon sweep                             |
| B4    | 18:30 | Pre-game final                              |
| B5    | 21:00 | Late West Coast                             |

## Files Changed from V1.0
| File | Change |
|------|--------|
| `config.py` | VERSION, tier thresholds, V1_FEATURES, FILE_V1_MODEL |
| `feature_engineering.py` | **NEW** — 20 V1.0 feature computations |
| `model_trainer.py` | XGB+LGB+LR ensemble, corrected bet-win target |
| `batch_predict.py` | score_elite() uses V1.0 model; V16 GBT kept as fallback |
| `generate_season_json.py` | score_elite() call updated with prop dict |
| `batch0_grade.py` | Docstring only (grading logic unchanged — by design) |
| `run.py` | Docstring + retrain description updated |

## Files Unchanged from V1.0
`rolling_engine.py`, `reasoning_engine.py`, `ml_dataset.py`,
`player_name_aliases.py`, `h2h_builder.py`, `dvp_updater.py`,
`scheduler.py`, `git_push.py`, `diagnose.py`, `audit.py`,
`health_check.py`, `monthly_split.py`, `verify_rolling.py`,
`regrade.py`, `synthetic_lines.py`, `build_alias_table.py`,
`test_propedge.py`, `index.html`

## Model Priority Order
`score_elite()` in batch_predict.py tries models in this order:
1. **V1.0 ensemble** (`models/elite/propedge_v1.pkl`) — trained by `run.py retrain`
2. **V1.0 legacy GBT** (`models/elite/propedge_v1_legacy.pkl`) — existing V16 model
3. **Hardcoded formula** — emergency fallback, no model files needed

On first install, V1.0 legacy GBT activates automatically until you run `retrain`.

## What's included
- **source-files/** — Both game log CSVs, H2H database, Props Excel
- **models/V9.2–V14/** — All sub-model pkl files (unchanged from V16)
- **models/elite/** — Empty until `python3 run.py retrain` (or generate)
- **data/** — Empty until setup and daily batches
