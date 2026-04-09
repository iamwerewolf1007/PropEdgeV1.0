#!/usr/bin/env python3
"""
PropEdge V1.0 — setup_v1.py
============================
Run ONCE from your PropEdgeV1.0Local directory.

What this does
--------------
  1. Verifies all V1.0 version strings are correct in every file
  2. Copies sub-models (V9.2, V10, V11, V12, V14 pkl files) from V16.0-Local
  3. Copies source files (game logs, Props Excel, H2H) from V16.0-Local
  4. Does NOT copy season JSONs or ML dataset — those are generated fresh
  5. Uninstalls stale V16 launchd agents (macOS only)
  6. Runs the control check to confirm system state

After setup, you must run:
  python3 run.py generate   ← builds fresh season JSONs + trains V1.0 model (~30 min)

Why "generate" not "retrain"
-----------------------------
  retrain  = train meta-model on EXISTING season JSONs (uses old V16 data)
  generate = read Props Excel → score each prop through sub-models → grade
             against actual game logs → write FRESH season JSONs with V1.0
             predictions → then train V1.0 meta-model

  generate is the correct command for a clean independent V1.0 installation.

Usage:
  cd ~/Documents/GitHub/PropEdgeV1.0Local
  python3 setup_v1.py [--dry-run] [--no-copy] [--no-uninstall] [--no-check]
"""
from __future__ import annotations
import os, sys, shutil, argparse, subprocess, platform
from pathlib import Path

ROOT      = Path(__file__).parent.resolve()
HOME      = Path.home()
V16_LOCAL = HOME / "Documents" / "GitHub" / "PropEdgeV16.0-Local"

GIT_REMOTE_V1  = "git@github.com:iamwerewolf1007/PropEdgeV1.0.git"
EXPECTED_LOCAL = HOME / "Documents" / "GitHub" / "PropEdgeV1.0Local"
EXPECTED_REPO  = HOME / "Documents" / "GitHub" / "PropEdgeV1.0"


# ─────────────────────────────────────────────────────────────────────────────
# WHAT TO COPY FROM V16
# Only sub-models and source data — NOT season JSONs or ML dataset
# Season JSONs are generated fresh by "python3 run.py generate"
# ─────────────────────────────────────────────────────────────────────────────
COPY_MANIFEST = [
    # Source files — raw inputs the prediction engine reads
    ("source-files/nba_gamelogs_2024_25.csv",
     "source-files/nba_gamelogs_2024_25.csv"),
    ("source-files/nba_gamelogs_2025_26.csv",
     "source-files/nba_gamelogs_2025_26.csv"),
    ("source-files/h2h_database.csv",
     "source-files/h2h_database.csv"),
    ("source-files/PropEdge_-_Match_and_Player_Prop_lines_.xlsx",
     "source-files/PropEdge_-_Match_and_Player_Prop_lines_.xlsx"),
    ("source-files/PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx",
     "source-files/PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx"),
    # Sub-models — fixed V9.2–V14 pkl files (never retrained here)
    ("models/V9.2/projection_model.pkl",       "models/V9.2/projection_model.pkl"),
    ("models/V10/projection_model.pkl",        "models/V10/projection_model.pkl"),
    ("models/V11/projection_model.pkl",        "models/V11/projection_model.pkl"),
    ("models/V12/projection_model.pkl",        "models/V12/projection_model.pkl"),
    ("models/V12/direction_classifier.pkl",    "models/V12/direction_classifier.pkl"),
    ("models/V12/calibrator.pkl",              "models/V12/calibrator.pkl"),
    ("models/V12/segment_model.pkl",           "models/V12/segment_model.pkl"),
    ("models/V12/quantile_models.pkl",         "models/V12/quantile_models.pkl"),
    ("models/V14/projection_model.pkl",        "models/V14/projection_model.pkl"),
    ("models/V14/direction_classifier.pkl",    "models/V14/direction_classifier.pkl"),
    ("models/V14/calibrator.pkl",              "models/V14/calibrator.pkl"),
    # Legacy meta-model — used as fallback ONLY during initial generate
    # (before propedge_v1.pkl exists). Renamed to v1_legacy.
    ("models/elite/propedge_elite_v2.pkl",     "models/elite/propedge_v1_legacy.pkl"),
]

# Files NOT copied intentionally:
# data/season_2024_25.json   — generated fresh by: python3 run.py generate
# data/season_2025_26.json   — generated fresh by: python3 run.py generate
# data/propedge_ml_dataset.xlsx — generated fresh by: python3 run.py generate
# data/audit_log.csv         — starts fresh
# data/dvp_rankings.json     — regenerated from game logs
# data/today.json            — populated by predict batches
# models/V12/player_trust.json   — regenerated from fresh graded plays
# models/V14/player_trust.json   — same
# models/elite/propedge_v17.pkl  — this is V17 naming; V1.0 version created by generate


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY VERSION STRINGS
# ─────────────────────────────────────────────────────────────────────────────
def check_version_strings() -> tuple[bool, list[str]]:
    stale = ["V16.0", "V17.0", "PropEdgeV16.0", "PropEdgeV17.0",
             "v17_", "V17_", "build_v17", "FILE_V17",
             "propedge_v17.pkl", "propedge_elite_v2.pkl"]
    skip_files = {"setup_v1.py", "propedge_control.py"}
    issues = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file(): continue
        if path.suffix not in {'.py', '.html', '.md', '.txt'}: continue
        if path.name in skip_files: continue
        if any(p.startswith('.') for p in path.parts): continue
        if '__pycache__' in str(path): continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except:
            continue
        found = [s for s in stale if s in text]
        if found:
            issues.append(f"{path.relative_to(ROOT)}: {found}")
    return len(issues) == 0, issues


# ─────────────────────────────────────────────────────────────────────────────
# COPY FROM V16
# ─────────────────────────────────────────────────────────────────────────────
def copy_from_v16(dry: bool) -> dict:
    stats = {"copied": 0, "exists": 0, "missing": 0, "errors": 0}
    if not V16_LOCAL.exists():
        print(f"  ⚠ V16.0-Local not found at {V16_LOCAL}")
        print(f"    Copy files manually — see SETUP_LOCAL.md")
        return stats

    for src_rel, dst_rel in COPY_MANIFEST:
        src = V16_LOCAL / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            stats["missing"] += 1
            print(f"  —   {src_rel}  (not in V16)")
            continue
        if dst.exists():
            stats["exists"] += 1
            print(f"  ✓   {dst_rel}  (already present — skip)")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dry:
                shutil.copy2(src, dst)
            size_mb = src.stat().st_size / 1048576
            stats["copied"] += 1
            print(f"  {'[dry]' if dry else '←'}   {dst_rel}  ({size_mb:.1f} MB)")
        except Exception as e:
            stats["errors"] += 1
            print(f"  ✗   {dst_rel}: {e}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# UNINSTALL STALE AGENTS
# ─────────────────────────────────────────────────────────────────────────────
def uninstall_stale_agents(dry: bool):
    if platform.system() != "Darwin":
        print("  Not on macOS — skipping launchd cleanup")
        return
    stale_labels = [
        "com.propedge.v16.batch0", "com.propedge.v16.batch1",
        "com.propedge.v16.batch2", "com.propedge.v16.batch3",
        "com.propedge.v16.batch4", "com.propedge.v16.batch5",
        "com.propedge.v16.daily",
        "com.propedge.v17.batch0", "com.propedge.v17.batch1",
        "com.propedge.v17.batch2", "com.propedge.v17.daily",
    ]
    agents_dir = HOME / "Library" / "LaunchAgents"
    removed = 0
    for label in stale_labels:
        plist = agents_dir / f"{label}.plist"
        if plist.exists():
            if not dry:
                try:
                    subprocess.run(["launchctl", "unload", str(plist)],
                                   capture_output=True, timeout=5)
                    plist.unlink()
                    removed += 1
                    print(f"  ✓ Removed {label}")
                except Exception as e:
                    print(f"  ⚠ Could not remove {label}: {e}")
            else:
                print(f"  [dry] Would remove {label}")
                removed += 1
    if removed == 0:
        print("  No stale agents found")
    else:
        print(f"  {removed} stale agent(s) removed")


# ─────────────────────────────────────────────────────────────────────────────
# ENSURE DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dirs(dry: bool):
    dirs = ['data', 'logs', 'source-files',
            'models/V9.2', 'models/V10', 'models/V11',
            'models/V12', 'models/V14', 'models/elite']
    for d in dirs:
        p = ROOT / d
        if p.exists():
            print(f"  ✓ {d}/")
        else:
            if not dry:
                p.mkdir(parents=True, exist_ok=True)
            print(f"  {'[dry] create' if dry else '✓ created'} {d}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--no-copy",       action="store_true")
    parser.add_argument("--no-uninstall",  action="store_true")
    parser.add_argument("--no-check",      action="store_true")
    args = parser.parse_args()
    dry = args.dry_run

    print(f"\n  PropEdge V1.0 — Setup  {'[DRY RUN] ' if dry else ''}")
    print("=" * 55)
    print(f"  Local folder: {ROOT}")
    print(f"  V16 source:   {V16_LOCAL}")

    # ── Step 1: Verify version strings ───────────────────────────────────────
    print("\n[1/5] Checking version strings...")
    clean, issues = check_version_strings()
    if clean:
        print("  ✓ All files are V1.0 clean")
    else:
        print(f"  ✗ {len(issues)} file(s) still have stale version strings:")
        for i in issues[:5]:
            print(f"    {i}")
        print("  These need to be fixed before proceeding.")
        sys.exit(1)

    # ── Step 2: Copy source files and sub-models from V16 ────────────────────
    if not args.no_copy:
        print("\n[2/5] Copying source files and sub-models from V16.0-Local...")
        print("  (season JSONs and ML dataset are NOT copied — will be generated fresh)")
        stats = copy_from_v16(dry)
        print(f"\n  → {stats['copied']} copied | {stats['exists']} already present | "
              f"{stats['missing']} not found | {stats['errors']} errors")
    else:
        print("\n[2/5] Copy skipped (--no-copy)")

    # ── Step 3: Ensure required directories ──────────────────────────────────
    print("\n[3/5] Ensuring required directories...")
    ensure_dirs(dry)

    # ── Step 4: Uninstall stale V16 launchd agents ───────────────────────────
    if not args.no_uninstall:
        print("\n[4/5] Removing stale V16/V17 launchd agents...")
        uninstall_stale_agents(dry)
    else:
        print("\n[4/5] Launchd cleanup skipped (--no-uninstall)")

    # ── Step 5: Control check ─────────────────────────────────────────────────
    if not args.no_check and not dry:
        print("\n[5/5] Running control check...\n")
        ctrl = ROOT / "propedge_control.py"
        if ctrl.exists():
            subprocess.run([sys.executable, str(ctrl)], cwd=str(ROOT))
    elif dry:
        print("\n[5/5] Control check skipped (dry run)")
    else:
        print("\n[5/5] Control check skipped (--no-check)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    if dry:
        print("  DRY RUN complete. Run without --dry-run to apply.")
    else:
        print("  Setup complete.")
        print()
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║  IMPORTANT: Run this next to generate fresh data     ║")
        print("  ║                                                      ║")
        print("  ║   python3 run.py generate                            ║")
        print("  ║                                                      ║")
        print("  ║  This reads your Props Excel files and game logs,    ║")
        print("  ║  scores each prop through the prediction engine,     ║")
        print("  ║  grades against actual results, writes fresh season  ║")
        print("  ║  JSONs, and trains the V1.0 meta-model. (~30 min)    ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print()
        print("  After generate:")
        print("    python3 run.py install           ← install scheduler")
        print("    python3 propedge_control.py      ← verify everything")
    print()


if __name__ == "__main__":
    main()
