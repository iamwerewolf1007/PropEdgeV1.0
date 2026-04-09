#!/usr/bin/env python3
"""
PropEdge V1.0 — fix_warnings.py
================================
Run this ONCE from your PropEdgeV1.0Local folder to fix the 3 remaining warnings:
  1. Removes stale V16/V17 launchd agents
  2. Fixes LOCAL_DIR in config.py (no-hyphen path)
  3. Reports GitHub Desktop repo status

Usage:
  python3 fix_warnings.py
"""
from __future__ import annotations
import subprocess, sys, platform
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
HOME = Path.home()

print(f"\n  PropEdge V1.0 — Fix Warnings")
print("=" * 50)
print(f"  Working dir: {ROOT}\n")

# ── Fix 1: Remove stale V16/V17 launchd agents ───────────────────────────────
print("[1/3] Removing stale V16/V17 launchd agents...")
if platform.system() != "Darwin":
    print("  Not macOS — skipping")
else:
    la = HOME / "Library" / "LaunchAgents"
    stale = [
        "com.propedge.v16.batch0", "com.propedge.v16.batch1",
        "com.propedge.v16.batch2", "com.propedge.v16.batch3",
        "com.propedge.v16.batch4", "com.propedge.v16.batch5",
        "com.propedge.v16.daily",
        "com.propedge.v17.batch0", "com.propedge.v17.batch1",
        "com.propedge.v17.batch2", "com.propedge.v17.daily",
    ]
    removed = 0
    for label in stale:
        plist = la / f"{label}.plist"
        if plist.exists():
            try:
                subprocess.run(["launchctl", "unload", str(plist)],
                               capture_output=True, timeout=5)
                plist.unlink()
                print(f"  ✓ Removed {label}")
                removed += 1
            except Exception as e:
                print(f"  ⚠ Could not remove {label}: {e}")
    if removed == 0:
        print("  — No stale agents found (already clean)")
    else:
        print(f"  → {removed} stale agent(s) removed")

# ── Fix 2: Patch LOCAL_DIR in config.py ──────────────────────────────────────
print("\n[2/3] Fixing LOCAL_DIR in config.py...")
cfg = ROOT / "config.py"
if cfg.exists():
    src = cfg.read_text(encoding="utf-8")
    original = src

    # Fix all variants that produce the hyphenated path at runtime
    fixes = [
        # The path literal that evaluates to PropEdgeV1.0-Local
        ('"PropEdgeV1.0-Local"',  '"PropEdgeV1.0Local"'),
        # Also catch any residual V16/V17-Local variants
        ('"PropEdgeV17.0-Local"', '"PropEdgeV1.0Local"'),
        ('"PropEdgeV16.0-Local"', '"PropEdgeV1.0Local"'),
        # Fix the whole LOCAL_DIR line as a fallback
        ('/ "PropEdgeV1.0-Local"',  '/ "PropEdgeV1.0Local"'),
    ]
    for old, new in fixes:
        src = src.replace(old, new)

    if src != original:
        cfg.write_text(src, encoding="utf-8")
        print("  ✓ config.py patched — LOCAL_DIR now points to PropEdgeV1.0Local")
        # Verify it evaluates correctly
        import importlib.util
        spec = importlib.util.spec_from_file_location("cfg_fix", cfg)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        print(f"  ✓ LOCAL_DIR runtime value: {m.LOCAL_DIR}")
    else:
        print("  — config.py already correct (no change needed)")
        # Show current value
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("cfg_fix2", cfg)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            print(f"  ℹ LOCAL_DIR = {m.LOCAL_DIR}")
        except:
            pass
else:
    print("  ✗ config.py not found")

# ── Fix 3: GitHub Desktop repo ───────────────────────────────────────────────
print("\n[3/3] GitHub Desktop repo check...")
repo = HOME / "Documents" / "GitHub" / "PropEdgeV1.0"
if repo.exists() and (repo / ".git").exists():
    print(f"  ✓ Repo found at {repo}")
    try:
        r = subprocess.run(["git", "-C", str(repo), "remote", "get-url", "origin"],
                           capture_output=True, text=True, timeout=5)
        print(f"  ✓ Remote: {r.stdout.strip()}")
    except:
        pass
else:
    print(f"  ⚠ Repo not found at {repo}")
    print()
    print("  To create it:")
    print("  1. Open GitHub Desktop")
    print("  2. File → Add Local Repository")
    print(f"     Path: {repo}")
    print("  3. OR: File → Clone Repository → URL tab")
    print("     URL: https://github.com/iamwerewolf1007/PropEdgeV1.0")
    print(f"     Local path: {HOME}/Documents/GitHub")
    print()
    print("  NOTE: This is only needed for git push to update the dashboard.")
    print("  The prediction engine works fully without it.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("  Done. Run python3 propedge_control.py to verify.")

# ── Critical reminder ─────────────────────────────────────────────────────────
print()
print("  ╔══════════════════════════════════════════════════╗")
print("  ║  REMINDER: Still need to generate fresh data:   ║")
print("  ║                                                  ║")
print("  ║   python3 run.py generate                        ║")
print("  ║                                                  ║")
print("  ║  This is what builds V1.0 predictions from       ║")
print("  ║  scratch using your Props Excel + game logs.     ║")
print("  ╚══════════════════════════════════════════════════╝")
print()

