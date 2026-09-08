"""
P7 — Golden-file regression test (Tue 8 Sep task).

Freezes defence outputs on 5 scenarios as golden files, with a test that
fails on drift. The defence code changes a dozen times over the next four
weeks (execution_plan_daily.md) — this is what stops a silent behaviour
change from poisoning every downstream number.

Usage:
  1. Once the full defence is wired (consistency layer + detections -> flag
     decision) and run on 5 clean scenarios, call `freeze_golden(...)` ONCE
     to write the golden files.
  2. From then on, `test_no_drift_from_golden` runs on every change and
     fails loudly if defence outputs drift from the frozen baseline.
  3. Re-run this whole file after Sat 3 Oct's attack-detection regression
     check per the plan (execution_plan_daily.md, Mon 5 Oct) — one final time
     before Tue 6 Oct's results freeze.

Do NOT regenerate golden files just to make a failing test pass — that
defeats the point. Regenerate only when a change to the defence is
deliberate and reviewed, and note the reason in decisions.md when you do.
"""

from __future__ import annotations  # Py3.7 compat: allows list[str] etc. without erroring

import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"


def freeze_golden(scenario_id: str, flag_results: dict) -> None:
    """
    flag_results: whatever the wired defence produces per scenario — e.g.
    {"frame_0": {"flagged": [...], "conflicting_area_m2": [...]}, ...}
    Must be JSON-serialisable (cast numpy types to native Python first).
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{scenario_id}.json"
    with open(path, "w") as f:
        json.dump(flag_results, f, indent=2, sort_keys=True)


def load_golden(scenario_id: str) -> dict:
    path = GOLDEN_DIR / f"{scenario_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No golden file for {scenario_id} — run freeze_golden() first, "
            "once the full defence is wired and you trust its output."
        )
    with open(path) as f:
        return json.load(f)


def compare_to_golden(scenario_id: str, current_results: dict, atol: float = 1e-6) -> list[str]:
    """Returns a list of human-readable mismatch descriptions; empty = match."""
    golden = load_golden(scenario_id)
    mismatches = []

    if set(golden.keys()) != set(current_results.keys()):
        mismatches.append(
            f"{scenario_id}: frame key mismatch — "
            f"golden has {set(golden.keys()) - set(current_results.keys())} extra, "
            f"current has {set(current_results.keys()) - set(golden.keys())} extra"
        )

    for frame_id in golden.keys() & current_results.keys():
        g, c = golden[frame_id], current_results[frame_id]
        for key in g:
            if key not in c:
                mismatches.append(f"{scenario_id}/{frame_id}: missing key '{key}' in current output")
                continue
            gv, cv = g[key], c[key]
            if isinstance(gv, (int, float)) and isinstance(cv, (int, float)):
                if not np.isclose(gv, cv, atol=atol):
                    mismatches.append(f"{scenario_id}/{frame_id}/{key}: golden={gv} current={cv}")
            elif gv != cv:
                mismatches.append(f"{scenario_id}/{frame_id}/{key}: golden={gv} current={cv}")

    return mismatches


@pytest.mark.skip(reason="Enable once golden files are frozen for the 5 chosen scenarios.")
def test_no_drift_from_golden():
    """
    Wire this up once the 5 golden scenarios are chosen and frozen:
      for scenario_id in FIVE_SCENARIOS:
          current = run_defence(scenario_id)  # your wired P4 pipeline
          mismatches = compare_to_golden(scenario_id, current)
          assert not mismatches, "\\n".join(mismatches)
    """
    pass
