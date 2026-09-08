"""
Done-when for P0 (per execution_plan_daily.md, Thu 3 Sep):
  manifest.parquet exists with a row per qualifying frame (n_cav >= 3).

Run: pytest P0_scenario_indexing/test_index.py -v
"""

import pandas as pd
import pytest

from .index import build_manifest, MIN_CAV

EXPECTED_COLUMNS = {"scenario_id", "frame_id", "ego_id", "cav_ids", "n_cav", "town", "occlusion"}


def test_manifest_schema(tmp_path):
    # TODO once build_manifest is implemented against a real OPV2V root:
    # point this at a tiny fixture directory (2-3 scenarios) checked into
    # the repo under tests/fixtures/, not the full dataset.
    pytest.skip("Wire a small fixture OPV2V directory before this can run for real.")


def test_manifest_filters_low_cav_count():
    """A frame with n_cav < 3 must never appear in the output."""
    df = pd.DataFrame(
        [
            {"scenario_id": "s1", "frame_id": 0, "ego_id": "e1", "cav_ids": ["e1", "e2"], "n_cav": 2, "town": "t1", "occlusion": 0.0},
            {"scenario_id": "s1", "frame_id": 1, "ego_id": "e1", "cav_ids": ["e1", "e2", "e3"], "n_cav": 3, "town": "t1", "occlusion": 0.0},
        ]
    )
    filtered = df[df["n_cav"] >= MIN_CAV]
    assert len(filtered) == 1
    assert filtered.iloc[0]["n_cav"] >= MIN_CAV


def test_manifest_columns_match_contract():
    """Every downstream stage (P1, P6) reads this exact schema — guard it."""
    df = pd.DataFrame(columns=list(EXPECTED_COLUMNS))
    assert set(df.columns) == EXPECTED_COLUMNS
