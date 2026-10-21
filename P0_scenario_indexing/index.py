"""
P0 — Scenario indexing.

Builds manifest.parquet: (scenario_id, frame_id, ego_id, cav_ids, n_cav, town, occlusion)
Filtered to n_cav >= 3.

This is a pure metadata pass over the OPV2V directory structure — no detector,
no GPU. It is the input to almost everything downstream: W2's fleet partition
(P6) and W3's topology construction both read scenario/town/n_cav from here.

TODO (today, Tue 8 Sep):
  - Point OPV2V_ROOT at your extracted dataset (see decisions.md 5 Sep for the
    known split-zip reassembly gotcha on Windows: use `copy /b`, not `cat`).
  - Walk scenario folders, read per-CAV metadata (pose files / config yaml —
    check OpenCOOD's own dataset loader for the exact field names, since
    P1/P3 will reuse whatever loader convention you pick here).
  - Occlusion: OPV2V doesn't ship this directly for all splits — decide now
    whether this is a computed proxy (e.g. LiDAR point density) or dropped
    entirely, and write the decision to decisions.md today, not silently.
"""

from __future__ import annotations  # Py3.7 compat: allows list[Path] etc. without erroring

from pathlib import Path
from typing import List
import re

import pandas as pd
import yaml

# OPV2V's official train/validate/test splits (unlike the test_culver_city
# extract used through Sep) dump raw numpy scalars straight into frame
# yaml files via plain yaml.dump() -- yaml.safe_load() refuses the resulting
# `!!python/object/apply:numpy.core.multiarray.scalar` tag on security
# grounds. OpenCOOD's own opencood/hypes_yaml/yaml_utils.py:load_yaml()
# already solves exactly this (it reads every frame yaml in OpenCOOD's own
# pipeline) by using yaml.Loader instead of yaml.SafeLoader -- reused here
# rather than inventing a second loader. Safe in this context since OPV2V's
# yaml files are the project's own trusted downloaded dataset, not
# untrusted/attacker-controlled input.
from opencood.hypes_yaml.yaml_utils import load_yaml as _opencood_load_yaml

MIN_CAV = 3

# ASSUMPTION — verify with inspect_scenario() before trusting build_manifest().
# Standard OPV2V/OpenCOOD layout: scenario_dir/cav_id_dir/{timestamp}.yaml.
# Some OpenCOOD versions mark the ego CAV with a literal `ego: true` key in
# every frame's yaml; others rely on convention (lowest numeric CAV id in the
# scenario = ego). We try the explicit key first and fall back to convention,
# but PRINT which path was taken so a silent wrong guess doesn't go unnoticed.
CAV_DIR_PATTERN = re.compile(r"^\d+$")  # CAV subfolders are numeric IDs


def inspect_scenario(scenario_dir: str) -> None:
    """
    Run this FIRST, on one real scenario folder, before trusting
    build_manifest(). Prints the directory structure and the raw keys of one
    yaml file so you can confirm the assumptions above actually hold for
    your extract.
    """
    scenario_path = Path(scenario_dir)
    if not scenario_path.exists():
        raise FileNotFoundError(scenario_dir)

    cav_dirs = sorted(
        [d for d in scenario_path.iterdir() if d.is_dir() and CAV_DIR_PATTERN.match(d.name)]
    )
    print(f"Scenario: {scenario_path.name}")
    print(f"CAV subfolders found: {[d.name for d in cav_dirs]}")

    if not cav_dirs:
        print("!! No numeric CAV subfolders found — the layout assumption above is WRONG "
              "for this extract. Look at what's actually in this folder and fix "
              "CAV_DIR_PATTERN / the walk logic before proceeding.")
        return

    first_cav = cav_dirs[0]
    yaml_files = sorted(first_cav.glob("*.yaml"))
    print(f"Frame yamls in {first_cav.name}: {len(yaml_files)} found, e.g. {[f.name for f in yaml_files[:3]]}")

    if not yaml_files:
        print("!! No yaml files found in the first CAV folder — check the extension "
              "and structure directly.")
        return

    sample = _opencood_load_yaml(str(yaml_files[0]))
    print(f"Top-level keys in {yaml_files[0].name}: {list(sample.keys())}")
    print(f"Has explicit 'ego' key: {'ego' in sample}  (value: {sample.get('ego', 'N/A')})")
    print(f"Has 'lidar_pose': {'lidar_pose' in sample}")
    print("\nACTION: compare the keys above against what build_manifest() assumes "
          "(see the module docstring / comments) and fix any mismatch before "
          "running build_manifest() on the full dataset.")


def _discover_scenarios(root: Path) -> list[Path]:
    """
    OPV2V extracts sometimes put scenarios directly under root (e.g. your
    test_culver_city extract from Sat 5 Sep), sometimes under a split
    subfolder (train/validate/test). Handle both: a directory is a
    "scenario" if it directly contains numeric CAV subfolders.
    """
    scenarios = []

    def has_cav_subdirs(d: Path) -> bool:
        return any(c.is_dir() and CAV_DIR_PATTERN.match(c.name) for c in d.iterdir())

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if has_cav_subdirs(child):
            scenarios.append(child)
        else:
            # one level deeper (split-subfolder case)
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and has_cav_subdirs(grandchild):
                    scenarios.append(grandchild)

    return scenarios


def build_manifest(opv2v_root: str, out_path: str) -> pd.DataFrame:
    """
    Walk `opv2v_root`, build one row per (scenario_id, frame_id) with the
    columns listed above, filter to n_cav >= MIN_CAV, and write out_path
    as parquet. Returns the dataframe for immediate inspection.

    NOTE on 'town' and 'occlusion': OPV2V's frame-level yaml does not
    reliably carry a town field, and occlusion isn't shipped directly either.
    Both are written as None here rather than guessed — decide today whether
    town is parsed from a split manifest file (if your extract has one) or
    dropped, and whether occlusion is a computed proxy or dropped. Log
    whichever you choose in decisions.md; don't let this stay silently None
    into Week 2.
    """
    root = Path(opv2v_root)
    if not root.exists():
        raise FileNotFoundError(
            f"OPV2V root not found: {opv2v_root}. "
            "Confirm the checksum'd download/extract from Sat 5 Sep is in place "
            "before building the manifest."
        )

    scenario_dirs = _discover_scenarios(root)
    if not scenario_dirs:
        raise RuntimeError(
            f"No scenario folders with numeric CAV subfolders found under {opv2v_root}. "
            "Run inspect_scenario() on a folder you KNOW is a scenario to see what's "
            "actually there, then fix _discover_scenarios()."
        )

    rows = []
    for scenario_dir in scenario_dirs:
        scenario_id = scenario_dir.name
        cav_dirs = sorted(
            [d for d in scenario_dir.iterdir() if d.is_dir() and CAV_DIR_PATTERN.match(d.name)],
            key=lambda d: int(d.name),
        )
        cav_ids = [d.name for d in cav_dirs]
        n_cav = len(cav_ids)

        # Frame ids = yaml stems present for EVERY cav (intersection), since a
        # frame only qualifies if all CAVs reported for it.
        frame_sets = []
        for cav_dir in cav_dirs:
            frame_sets.append({f.stem for f in cav_dir.glob("*.yaml")})
        common_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []

        # Determine ego: try explicit 'ego' key on the first frame of each
        # CAV; fall back to "lowest numeric CAV id" convention if no CAV
        # yaml declares it explicitly. This fallback is a guess — confirmed
        # or refuted by inspect_scenario() before you trust it at scale.
        ego_id = cav_ids[0] if cav_ids else None
        for cav_dir, cid in zip(cav_dirs, cav_ids):
            sample_frame = next(cav_dir.glob("*.yaml"), None)
            if sample_frame is None:
                continue
            data = _opencood_load_yaml(str(sample_frame))
            if data.get("ego") is True:
                ego_id = cid
                break

        for frame_id in common_frames:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "frame_id": frame_id,
                    "ego_id": ego_id,
                    "cav_ids": cav_ids,
                    "n_cav": n_cav,
                    "town": None,   # see docstring — decide today, don't leave silently None
                    "occlusion": None,  # see docstring
                }
            )

    df = pd.DataFrame(
        rows,
        columns=["scenario_id", "frame_id", "ego_id", "cav_ids", "n_cav", "town", "occlusion"],
    )
    n_before = len(df)
    df = df[df["n_cav"] >= MIN_CAV] if not df.empty else df
    print(f"{n_before} total frames found, {len(df)} qualify with n_cav >= {MIN_CAV}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # cav_ids is a list column — parquet handles this fine via pyarrow, but
    # flagging it since it's the one non-scalar column in the schema.
    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--opv2v-root", help="Root to walk for build_manifest")
    parser.add_argument("--out", default="./cache/manifest.parquet")
    parser.add_argument("--inspect", help="Path to ONE scenario folder to inspect first")
    args = parser.parse_args()

    if args.inspect:
        inspect_scenario(args.inspect)
    elif args.opv2v_root:
        manifest = build_manifest(args.opv2v_root, args.out)
        print(f"Wrote {len(manifest)} qualifying frames to {args.out}")
    else:
        parser.error("Pass either --inspect <scenario_dir> (do this first) or --opv2v-root <root>")
