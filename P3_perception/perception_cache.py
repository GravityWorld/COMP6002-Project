"""
P3 — Frozen perception pass + caching.

Scheduled for Fri 11 Sep (Week 2 pilot). Placeholder for now.

Reuse the exact PointPillars checkpoint + inference path validated on Sat 5
Sep (epoch 30, late-fusion, OpenCOOD model zoo) — don't re-derive the
environment fix-ups (num_workers=0, the extract_timestamps() Windows-path
patch) from scratch; they're already documented in decisions.md.

Later (per scope_and_timeline_oct19.md decision 1): this stage also caches
BEV feature crops for Part 2's correction head, not just detections — plan
the cache schema with that second consumer in mind when you build this for
real, even though only detections/FPR need it in Week 1-2.
"""


def run_frozen_inference_and_cache(scenario_ids, out_dir):
    raise NotImplementedError(
        "Week 2 — Fri 11 Sep task. Reuse Sat 5 Sep's validated checkpoint + "
        "environment fixes (see decisions.md) rather than re-deriving them."
    )
