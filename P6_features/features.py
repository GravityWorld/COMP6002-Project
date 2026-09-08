"""
P6 — Feature assembly + splits. Scheduled Mon 14 / Tue 15 Sep (Week 2 GATE).
Placeholder implementation, but the SCHEMA below is written now because it
freezes Sep 15 and every one of Part 3's five DFL axes reads these three
files afterward — a guess here is an October retrofit
(scope_and_timeline_oct19.md).

Three artefacts (~20 lines of code each, ruinously expensive to retrofit):

  1. vehicle_id       — 10-50 virtual vehicles, partitioned by
                         town x degradation_type x severity.
                         This partition IS the non-IID split.
  2. edge_list_static — fully connected graph (T-static, idealised).
  3. edge_list_traffic— derived from scenario co-occurrence (T-traffic,
                         sparse/dynamic/realistic). Two vehicles are
                         neighbours in round t iff they're collaborating
                         in round t.

Split by scenario and town, NEVER by frame — frame-level splits leak
adjacent frames across the split and inflate every downstream number.

If a field needs to be added after the Sep 15 freeze: add it as a NEW
OPTIONAL column with a default, never as a change to an existing column —
that's the only way to keep every already-run result valid
(execution_plan_daily.md, "three things that will most likely go wrong").
"""


def build_vehicle_partition(manifest, n_vehicles: int):
    raise NotImplementedError("Week 2 — Mon 14 Sep task.")


def build_edge_list_static(vehicle_ids):
    raise NotImplementedError("Week 2 — Tue 15 Sep task (fully connected).")


def build_edge_list_traffic(manifest, vehicle_ids):
    raise NotImplementedError("Week 2 — Tue 15 Sep task (co-occurrence derived).")
