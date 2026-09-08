# Thesis repo — P0-P7 pipeline

DFL-headline pivot (decided 2026-09-02). See project docs:
`scope_and_timeline_oct19.md`, `execution_plan_daily.md`, `decisions.md`.

## Environment

Reuses the `opencood` conda env validated Sat 5 Sep 2026 (Python 3.7.11,
PyTorch 1.13.1+cu117, spconv-cu117, RTX 4070 Ti). Don't create a second env —
install `requirements.txt` into the existing one.

Known environment fixes already applied and documented in `decisions.md`
(reapply if the repo/env is ever reset):
- `pip install -r requirements.txt` then `python setup.py develop --no-deps`
  (not plain `setup.py develop` — breaks on modern scikit-image packaging)
- CUDA Toolkit 11.7 installed alongside the newer driver (spconv needs the
  toolkit's runtime DLLs, not just the driver)
- `num_workers=0` in OpenCOOD's inference DataLoader (Windows spawn +
  spconv pickling issue)
- `extract_timestamps()` patched to `os.path.basename(file)` — OpenCOOD's
  original Unix-path assumption breaks on Windows paths

## Stage layout

| Stage | Status as of Tue 8 Sep | Scheduled |
|---|---|---|
| P0 scenario indexing | scaffolded, not implemented | today |
| P1 alignment | scaffolded, not implemented | today/tomorrow |
| P2 degradation | scaffolded only | Week 2 (Wed/Thu 9-10 Sep) |
| P3 perception caching | scaffolded only | Week 2 (Fri 11 Sep) |
| P4 defence (consistency layer) | **working implementation**, needs validation against CAD's code + real data | today |
| P5 labelling | scaffolded only | Week 2 (Sun 13 Sep) |
| P6 features (schema) | schema documented, not implemented | Week 2 GATE (Mon/Tue 14-15 Sep) |
| P7 golden-file test | harness ready, needs 5 scenarios frozen | today |

## Today's actual priority order (catch-up day, Tue 8 Sep)

1. Fill in `configs/paths.yaml` (check disk space for cache_root NOW).
2. P0: implement `build_manifest()` against your real OPV2V extract.
3. P4: validate `consistency_layer.py` against CAD's released code
   (arXiv:2309.12955) — confirm RANSAC/DBSCAN params and the exact 2.7 m²
   threshold definition; run the tests in `test_consistency_layer.py`.
4. P1: implement `align_to_ego()`, reusing the pose-parsing convention
   already working from Sat 5 Sep's detector inference.
5. Wire P4 + the detector together, run on 5 clean scenarios, get a clean
   FPR number, write it in `decisions.md`.
6. Freeze golden files (`P7_evaluation/golden_file_test.py::freeze_golden`)
   on those 5 scenarios.
7. Write today's gate decision in `decisions.md`.

## Non-negotiable habit

Append to `decisions.md` daily — what ran, what the number was, what you
changed. This is the raw material for the Week 7 methods section.
