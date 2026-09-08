import pytest


def test_not_yet_scheduled():
    pytest.skip("P6 is a Week 2 (Mon 14 / Tue 15 Sep, GATE) task — not today's priority.")


def test_no_scenario_in_two_splits_placeholder():
    pytest.skip(
        "Once splits exist: assert no scenario_id appears in more than one "
        "of train/val/test. This guard is explicitly required by the plan."
    )
