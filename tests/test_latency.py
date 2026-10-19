import pytest

from extension.latency import JUDGES, percentile


def test_percentile_interpolates():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 0) == 10 and percentile(values, 100) == 50
    assert percentile(values, 95) == pytest.approx(48)


def test_json_round_trip_turns_tuples_into_lists():
    # Why latency.py compares lists: cached DAT scores come back from JSON as lists.
    import json
    assert json.loads(json.dumps({"parsed": (3, 4)}))["parsed"] == [3, 4] != (3, 4)


def test_rotation_gives_every_judge_every_position():
    seen = {judge: set() for judge in JUDGES}
    for position in range(3):
        shift = position % len(JUDGES)
        for slot, judge in enumerate(JUDGES[shift:] + JUDGES[:shift]):
            seen[judge].add(slot)
    assert all(slots == {0, 1, 2} for slots in seen.values())
