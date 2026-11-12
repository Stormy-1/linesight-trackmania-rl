import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_map_split import DEV_MAPS, MIN_FAILING_IN_TEST, MIN_FINISHING_IN_TEST, TEST_PER_TIER, make_split

TIERS = ("White", "Green", "Blue", "Red", "Black")


def fake_manifest():
    manifest = {}
    for letter, tier in zip("ABCDE", TIERS):
        for n in range(1, 16 if tier != "Black" else 4):
            manifest[f"{letter}{n:02d}-Race"] = {"group": tier, "centerline": "x.npy", "suspect": False}
    for stem in DEV_MAPS:
        manifest.setdefault(stem, {"group": "White", "centerline": "x.npy", "suspect": False})
    manifest["E03-Race"]["suspect"] = True
    return manifest


def every_third_finishes(manifest):
    return {m: i % 3 == 0 for i, m in enumerate(sorted(manifest))}


def test_split_is_deterministic_for_a_seed():
    manifest = fake_manifest()
    status = every_third_finishes(manifest)
    assert make_split(manifest, status, seed=7) == make_split(manifest, status, seed=7)


def test_test_set_is_stratified_by_tier_and_excludes_dev_maps():
    manifest = fake_manifest()
    split = make_split(manifest, every_third_finishes(manifest), seed=1)
    for tier, count in TEST_PER_TIER.items():
        assert sum(manifest[m]["group"] == tier for m in split["test"]) == count
    assert not set(split["test"]) & set(DEV_MAPS)


def test_test_set_mixes_finishing_and_failing_maps():
    manifest = fake_manifest()
    status = every_third_finishes(manifest)
    split = make_split(manifest, status, seed=3)
    finishing = sum(status[m] for m in split["test"])
    assert finishing >= MIN_FINISHING_IN_TEST
    assert len(split["test"]) - finishing >= MIN_FAILING_IN_TEST


def test_rl_train_and_offline_data_never_contain_test_maps():
    manifest = fake_manifest()
    split = make_split(manifest, every_third_finishes(manifest), seed=5)
    test = set(split["test"])
    assert not test & set(split["rl_train"])
    assert not test & set(split["offline_train"])
    assert "E03-Race" not in split["offline_train"]  # suspect replay


def test_impossible_mix_raises():
    manifest = fake_manifest()
    with pytest.raises(ValueError):
        make_split(manifest, {m: True for m in manifest}, seed=1)


def test_committed_split_is_consistent():
    split = json.loads((ROOT / "maps" / "map_split.json").read_text())
    test = set(split["test"])
    assert len(test) == len(split["test"]) >= 8
    assert not test & set(split["dev"])
    assert not test & set(split["rl_train"])
    assert not test & set(split["offline_train"])
