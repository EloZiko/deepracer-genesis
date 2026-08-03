"""The track catalog API (deepracer_genesis.tracks)."""

import pytest

from deepracer_genesis import tracks


def test_names_lists_all_tracks_sorted():
    ns = tracks.names()
    assert ns == sorted(ns) and len(ns) >= 3
    assert {"reinvent_base", "Oval_track"} <= set(ns)


def test_base_and_generated_partition_names():
    assert set(tracks.base()) | set(tracks.generated()) == set(tracks.names())
    assert set(tracks.base()) & set(tracks.generated()) == set()
    assert "reinvent_base" in tracks.base()


def test_exists_and_require():
    assert tracks.exists("reinvent_base") and not tracks.exists("nope")
    assert tracks.require("reinvent_base") == "reinvent_base"
    with pytest.raises(KeyError, match="unknown track"):
        tracks.require("nope")


def test_info_metadata():
    i = tracks.info("reinvent_base")
    assert i.source == "base" and i.num_waypoints == 118
    assert 15 < i.length_m < 20 and i.avg_width_m > 0
    assert i.route_path.endswith(".npy")


def test_catalog_covers_every_track():
    assert {c.name for c in tracks.catalog()} == set(tracks.names())
