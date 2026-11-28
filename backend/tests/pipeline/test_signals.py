"""Tests for Signals persistence and profile-capped signal bonuses."""
from app.pipeline.signals import Signals, load_signals, save_signals, signal_bonus


def test_signal_bonus_caps_by_profile():
    signals = Signals(
        chat=[{"start": 10.0, "end": 20.0, "value": 1.0}],
        audio=[{"start": 15.0, "end": 18.0, "value": 1.0}],
    )
    # chat 1.0 -> 15 for stream, 8 for podcast; audio 1.0 -> 6
    assert signal_bonus(12.0, 25.0, signals, "stream") == 21
    assert signal_bonus(12.0, 25.0, signals, "podcast") == 14
    # chat only overlap
    assert signal_bonus(10.0, 14.0, signals, "stream") == 15
    assert signal_bonus(10.0, 14.0, signals, "podcast") == 8
    # no overlap
    assert signal_bonus(30.0, 40.0, signals, "stream") == 0


def test_signals_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "nested" / "signals.json"
    assert load_signals(path) == Signals()
    s = Signals(chat=[{"start": 1.0, "end": 11.0, "value": 0.5}], audio=[{"start": 5.0, "end": 6.0, "value": 0.8}])
    save_signals(path, s)
    assert load_signals(path) == s
