from silent_speech_interpretability.data.textgrid import (
    TextGridInterval,
    read_textgrid,
    write_textgrid,
)


def test_textgrid_round_trip_preserves_labeled_intervals(tmp_path):
    path = tmp_path / "example.TextGrid"
    tiers = {
        "words": [
            TextGridInterval(0.2, 0.8, 'say "hello"'),
            TextGridInterval(1.0, 1.5, "now"),
        ],
        "phones": [
            TextGridInterval(0.2, 0.5, "s"),
            TextGridInterval(0.5, 0.8, "eɪ"),
        ],
    }

    write_textgrid(path, 2.0, tiers)
    parsed = read_textgrid(path)

    assert parsed == tiers
    assert 'text = "say ""hello"""' in path.read_text(encoding="utf-8")


def test_textgrid_rejects_overlapping_intervals(tmp_path):
    path = tmp_path / "overlap.TextGrid"
    intervals = [
        TextGridInterval(0.0, 0.7, "a"),
        TextGridInterval(0.6, 1.0, "b"),
    ]

    try:
        write_textgrid(path, 1.0, {"phones": intervals})
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected overlapping intervals to be rejected")
