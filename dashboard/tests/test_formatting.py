from dashboard import formatting as f


def _report(warnings=None):
    return {
        "metrics": {
            "dsr": {"warnings": []},
            "pbo": {"warnings": warnings or []},
            "decay_ratio": {"warnings": ["only 2 folds (< 4): insufficient history for a reliable decay ratio"]},
        },
        "thresholds": {"pbo": {"green": 0.2, "yellow": 0.4}, "dsr": {"green": 0.95, "yellow": 0.5},
                       "decay_ratio": {"green": 0.7, "yellow": 0.4}},
    }


def test_grade_color_and_emoji_fall_back_to_unknown():
    assert f.grade_color("green") == f.GRADE_COLORS["green"]
    assert f.grade_color("purple") == f.GRADE_COLORS["unknown"]
    assert f.grade_emoji("red") == "🔴"
    assert f.grade_emoji("??") == "⚪"


def test_format_value_and_pct():
    assert f.format_value(None) == "n/a"
    assert f.format_value(0.31456) == "0.31"
    assert f.format_value(2, 0) == "2"
    assert f.format_value("x") == "x"
    assert f.format_pct(0.934) == "93.4%"
    assert f.format_pct(None) == "n/a"


def test_collect_warnings_keeps_metric_order():
    ws = f.collect_warnings(_report(["only 3 strategies (< 5); PBO is coarse"]))
    assert [m for m, _ in ws] == ["pbo", "decay_ratio"]


def test_has_insufficient_history():
    assert f.has_insufficient_history(_report())
    clean = {"metrics": {"dsr": {"warnings": []}}}
    assert not f.has_insufficient_history(clean)


def test_threshold_caption_direction():
    th = _report()["thresholds"]
    assert f.threshold_caption("pbo", th).startswith("green < 0.20")
    assert f.threshold_caption("dsr", th).startswith("green > 0.95")


def test_metric_title():
    assert f.metric_title("pbo").startswith("Probability")
    assert f.metric_title("other") == "other"
