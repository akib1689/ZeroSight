"""Tests for metrics/exec_accuracy.py — execution-equivalence scoring."""

import sqlite3

from metrics.exec_accuracy import (
    _values_equal,
    result_sets_equal,
    score_execution_accuracy,
)


def test_mixed_none_and_str_identical_sets():
    """Regression: BIRD bird_397 — a column mixing NULLs and strings must sort."""
    pred = [("{5}{W}{W}",), (None,), ("{3}{W}",)]
    gold = [("{3}{W}",), ("{5}{W}{W}",), (None,)]
    assert result_sets_equal(pred, gold)


def test_mixed_none_and_str_different_sets():
    pred = [("{5}{W}{W}",), (None,), ("{3}{W}",)]
    gold = [("{4}{W}",), (None,), ("{3}{W}",)]
    assert not result_sets_equal(pred, gold)


def test_mixed_none_and_float():
    pred = [(None,), (1.5,), (2.0,)]
    gold = [(2.0,), (None,), (1.5,)]
    assert result_sets_equal(pred, gold)


def test_mixed_int_and_str():
    pred = [("abc",), (1,), (2,)]
    gold = [(2,), ("abc",), (1,)]
    assert result_sets_equal(pred, gold)


def test_bytes_column():
    pred = [(b"x",), (b"y",)]
    gold = [(b"y",), (b"x",)]
    assert result_sets_equal(pred, gold)


def test_multi_column_mixed_types():
    pred = [(None, "a"), (1, None), ("x", 2.0)]
    gold = [(1, None), ("x", 2.0), (None, "a")]
    assert result_sets_equal(pred, gold)


def test_numeric_order_insensitivity():
    assert result_sets_equal([(10,), (9,), (2,)], [(2,), (9,), (10,)])


def test_float_tolerance_preserved():
    assert result_sets_equal([(1.0000001,), (2.5,)], [(1.0000002,), (2.5,)])


def test_row_count_mismatch_short_circuits():
    assert not result_sets_equal([(1,)], [(1,), (2,)])


def test_empty_sets_equal():
    assert result_sets_equal([], [])


def test_values_equal_none_none():
    assert _values_equal(None, None)


def test_values_equal_none_vs_value():
    assert not _values_equal(None, "x")


def test_score_execution_accuracy_with_nulls(tmp_path):
    db_path = tmp_path / "card_games.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE cards (manaCost TEXT)")
    conn.executemany(
        "INSERT INTO cards VALUES (?)",
        [("{5}{W}{W}",), (None,), ("{3}{W}",)],
    )
    conn.commit()
    conn.close()

    sql = "SELECT manaCost FROM cards"
    result = score_execution_accuracy(sql, sql, str(db_path))
    assert result["correct"] is True
    assert result["pred_row_count"] == 3
    assert result["gold_row_count"] == 3
