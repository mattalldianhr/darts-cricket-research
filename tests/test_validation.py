"""Unit tests for validation.py — tournament runner and Frongello principles."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import AIPlayer
from strategies import ChaseAndCover, LeadThenCover, SequentialCloser
from validation import (
    MatchupResult,
    PrincipleResult,
    RoundRobinResult,
    ValidationReport,
    run_matchup,
    run_round_robin,
)


# ── run_matchup tests ────────────────────────────────────────────────


def test_matchup_totals():
    """wins_a + wins_b should equal total_games."""
    bot_a = SequentialCloser(player_id=0)
    bot_b = LeadThenCover(player_id=1)
    result = run_matchup(bot_a, bot_b, 100, "S1", "S2")
    assert result.wins_a + result.wins_b == result.total_games
    assert result.total_games == 100


def test_matchup_win_rates_sum():
    """win_rate_a + win_rate_b should approximately equal 1.0."""
    bot_a = SequentialCloser(player_id=0)
    bot_b = SequentialCloser(player_id=1)
    result = run_matchup(bot_a, bot_b, 200, "S1a", "S1b")
    assert abs(result.win_rate_a + result.win_rate_b - 1.0) < 0.01


def test_matchup_has_valid_scores():
    """Average scores should be non-negative."""
    bot_a = LeadThenCover(player_id=0)
    bot_b = ChaseAndCover(player_id=1)
    result = run_matchup(bot_a, bot_b, 100, "S2", "S10")
    assert result.avg_score_a >= 0
    assert result.avg_score_b >= 0
    assert result.avg_game_length > 0


def test_matchup_s1_vs_s1_roughly_equal():
    """Two identical S1 bots should have roughly equal win rates (within 15%)."""
    bot_a = SequentialCloser(player_id=0)
    bot_b = SequentialCloser(player_id=1)
    result = run_matchup(bot_a, bot_b, 500, "S1a", "S1b")
    assert abs(result.win_rate_a - 0.5) < 0.15, (
        f"S1 vs S1 win rate too skewed: {result.win_rate_a:.3f}"
    )


# ── run_round_robin tests ────────────────────────────────────────────


def test_round_robin_all_pairs():
    """Round robin with 3 players should produce 3 matchups."""
    players = {
        "S1": SequentialCloser(player_id=0),
        "S2": LeadThenCover(player_id=0),
        "S10": ChaseAndCover(player_id=0),
    }
    result = run_round_robin(players, 100)
    assert len(result.matchups) == 3  # C(3,2) = 3
    assert len(result.rankings) == 3
    # Check win_rate_matrix has all pairs
    for name in players:
        assert name in result.win_rate_matrix


# ── Dataclass tests ──────────────────────────────────────────────────


def test_principle_result_pass():
    """PrincipleResult correctly reports passed=True."""
    pr = PrincipleResult(
        name="Test", passed=True, details="All good", evidence={"x": 1}
    )
    assert pr.passed is True
    assert pr.name == "Test"


def test_principle_result_fail():
    """PrincipleResult correctly reports passed=False."""
    pr = PrincipleResult(
        name="Test", passed=False, details="Failed check", evidence={}
    )
    assert pr.passed is False


def test_validation_report_structure():
    """ValidationReport holds all 4 principles and matchup data."""
    principles = [
        PrincipleResult(name=f"P{i}", passed=i % 2 == 0, details="", evidence={})
        for i in range(4)
    ]
    matchups = [
        MatchupResult("A", "B", 5, 5, 10, 0.5, 0.5, 10.0, 10.0, 20.0)
    ]
    report = ValidationReport(
        principles=principles, matchups=matchups, summary="Test summary"
    )
    assert len(report.principles) == 4
    assert len(report.matchups) == 1
    assert report.summary == "Test summary"
