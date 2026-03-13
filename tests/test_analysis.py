"""Unit tests for analysis.py — Q-table introspection and visualization."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import AIPlayer
from analysis import (
    analyze_chase_behavior,
    analyze_opening_preferences,
    analyze_scoring_vs_closing,
    analyze_state_policy,
    build_state,
    generate_strategy_heatmap,
    generate_validation_plots,
)
from config import MARKS_TO_CLOSE, NUM_ACTIONS


def test_build_state_correct_format():
    """build_state produces the expected encoded tuple structure."""
    state = build_state(my_score=10, opp_score=5,
                        my_marks=(1, 2, 3, 0, 0, 0, 0),
                        opp_marks=(0, 0, 0, 1, 2, 3, 0))
    # 10-5=5, which is "ahead" bucket (1)
    # my: 1->inprog, 2->inprog, 3->closed(opp open)=2, 0->untouched, 0, 0, 0
    # opp: 0->untouched, 0, 0, 1->inprog, 2->inprog, 3->closed(my open)=2, 0
    assert state == (1, ((1, 1, 2, 0, 0, 0, 0), (0, 0, 0, 1, 1, 2, 0)))


def test_build_state_encodes_marks():
    """Marks are encoded as strategic categories (0-3)."""
    state = build_state(my_marks=(5, 4, 3, 2, 1, 0, 6),
                        opp_marks=(0, 0, 0, 0, 0, 0, 10))
    score_bucket, (my_m, opp_m) = state
    assert all(0 <= m <= 3 for m in my_m)
    assert all(0 <= m <= 3 for m in opp_m)
    # my: closed+opp_open=2, closed+opp_open=2, closed+opp_open=2, inprog=1, inprog=1, untouched=0, both_closed=3
    assert my_m == (2, 2, 2, 1, 1, 0, 3)
    # opp: all untouched=0 except Bull which is both_closed=3
    assert opp_m == (0, 0, 0, 0, 0, 0, 3)


def test_analyze_opening_preferences_returns_all_actions():
    """Opening preferences should list all NUM_ACTIONS actions, sorted by Q-value."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    starting = build_state()
    # Set known Q-values
    for a in range(NUM_ACTIONS):
        agent.q_table[starting][a] = float(a)

    prefs = analyze_opening_preferences(agent)
    assert len(prefs) == NUM_ACTIONS
    # Verify descending Q-value order
    q_values = [qv for _, _, qv in prefs]
    assert q_values == sorted(q_values, reverse=True)


def test_analyze_state_policy_correct_ranking():
    """analyze_state_policy returns correct ranking for known Q-values."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    state = build_state(my_score=20, opp_score=10)
    # Set specific Q-values
    agent.q_table[state][3] = 50.0
    agent.q_table[state][7] = 30.0
    agent.q_table[state][0] = 10.0

    policy = analyze_state_policy(agent, state)
    assert policy[0][0] == 3  # action 3 has highest Q
    assert policy[0][2] == 50.0
    assert policy[1][0] == 7  # action 7 second
    assert policy[2][0] == 0  # action 0 third


def test_analyze_chase_behavior_structure():
    """Chase behavior returns dict with expected keys."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    result = analyze_chase_behavior(agent)
    assert isinstance(result, dict)
    assert len(result) == 7  # One entry per target
    # Check that each entry has the expected subkeys
    for key, val in result.items():
        assert "top_action" in val
        assert "q_value" in val
        assert "chases_target" in val


def test_analyze_scoring_vs_closing_structure():
    """Scoring vs closing returns dict with 4 scenario keys."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    result = analyze_scoring_vs_closing(agent)
    assert isinstance(result, dict)
    expected_keys = {"behind_40", "tied", "ahead_40", "ahead_80"}
    assert set(result.keys()) == expected_keys
    for key, val in result.items():
        assert "score_diff" in val
        assert "top_action" in val
        assert "strategy" in val
        assert val["strategy"] in ("scoring", "closing")


def test_generate_strategy_heatmap_creates_png(tmp_path):
    """Heatmap generation produces a PNG file."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    output_path = str(tmp_path / "heatmap.png")
    generate_strategy_heatmap(agent, output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_generate_validation_plots_creates_pngs(tmp_path):
    """Validation plot generation produces PNG files from a mock report."""

    class MockReport:
        round_robin_results = {
            "AgentA": {"AgentA": 0.5, "AgentB": 0.6},
            "AgentB": {"AgentA": 0.4, "AgentB": 0.5},
        }
        principle_results = {
            "principle_1": True,
            "principle_2": False,
            "principle_3": True,
        }
        opening_preferences = [
            (0, "20-triple", 45.0),
            (1, "20-double", 30.0),
            (2, "20-single", 15.0),
        ]

    output_dir = str(tmp_path / "plots")
    generate_validation_plots(MockReport(), output_dir)

    assert os.path.exists(os.path.join(output_dir, "round_robin_matrix.png"))
    assert os.path.exists(os.path.join(output_dir, "principle_results.png"))
    assert os.path.exists(os.path.join(output_dir, "opening_preferences.png"))
