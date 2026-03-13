"""Unit tests for advisor.py — throw recommendation engine."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import AIPlayer
from advisor import (
    ThrowRecommendation,
    TurnAdvice,
    explain_recommendation,
    get_advice,
    load_advisor,
    state_from_board,
)
from config import NUM_ACTIONS
from game import DartsCricketGame


def test_state_from_board_zeros():
    """All-zero board matches expected starting state (bucketed)."""
    state = state_from_board(0, 0, [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])
    # Score bucket 0 = tied, marks all zeros
    expected = (0, ((0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)))
    assert state == expected


def test_state_from_board_encodes_marks():
    """Marks are encoded as strategic categories. Score is bucketed."""
    # my_marks:  [5, 4, 3, 2, 1, 0, 6] -> closed/closed/closed/inprog/inprog/untouched/closed
    # opp_marks: [0, 0, 0, 0, 0, 0, 10] -> untouched/untouched/untouched/untouched/untouched/untouched/closed
    state = state_from_board(10, 20, [5, 4, 3, 2, 1, 0, 6], [0, 0, 0, 0, 0, 0, 10])
    score_bucket, (my_m, opp_m) = state
    # 10 - 20 = -10, which is "behind" (-1 bucket)
    assert score_bucket == -1
    # my_marks: 5->closed,opp open=2; 4->closed,opp open=2; 3->closed,opp open=2;
    #           2->in progress=1; 1->in progress=1; 0->untouched=0; 6->closed,opp closed=3
    assert my_m == (2, 2, 2, 1, 1, 0, 3)
    # opp_marks: first 6 are 0->untouched=0; last is 10->closed,my open=2 (since my_marks[6]=6>=3)
    # Wait, both are closed for Bull, so opp gets 3 (both closed)
    assert opp_m == (0, 0, 0, 0, 0, 0, 3)


def test_state_from_board_roundtrip():
    """Build state, verify tuple structure is hashable and correct shape."""
    state = state_from_board(15, 30, [1, 2, 3, 0, 1, 2, 0], [3, 3, 0, 0, 0, 0, 1])
    # Should be hashable (usable as dict key)
    d = {state: "test"}
    assert d[state] == "test"
    # Correct shape: (score_bucket, ((7 ints), (7 ints)))
    assert len(state) == 2
    assert isinstance(state[0], int)  # score_bucket is an int
    assert len(state[1]) == 2
    assert len(state[1][0]) == 7
    assert len(state[1][1]) == 7


def test_get_advice_returns_correct_count():
    """get_advice returns TurnAdvice with the requested number of recommendations."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    game = DartsCricketGame()
    # Set some Q-values so results aren't all zeros
    state = game.get_state(0)
    for a in range(NUM_ACTIONS):
        agent.q_table[state][a] = float(NUM_ACTIONS - a)

    advice = get_advice(agent, game, player_perspective=0, top_n=5)
    assert isinstance(advice, TurnAdvice)
    assert len(advice.recommendations) == 5
    assert isinstance(advice.strategic_context, str)
    assert len(advice.strategic_context) > 0


def test_get_advice_ranked_by_q_value():
    """Recommendations are sorted by Q-value descending."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    game = DartsCricketGame()
    state = game.get_state(0)
    for a in range(NUM_ACTIONS):
        agent.q_table[state][a] = float(NUM_ACTIONS - a)

    advice = get_advice(agent, game, player_perspective=0, top_n=NUM_ACTIONS)
    q_values = [rec.q_value for rec in advice.recommendations]
    assert q_values == sorted(q_values, reverse=True)
    # Rank should be sequential
    ranks = [rec.rank for rec in advice.recommendations]
    assert ranks == list(range(1, NUM_ACTIONS + 1))


def test_explain_recommendation_returns_string():
    """explain_recommendation returns a non-empty string."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    game = DartsCricketGame()
    state = game.get_state(0)

    explanation = explain_recommendation(0, 10.5, state, game, perspective=0)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_load_advisor_epsilon_zero(tmp_path):
    """load_advisor sets epsilon to 0 for pure exploitation."""
    agent = AIPlayer(player_id=0, epsilon=0.5)
    state = ((0, 0), ((0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)))
    agent.q_table[state][0] = 42.0
    path = str(tmp_path / "test_q.pkl")
    agent.save_q_table(path)

    loaded = load_advisor(path)
    assert loaded.epsilon == 0.0


def test_load_advisor_preserves_q_values(tmp_path):
    """load_advisor correctly loads Q-table values."""
    agent = AIPlayer(player_id=0)
    state = ((0, 0), ((0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)))
    agent.q_table[state][3] = 99.0
    agent.q_table[state][7] = -5.5
    path = str(tmp_path / "test_q.pkl")
    agent.save_q_table(path)

    loaded = load_advisor(path)
    assert loaded.q_table[state][3] == 99.0
    assert loaded.q_table[state][7] == -5.5
