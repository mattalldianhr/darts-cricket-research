"""Unit tests for DartsCricketGame — scoring, closure, and win detection."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game import DartsCricketGame


def test_initial_state():
    """Fresh game: scores=[0,0], all marks 0, player 0 first, 3 darts."""
    game = DartsCricketGame()
    assert game.scores == [0, 0]
    assert game.marks == [[0] * 7, [0] * 7]
    assert game.current_player == 0
    assert game.darts_remaining == 3
    assert game.game_over is False
    assert game.winner is None


def test_single_hit_adds_one_mark():
    """Single at 20 adds 1 mark."""
    game = DartsCricketGame()
    result = game.throw_dart(20, "single")
    target_idx = game.targets.index(20)
    assert game.marks[0][target_idx] == 1
    assert result["marks_added"] == 1


def test_double_hit_adds_two_marks():
    """Double at 20 adds 2 marks."""
    game = DartsCricketGame()
    result = game.throw_dart(20, "double")
    target_idx = game.targets.index(20)
    assert game.marks[0][target_idx] == 2
    assert result["marks_added"] == 2


def test_triple_hit_closes_number():
    """Triple at 20 -> 3 marks (closed)."""
    game = DartsCricketGame()
    result = game.throw_dart(20, "triple")
    target_idx = game.targets.index(20)
    assert game.marks[0][target_idx] == 3
    assert result["closed"] is True
    assert game.is_closed(0, target_idx)


def test_scoring_on_closed_number():
    """Close 20 (triple), then single 20 again = 20 points (opponent hasn't closed)."""
    game = DartsCricketGame()
    # Throw triple to close 20 (dart 1)
    game.throw_dart(20, "triple")
    # Throw single 20 for points (dart 2)
    result = game.throw_dart(20, "single")
    assert result["points_scored"] == 20
    assert game.scores[0] == 20


def test_no_scoring_if_both_closed():
    """Both close 20, no points on further hits."""
    game = DartsCricketGame()
    target_idx = game.targets.index(20)

    # Player 0 closes 20 (dart 1)
    game.throw_dart(20, "triple")
    # Player 0 uses remaining 2 darts on something else
    game.throw_dart(15, "single")
    game.throw_dart(15, "single")

    # Now player 1's turn — close 20
    assert game.current_player == 1
    game.throw_dart(20, "triple")

    # Player 1 tries to score on 20 — both closed, no points
    result = game.throw_dart(20, "single")
    assert result["points_scored"] == 0
    assert game.scores[1] == 0


def test_triple_on_open_closes_no_excess_score():
    """Triple from 0 marks -> closes but 0 excess -> 0 points."""
    game = DartsCricketGame()
    result = game.throw_dart(20, "triple")
    assert result["closed"] is True
    assert result["points_scored"] == 0


def test_scoring_excess_marks():
    """2 marks on 20, then triple -> 1 closes, 2 excess -> 40 points."""
    game = DartsCricketGame()
    # Add 2 marks first (dart 1)
    game.throw_dart(20, "double")
    # Now triple: from 2 marks, add 3 = 5 total. Excess above 3 = 2. 20 * 2 = 40
    result = game.throw_dart(20, "triple")
    assert result["closed"] is True
    assert result["points_scored"] == 40
    assert game.scores[0] == 40


def test_win_condition_all_closed_leading():
    """All 7 closed + score >= opponent -> wins."""
    game = DartsCricketGame()
    # Close all targets for player 0 using triples
    for target in game.targets:
        game.throw_dart(target, "triple")
        # Keep darts accounting right: after 3 darts player switches
        # so we need to handle turn switches
        if game.current_player != 0 and not game.game_over:
            # Burn player 1's turn
            game.throw_dart(15, "single")
            game.throw_dart(15, "single")
            game.throw_dart(15, "single")

    # Player 0 should have won (all closed, tied at 0 means >= opponent)
    assert game.game_over is True
    assert game.winner == 0


def test_no_win_if_not_all_closed():
    """6/7 closed -> no win."""
    game = DartsCricketGame()
    # Close only 6 targets for player 0
    targets_to_close = game.targets[:6]
    for target in targets_to_close:
        game.throw_dart(target, "triple")
        if game.current_player != 0 and not game.game_over:
            game.throw_dart(15, "single")
            game.throw_dart(15, "single")
            game.throw_dart(15, "single")

    assert game.game_over is False
    assert game.winner is None


def test_no_win_if_behind_on_score():
    """All closed but opponent has more points -> no win."""
    game = DartsCricketGame()

    # Player 0 uses dart 1 on 15
    game.throw_dart(15, "single")
    # Player 0 uses dart 2 on 16
    game.throw_dart(16, "single")
    # Player 0 uses dart 3 on 17
    game.throw_dart(17, "single")

    # Player 1's turn: close 20 and score on it
    assert game.current_player == 1
    game.throw_dart(20, "triple")  # close 20
    game.throw_dart(20, "triple")  # score 60 points
    game.throw_dart(20, "triple")  # score 60 more

    # Now give player 0 all 7 closed by directly setting marks
    # (simulating they closed everything, but behind on score)
    for i in range(len(game.targets)):
        game.marks[0][i] = 3

    # Player 0 turn again
    assert game.current_player == 0
    # Trigger win check with a throw
    game.throw_dart(15, "single")

    # Player 0 has all closed but 0 score vs player 1's 120
    # So player 0 can't win
    assert game.winner != 0
    # Player 1 hasn't closed everything, so shouldn't win either
    assert not game.all_closed(1)


def test_turn_switching():
    """After 3 darts, current_player switches to other."""
    game = DartsCricketGame()
    assert game.current_player == 0

    game.throw_dart(20, "single")
    game.throw_dart(20, "single")
    game.throw_dart(20, "single")

    assert game.current_player == 1
    assert game.darts_remaining == 3


def test_get_state_perspective():
    """P0 perspective has P0 data first, P1 perspective has P1 data first."""
    game = DartsCricketGame()
    game.throw_dart(20, "triple")  # P0 closes 20

    state_p0 = game.get_state(0)
    state_p1 = game.get_state(1)

    # P0 perspective: self closed 20, opponent hasn't -> category 2 (scoreable)
    assert state_p0[1][0][game.targets.index(20)] == 2  # self: closed, opp open
    assert state_p0[1][1][game.targets.index(20)] == 0  # opp: untouched

    # P1 perspective: reversed — opponent closed 20, we haven't
    assert state_p1[1][0][game.targets.index(20)] == 0  # self: untouched
    assert state_p1[1][1][game.targets.index(20)] == 2  # opp: closed, we open


def test_get_state_hashable():
    """Can use state as dict key."""
    game = DartsCricketGame()
    state = game.get_state(0)
    d = {state: "test"}
    assert d[state] == "test"


def test_reset_clears_everything():
    """After play + reset, all zeroed."""
    game = DartsCricketGame()
    game.throw_dart(20, "triple")
    game.throw_dart(20, "triple")
    assert game.scores[0] == 60  # scored after closing

    game.reset()
    assert game.scores == [0, 0]
    assert game.marks == [[0] * 7, [0] * 7]
    assert game.current_player == 0
    assert game.darts_remaining == 3
    assert game.game_over is False
    assert game.winner is None
    assert game.total_darts_thrown == 0


def test_bullseye_scoring():
    """Bull scores 25 per excess mark."""
    game = DartsCricketGame()
    bull_idx = game.targets.index(25)

    # Close bullseye: double (2 marks) + single (1 mark) = 3 marks
    game.throw_dart(25, "double")
    assert not game.is_closed(0, bull_idx)
    game.throw_dart(25, "single")
    assert game.is_closed(0, bull_idx)

    # Score on bullseye with single (dart 3) = 25 points
    result = game.throw_dart(25, "single")
    assert result["points_scored"] == 25
    assert game.scores[0] == 25


def test_marks_encoded_in_state():
    """get_state encodes marks as strategic categories."""
    game = DartsCricketGame()
    target_idx = game.targets.index(20)

    # Triple + triple = 6 actual marks (closed, opponent has 0)
    game.throw_dart(20, "triple")
    game.throw_dart(20, "triple")

    assert game.marks[0][target_idx] == 6

    state = game.get_state(0)
    # Closed with opponent open = category 2 (scoreable)
    assert state[1][0][target_idx] == 2
