"""Unit tests for Frongello strategy bots."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from config import BULL, DARTS_PER_TURN, MARKS_TO_CLOSE, TARGETS
from game import DartsCricketGame
from strategies import (
    ChaseAndCover,
    FrongelloStrategy,
    LeadPlusExtraDarts,
    LeadThenCover,
    SequentialCloser,
    StrategyBot,
    STRATEGY_CLASSES,
)


def _close_target(game: DartsCricketGame, player: int, target: int) -> None:
    """Helper: directly set a target to closed for a player."""
    idx = TARGETS.index(target)
    game.marks[player][idx] = MARKS_TO_CLOSE


# ── Return format tests ──────────────────────────────────────────────


def test_play_turn_return_format():
    """All bots return dict with required keys."""
    required_keys = {"throws", "total_points", "triple_count", "points_scored"}
    for BotClass in [SequentialCloser, LeadThenCover, LeadPlusExtraDarts, ChaseAndCover]:
        game = DartsCricketGame()
        bot = BotClass(player_id=0)
        result = bot.play_turn(game)
        assert required_keys.issubset(result.keys()), (
            f"{BotClass.__name__} missing keys: {required_keys - result.keys()}"
        )
        assert isinstance(result["throws"], list)
        assert isinstance(result["total_points"], int)
        assert isinstance(result["triple_count"], int)
        assert isinstance(result["points_scored"], bool)


def test_throw_entry_format():
    """Each throw entry has action, target, hit_type, result keys."""
    game = DartsCricketGame()
    bot = SequentialCloser(player_id=0)
    result = bot.play_turn(game)
    for throw in result["throws"]:
        assert "action" in throw
        assert "target" in throw
        assert "hit_type" in throw
        assert "result" in throw


def test_play_turn_throws_count():
    """A turn should produce at most DARTS_PER_TURN throws."""
    for BotClass in [SequentialCloser, LeadThenCover, LeadPlusExtraDarts, ChaseAndCover]:
        game = DartsCricketGame()
        bot = BotClass(player_id=0)
        result = bot.play_turn(game)
        assert len(result["throws"]) <= DARTS_PER_TURN


# ── SequentialCloser (S1) ────────────────────────────────────────────


def test_sequential_closer_targets_20_first():
    """On empty board, S1 should target 20 (highest value in descending order)."""
    game = DartsCricketGame()
    bot = SequentialCloser(player_id=0)
    target, hit_type = bot.choose_throw(game)
    assert target == 20
    assert hit_type == "triple"


def test_sequential_closer_moves_to_next():
    """After closing 20, S1 should target 19."""
    game = DartsCricketGame()
    bot = SequentialCloser(player_id=0)
    _close_target(game, 0, 20)
    target, _ = bot.choose_throw(game)
    assert target == 19


def test_sequential_closer_scores_when_behind():
    """All closed, behind on score: S1 scores on highest scoreable target."""
    game = DartsCricketGame()
    bot = SequentialCloser(player_id=0)
    for t in TARGETS:
        _close_target(game, 0, t)
    game.scores[1] = 50
    game.scores[0] = 0
    target, _ = bot.choose_throw(game)
    opp_unclosed = [TARGETS[i] for i in range(len(TARGETS))
                    if game.marks[1][i] < MARKS_TO_CLOSE]
    assert target in opp_unclosed


def test_s1_never_scores_mid_game():
    """S1 closes in order even when it has scoreable targets and is behind."""
    game = DartsCricketGame()
    bot = SequentialCloser(player_id=0)
    _close_target(game, 0, 20)  # we closed 20, could score on it
    game.scores[0] = 0
    game.scores[1] = 50  # behind
    target, _ = bot.choose_throw(game)
    assert target == 19  # S1 always closes next, never scores mid-game


# ── LeadThenCover (S2) ───────────────────────────────────────────────


def test_lead_then_cover_scores_when_behind():
    """S2 should score on scoreable targets when behind."""
    game = DartsCricketGame()
    bot = LeadThenCover(player_id=0)
    _close_target(game, 0, 20)
    game.scores[0] = 0
    game.scores[1] = 30
    target, _ = bot.choose_throw(game)
    assert target == 20  # Score on 20 (closed by us, open for opponent)


def test_lead_then_cover_covers_when_ahead():
    """S2 ahead: covers highest target opponent hasn't closed (defensive)."""
    game = DartsCricketGame()
    bot = LeadThenCover(player_id=0)
    _close_target(game, 0, 20)
    game.scores[0] = 50
    game.scores[1] = 0
    # Opponent hasn't closed anything → cover highest opp-unclosed we haven't closed
    # We closed 20, opp hasn't closed 20. But WE already closed 20.
    # So cover = highest opp-unclosed that we haven't closed = 19
    target, _ = bot.choose_throw(game)
    assert target == 19


def test_lead_then_cover_covers_when_ahead_by_1():
    """S2 (lead_multiplier=0) switches to covering as soon as ahead by any amount."""
    game = DartsCricketGame()
    bot = LeadThenCover(player_id=0)
    _close_target(game, 0, 20)
    game.scores[0] = 1
    game.scores[1] = 0
    target, _ = bot.choose_throw(game)
    assert target == 19  # Cover: highest opp-unclosed we haven't closed


def test_s2_covers_opponent_unclosed():
    """S2 in cover mode targets highest number the opponent hasn't closed."""
    game = DartsCricketGame()
    bot = LeadThenCover(player_id=0)
    # We closed 20, 19. Opponent closed 20 only.
    _close_target(game, 0, 20)
    _close_target(game, 0, 19)
    _close_target(game, 1, 20)
    game.scores[0] = 50  # ahead → cover mode
    game.scores[1] = 0
    target, _ = bot.choose_throw(game)
    # Opponent hasn't closed: 19, 18, 17, 16, 15, Bull
    # We haven't closed: 18, 17, 16, 15, Bull
    # Highest opp-unclosed that we haven't closed = 19... but we closed 19!
    # Next = 18
    assert target == 18


def test_s2_scores_on_closed_target():
    """S2 behind with scoreable target: aims at it for points."""
    game = DartsCricketGame()
    bot = LeadThenCover(player_id=0)
    _close_target(game, 0, 20)
    _close_target(game, 0, 19)
    # Opponent closed 20 but not 19 → 19 is scoreable
    _close_target(game, 1, 20)
    game.scores[0] = 0
    game.scores[1] = 30  # behind
    target, _ = bot.choose_throw(game)
    assert target == 19  # Score on 19 (highest scoreable)


def test_s2_closes_self_when_nothing_scoreable():
    """S2 behind but nothing scoreable: closes highest unclosed by self."""
    game = DartsCricketGame()
    bot = LeadThenCover(player_id=0)
    game.scores[0] = 0
    game.scores[1] = 30  # behind
    # Nothing closed by either player → nothing scoreable → close highest unclosed
    target, _ = bot.choose_throw(game)
    assert target == 20


# ── S3 threshold tests ───────────────────────────────────────────────


def test_s3_scores_when_below_threshold():
    """S3 (lead×3) should score when lead is below 3× highest unblocked."""
    game = DartsCricketGame()
    bot = STRATEGY_CLASSES["S3"](player_id=0)
    _close_target(game, 0, 20)  # can score on 20
    # threshold = 3 × 20 = 60; lead = 10 - 0 = 10 ≤ 60 → score
    game.scores[0] = 10
    game.scores[1] = 0
    target, _ = bot.choose_throw(game)
    assert target == 20  # score on 20 (scoreable)


def test_s3_covers_when_above_threshold():
    """S3 (lead×3) should cover when lead exceeds 3× highest unblocked."""
    game = DartsCricketGame()
    bot = STRATEGY_CLASSES["S3"](player_id=0)
    _close_target(game, 0, 20)  # can score on 20
    # threshold = 3 × 20 = 60; lead = 61 - 0 = 61 > 60 → cover
    game.scores[0] = 61
    game.scores[1] = 0
    target, _ = bot.choose_throw(game)
    # Cover: highest opp-unclosed we haven't closed = 19
    assert target == 19


# ── ChaseAndCover (S10) ──────────────────────────────────────────────


def test_chase_and_cover_prioritizes_opponent_closed():
    """S10 should chase opponent's closed targets before anything else."""
    game = DartsCricketGame()
    bot = ChaseAndCover(player_id=0)
    _close_target(game, 1, 18)
    target, _ = bot.choose_throw(game)
    assert target == 18


def test_chase_and_cover_falls_back_to_s2():
    """S10 falls back to S2 logic when no opponent-closed targets to chase."""
    game = DartsCricketGame()
    bot = ChaseAndCover(player_id=0)
    target, _ = bot.choose_throw(game)
    assert target == 20  # No chase targets → S2 fallback → close highest unclosed


def test_chase_and_cover_chases_highest_first():
    """S10 chases the highest-value target the opponent closed."""
    game = DartsCricketGame()
    bot = ChaseAndCover(player_id=0)
    _close_target(game, 1, 15)
    _close_target(game, 1, 19)
    target, _ = bot.choose_throw(game)
    assert target == 19


def test_chase_skips_already_closed_by_self():
    """S10 doesn't chase targets we've already closed."""
    game = DartsCricketGame()
    bot = ChaseAndCover(player_id=0)
    _close_target(game, 1, 20)
    _close_target(game, 0, 20)  # we already closed 20
    _close_target(game, 1, 18)
    target, _ = bot.choose_throw(game)
    assert target == 18  # Skip 20 (we closed it), chase 18


# ── LeadPlusExtraDarts (S6) ──────────────────────────────────────────


def test_lead_plus_extra_darts_scores_after_closing():
    """S6 should use extra darts to score after closing the main target."""
    game = DartsCricketGame()
    bot = LeadPlusExtraDarts(player_id=0)
    _close_target(game, 0, 20)
    _close_target(game, 0, 19)
    game.scores[1] = 20
    target1, _ = bot.choose_throw(game)
    target2, _ = bot.choose_throw(game)
    assert target2 in TARGETS


def test_s6_redirects_when_cant_close():
    """S6 redirects remaining darts to scoring if can't close turn target."""
    game = DartsCricketGame()
    bot = STRATEGY_CLASSES["S6"](player_id=0)
    _close_target(game, 0, 20)  # scoreable
    game.scores[0] = 0
    game.scores[1] = 50  # behind → score mode
    # Dart 1: score on 20 (scoreable). Turn target = 20.
    target1, _ = bot.choose_throw(game)
    assert target1 == 20
    # Dart 2: turn target (20) is already closed (marks_needed=0)
    # → redirect to scoring on highest scoreable
    target2, _ = bot.choose_throw(game)
    assert target2 == 20  # still scoreable


def test_s6_keeps_aiming_when_can_close():
    """S6 keeps aiming at turn target if it can still be closed with remaining darts."""
    game = DartsCricketGame()
    bot = STRATEGY_CLASSES["S6"](player_id=0)
    game.marks[0][TARGETS.index(20)] = 1  # 1 mark, needs 2 more
    game.scores[0] = 0
    game.scores[1] = 0  # tied → score mode for S6 (threshold=0, lead=0)
    # Dart 1: nothing scoreable → close highest unclosed = 20. Turn target = 20.
    target1, _ = bot.choose_throw(game)
    assert target1 == 20
    # Dart 2: needs 2 marks, 2 darts left → can close → keep aiming at 20
    target2, _ = bot.choose_throw(game)
    assert target2 == 20


# ── S14 combined (chase + threshold + extra darts) ────────────────────


def test_s14_chase_takes_priority():
    """S14 should chase opponent-closed targets before anything else."""
    game = DartsCricketGame()
    bot = STRATEGY_CLASSES["S14"](player_id=0)
    _close_target(game, 1, 18)
    _close_target(game, 0, 20)
    game.scores[0] = 0
    game.scores[1] = 30
    target, _ = bot.choose_throw(game)
    assert target == 18  # chase takes priority


# ── All 17 strategies complete a full game ────────────────────────────


@pytest.mark.parametrize("name", list(STRATEGY_CLASSES.keys()))
def test_strategy_completes_game(name):
    """Each strategy can play a full game without errors."""
    BotClass = STRATEGY_CLASSES[name]
    game = DartsCricketGame()
    bot_a = BotClass(player_id=0)
    bot_b = SequentialCloser(player_id=1)
    bots = [bot_a, bot_b]
    turns = 0
    max_turns = 200
    while not game.game_over and turns < max_turns:
        current_bot = bots[game.current_player]
        current_bot.play_turn(game)
        turns += 1
    assert game.game_over, f"{name} game didn't finish in {max_turns} turns"
    assert game.winner in (0, 1)


@pytest.mark.parametrize("name", list(STRATEGY_CLASSES.keys()))
def test_strategy_targets_20_first(name):
    """Each strategy starts by targeting 20 on an empty board."""
    BotClass = STRATEGY_CLASSES[name]
    game = DartsCricketGame()
    bot = BotClass(player_id=0)
    target, hit_type = bot.choose_throw(game)
    assert target == 20
    assert hit_type == "triple"


# ── Full game completion (legacy test) ────────────────────────────────


def test_all_bots_complete_full_game():
    """All bots can play a full game without errors."""
    for BotClass in [SequentialCloser, LeadThenCover, LeadPlusExtraDarts, ChaseAndCover]:
        game = DartsCricketGame()
        bot_a = BotClass(player_id=0)
        bot_b = SequentialCloser(player_id=1)
        bots = [bot_a, bot_b]
        turns = 0
        max_turns = 200
        while not game.game_over and turns < max_turns:
            current_bot = bots[game.current_player]
            current_bot.play_turn(game)
            turns += 1
        assert game.game_over, f"{BotClass.__name__} game didn't finish in {max_turns} turns"
        assert game.winner in (0, 1)


# ── Registry ──────────────────────────────────────────────────────────


def test_registry_has_17_strategies():
    """STRATEGY_CLASSES should have exactly 17 entries (S1-S17)."""
    assert len(STRATEGY_CLASSES) == 17
    for i in range(1, 18):
        assert f"S{i}" in STRATEGY_CLASSES


def test_backward_compatible_aliases():
    """Old class names still work."""
    assert SequentialCloser is STRATEGY_CLASSES["S1"]
    assert LeadThenCover is STRATEGY_CLASSES["S2"]
    assert LeadPlusExtraDarts is STRATEGY_CLASSES["S6"]
    assert ChaseAndCover is STRATEGY_CLASSES["S10"]


def test_all_strategies_are_strategy_bot_subclasses():
    """Every strategy class should be a subclass of StrategyBot."""
    for name, cls in STRATEGY_CLASSES.items():
        bot = cls(player_id=0)
        assert isinstance(bot, StrategyBot), f"{name} is not a StrategyBot"
        assert isinstance(bot, FrongelloStrategy), f"{name} is not a FrongelloStrategy"


# ══════════════════════════════════════════════════════════════════════
# Scenario-based tests: verify decisions on specific board states
# ══════════════════════════════════════════════════════════════════════


def _make_game(**kwargs) -> DartsCricketGame:
    """Create a game with specific board state.

    Keyword args:
        p0_closed: list of target values P0 has closed
        p1_closed: list of target values P1 has closed
        p0_marks: dict of {target_value: marks_count} for partial progress
        p1_marks: dict of {target_value: marks_count} for partial progress
        scores: (p0_score, p1_score)
        current_player: 0 or 1
        darts_remaining: int
    """
    game = DartsCricketGame()
    for t in kwargs.get("p0_closed", []):
        _close_target(game, 0, t)
    for t in kwargs.get("p1_closed", []):
        _close_target(game, 1, t)
    for t, m in kwargs.get("p0_marks", {}).items():
        game.marks[0][TARGETS.index(t)] = m
    for t, m in kwargs.get("p1_marks", {}).items():
        game.marks[1][TARGETS.index(t)] = m
    s0, s1 = kwargs.get("scores", (0, 0))
    game.scores[0] = s0
    game.scores[1] = s1
    game.current_player = kwargs.get("current_player", 0)
    game.darts_remaining = kwargs.get("darts_remaining", DARTS_PER_TURN)
    return game


# ── S1 scenarios ──────────────────────────────────────────────────────


class TestS1Scenarios:
    """S1: Always close 20→19→...→Bull in order. Score only after all closed."""

    def test_empty_board(self):
        """S1 targets 20 on empty board."""
        game = _make_game()
        bot = SequentialCloser(player_id=0)
        assert bot.choose_throw(game) == (20, "triple")

    def test_ignores_scoring_opportunity(self):
        """S1 does NOT score even when it has a scoreable target and is behind."""
        game = _make_game(p0_closed=[20], scores=(0, 100))
        bot = SequentialCloser(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 19  # closes next, ignores scoring on 20

    def test_closes_in_strict_order(self):
        """S1 closes 20→19→18 even if lower numbers are 'easier'."""
        game = _make_game(p0_closed=[20, 19], p0_marks={17: 2})
        bot = SequentialCloser(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 18  # not 17 despite 17 being 1 mark from closing

    def test_scores_only_when_all_closed_and_behind(self):
        """S1 scores on highest scoreable only after closing all 7 targets."""
        game = _make_game(
            p0_closed=[20, 19, 18, 17, 16, 15, BULL],
            p1_closed=[20, 19],
            scores=(0, 50),
        )
        bot = SequentialCloser(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 18  # highest scoreable (we closed, opp hasn't)

    def test_bull_when_all_closed_and_ahead(self):
        """S1 throws Bull when all closed and winning."""
        game = _make_game(
            p0_closed=[20, 19, 18, 17, 16, 15, BULL],
            p1_closed=[20, 19, 18, 17, 16, 15, BULL],
            scores=(50, 0),
        )
        bot = SequentialCloser(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == BULL


# ── S2 scenarios ──────────────────────────────────────────────────────


class TestS2Scenarios:
    """S2: Score until ahead by any amount, then cover (opponent-unclosed)."""

    def test_empty_board_tied(self):
        """Tied with nothing scoreable → close highest unclosed (20)."""
        game = _make_game()
        bot = LeadThenCover(player_id=0)
        assert bot.choose_throw(game) == (20, "triple")

    def test_scores_when_behind_with_scoreable(self):
        """Behind with scoreable → aim at highest scoreable."""
        game = _make_game(p0_closed=[20, 19], p1_closed=[20], scores=(0, 30))
        bot = LeadThenCover(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 19  # 19 is scoreable (we closed, opp hasn't)

    def test_scores_when_tied_with_scoreable(self):
        """Tied (lead=0 ≤ threshold=0) → still in score mode."""
        game = _make_game(p0_closed=[20], scores=(0, 0))
        bot = LeadThenCover(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 20  # score on 20 (scoreable, tied → score mode)

    def test_covers_opponent_unclosed_when_ahead(self):
        """Ahead → cover = close highest target opp hasn't closed."""
        game = _make_game(
            p0_closed=[20, 19], p1_closed=[20, 18], scores=(30, 0)
        )
        bot = LeadThenCover(player_id=0)
        target, _ = bot.choose_throw(game)
        # Opp unclosed: 19, 17, 16, 15, Bull. We haven't closed: 18, 17, 16, 15, Bull.
        # Highest opp-unclosed we haven't closed = 17 (we already closed 19)
        assert target == 17

    def test_covers_goes_back_to_defend(self):
        """Cover mode: go back and close a number the opp left open that we skipped."""
        game = _make_game(
            p0_closed=[20, 19, 18], p1_closed=[20, 19, 18, 17, 16],
            scores=(80, 0),
        )
        bot = LeadThenCover(player_id=0)
        target, _ = bot.choose_throw(game)
        # Opp unclosed: 15, Bull. We haven't closed: 17, 16, 15, Bull.
        # Highest opp-unclosed we haven't closed = 15
        assert target == 15

    def test_close_self_when_nothing_to_cover(self):
        """When we've closed everything the opp hasn't, close our own remaining."""
        game = _make_game(
            p0_closed=[20, 19, 18, 17, 16, 15], p1_closed=[20],
            scores=(200, 0),
        )
        bot = LeadThenCover(player_id=0)
        target, _ = bot.choose_throw(game)
        # Opp unclosed: 19, 18, 17, 16, 15, Bull. We've closed all of those except Bull.
        # Highest opp-unclosed we haven't closed = Bull
        assert target == BULL


# ── S3 scenarios ──────────────────────────────────────────────────────


class TestS3Scenarios:
    """S3: Score until lead > 3× highest opponent-unblocked value."""

    def test_scores_at_threshold_boundary(self):
        """Lead exactly at threshold → still in score mode (lead ≤ threshold)."""
        game = _make_game(p0_closed=[20], scores=(60, 0))
        # threshold = 3 × 20 = 60. lead = 60 ≤ 60 → score
        bot = STRATEGY_CLASSES["S3"](player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 20  # score on 20

    def test_covers_above_threshold(self):
        """Lead above threshold → cover mode."""
        game = _make_game(p0_closed=[20], scores=(61, 0))
        # threshold = 3 × 20 = 60. lead = 61 > 60 → cover
        bot = STRATEGY_CLASSES["S3"](player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 19  # cover: highest opp-unclosed we haven't closed

    def test_threshold_uses_opponent_unclosed(self):
        """Threshold is based on highest number the OPPONENT hasn't closed."""
        game = _make_game(
            p0_closed=[20, 19], p1_closed=[20, 19], scores=(55, 0)
        )
        # Opp has closed 20 and 19. Highest opp-unclosed = 18.
        # threshold = 3 × 18 = 54. lead = 55 > 54 → cover
        bot = STRATEGY_CLASSES["S3"](player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 18  # cover mode


# ── S6 scenarios (extra darts) ────────────────────────────────────────


class TestS6Scenarios:
    """S6: Same as S2 + redirect extra darts to scoring when can't close.

    Frongello: "if at any point the next open number can not be completely
    covered assuming single hits with darts remaining in hand, the strategy
    aims these 'extra darts' at the highest number the strategy has closed
    and can point on."

    The redirect only fires when the core S2 decision would be to CLOSE
    a target (target is unclosed by self) and that target can't be finished
    with remaining darts assuming singles.
    """

    def test_no_redirect_when_scoring(self):
        """Core decision = score on closed target → no redirect, just score."""
        game = _make_game(p0_closed=[20], scores=(0, 30), darts_remaining=1)
        bot = STRATEGY_CLASSES["S6"](player_id=0)
        target, _ = bot.choose_throw(game)
        # Behind → score mode → 20 is scoreable → aim at 20 (no redirect, it's scoring)
        assert target == 20

    def test_redirect_when_closing_target_cant_finish(self):
        """Core decision = close 19 but only 1 dart left (need 3) → redirect to scoring."""
        game = _make_game(
            p0_closed=[20],  # 20 is scoreable (we closed, opp hasn't)
            scores=(0, 0),   # tied → S6 score mode (threshold=0, lead=0)
            darts_remaining=1,  # only 1 dart
        )
        # Core S2 decision when tied: nothing to score on wait — 20 IS scoreable,
        # so S2 would score on 20.  Need a case where core says "close".
        # Let's set it up so we're ahead → cover mode → close highest opp-unclosed
        game2 = _make_game(
            p0_closed=[20],   # closed 20
            scores=(10, 0),   # ahead by 10 → cover mode
            darts_remaining=1,
        )
        bot = STRATEGY_CLASSES["S6"](player_id=0)
        target, _ = bot.choose_throw(game2)
        # Cover: close highest opp-unclosed we haven't closed = 19.
        # But 19 needs 3 marks and only 1 dart → redirect!
        # Scoreable = [20] (we closed, opp hasn't) → redirect to 20
        assert target == 20

    def test_no_redirect_when_can_finish_closing(self):
        """Core decision = close 20 with 2 marks needed, 2 darts left → no redirect."""
        game = _make_game(
            p0_marks={20: 1},  # 1 mark on 20, needs 2 more
            scores=(10, 0),    # ahead → cover mode
            darts_remaining=2,
        )
        bot = STRATEGY_CLASSES["S6"](player_id=0)
        target, _ = bot.choose_throw(game)
        # Cover: 20 is highest opp-unclosed we haven't closed. Need 2, have 2 → can close
        assert target == 20

    def test_redirect_fallback_to_opponent_unclosed(self):
        """Redirect fires but nothing scoreable → close highest opp-unclosed."""
        game = _make_game(
            p0_closed=[20], p1_closed=[20],  # 20 dead, not scoreable
            scores=(10, 0),   # ahead → cover mode
            darts_remaining=1,
        )
        bot = STRATEGY_CLASSES["S6"](player_id=0)
        target, _ = bot.choose_throw(game)
        # Cover: 19 highest opp-unclosed we haven't closed. Need 3, have 1 → redirect!
        # But nothing scoreable (20 is dead).
        # Fallback: close highest opp-unclosed not closed by either = 19
        assert target == 19

    def test_s6_scores_same_as_s2_when_behind(self):
        """When behind with scoreable targets, S6 behaves like S2 (scores)."""
        game = _make_game(p0_closed=[20], scores=(0, 30))
        s2 = LeadThenCover(player_id=0)
        s6 = STRATEGY_CLASSES["S6"](player_id=0)
        t2, _ = s2.choose_throw(game)
        t6, _ = s6.choose_throw(game)
        assert t2 == t6 == 20  # both score on 20


# ── S10 scenarios (chase) ─────────────────────────────────────────────


class TestS10Scenarios:
    """S10: Chase opponent's closed targets first, then S2 logic."""

    def test_chase_highest_opponent_closed(self):
        """Chase the highest-value target the opponent closed that we haven't."""
        game = _make_game(p1_closed=[20, 17])
        bot = ChaseAndCover(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 20

    def test_chase_skips_both_closed(self):
        """Don't chase a target both players have closed."""
        game = _make_game(p0_closed=[20], p1_closed=[20, 18])
        bot = ChaseAndCover(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 18  # 20 already chased, chase 18

    def test_no_chase_targets_scores_when_behind(self):
        """No chase targets, behind → S2 score logic."""
        game = _make_game(p0_closed=[20], scores=(0, 30))
        bot = ChaseAndCover(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 20  # score on 20 (highest scoreable)

    def test_no_chase_targets_covers_when_ahead(self):
        """No chase targets, ahead → S2 cover logic."""
        game = _make_game(p0_closed=[20], scores=(50, 0))
        bot = ChaseAndCover(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 19  # cover: highest opp-unclosed we haven't closed

    def test_chase_takes_priority_over_scoring(self):
        """Chase overrides scoring even when behind with scoreables."""
        game = _make_game(
            p0_closed=[20], p1_closed=[18], scores=(0, 100)
        )
        bot = ChaseAndCover(player_id=0)
        target, _ = bot.choose_throw(game)
        assert target == 18  # chase 18 even though could score on 20


# ── S14 scenarios (chase + extra darts) ───────────────────────────────


class TestS14Scenarios:
    """S14: Chase first, then S2 with extra-darts redirect."""

    def test_chase_before_extra_darts(self):
        """Chase fires even when extra-darts redirect would apply."""
        game = _make_game(
            p0_closed=[20], p1_closed=[17],
            scores=(10, 0),  # ahead → cover mode
            darts_remaining=1,
        )
        bot = STRATEGY_CLASSES["S14"](player_id=0)
        target, _ = bot.choose_throw(game)
        # Chase 17 takes priority over everything (even though cover would
        # pick 19 and redirect would fire)
        assert target == 17

    def test_extra_darts_after_no_chase(self):
        """When no chase targets, extra-darts redirect works on closing targets."""
        game = _make_game(
            p0_closed=[20],  # 20 is scoreable
            scores=(10, 0),  # ahead → cover mode
            darts_remaining=1,
        )
        bot = STRATEGY_CLASSES["S14"](player_id=0)
        target, _ = bot.choose_throw(game)
        # No chase targets. Cover: close 19 (highest opp-unclosed we haven't closed).
        # But 19 needs 3 marks and only 1 dart → redirect to scoring on 20.
        assert target == 20


# ── Cross-strategy comparison scenarios ───────────────────────────────


class TestStrategyCrossComparison:
    """Verify different strategies choose different actions in the same state."""

    def test_s1_vs_s2_behind_with_scoreable(self):
        """S1 closes next; S2 scores. Same board, different decisions."""
        game = _make_game(p0_closed=[20], scores=(0, 30))

        s1 = SequentialCloser(player_id=0)
        s2 = LeadThenCover(player_id=0)

        t1, _ = s1.choose_throw(game)
        t2, _ = s2.choose_throw(game)

        assert t1 == 19  # S1: close next in order
        assert t2 == 20  # S2: score on 20

    def test_s2_vs_s10_opponent_closed_target(self):
        """S2 ignores opponent closures; S10 chases them."""
        game = _make_game(
            p0_closed=[20], p1_closed=[18], scores=(0, 0)
        )

        s2 = LeadThenCover(player_id=0)
        s10 = ChaseAndCover(player_id=0)

        t2, _ = s2.choose_throw(game)
        t10, _ = s10.choose_throw(game)

        assert t2 == 20  # S2: score on 20 (tied → score mode)
        assert t10 == 18  # S10: chase 18

    def test_s2_vs_s3_different_thresholds(self):
        """S2 covers at lead=1; S3 keeps scoring until 3× threshold."""
        game = _make_game(p0_closed=[20], scores=(20, 0))
        # S2: threshold=0, lead=20>0 → cover
        # S3: threshold=3×20=60, lead=20≤60 → score

        s2 = LeadThenCover(player_id=0)
        s3 = STRATEGY_CLASSES["S3"](player_id=0)

        t2, _ = s2.choose_throw(game)
        t3, _ = s3.choose_throw(game)

        assert t2 == 19  # S2: cover (ahead)
        assert t3 == 20  # S3: still scoring (below threshold)
