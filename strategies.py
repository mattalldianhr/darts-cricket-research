"""Frongello strategy bots for darts cricket validation.

All 17 strategies (S1-S17) from Frongello's paper, implemented as a single
parameterized ``FrongelloStrategy`` class.  Each strategy is a combination of
three parameters:

    lead_multiplier  – score-lead threshold before switching from scoring to
                       covering.  ``None`` means *always* cover (S1).
    use_extra_darts  – within a turn, if the current closing target can't be
                       finished with remaining darts (assuming singles),
                       redirect those darts to scoring.
    use_chase        – first priority: close any target the opponent has closed
                       that we haven't (highest value first).

Decision flow per dart (priority order):

    1. Chase (S10-S17): close highest target opponent closed that we haven't.
    2. Score / Cover decision (S2-S17):
       - If lead ≤ threshold: SCORE — aim at highest scoreable target
         (we closed, opponent hasn't).  If nothing scoreable, close highest
         unclosed by self (to create a scoring opportunity).
       - If lead > threshold: COVER — close highest target NOT closed by
         opponent (defensive, blocks their future scoring).
    3. Extra darts (S6-S9, S14-S17): on darts 2-3, if the turn's closing
       target can't be finished this turn (marks needed > darts remaining,
       assuming singles), redirect to scoring on highest pointable target.
    4. S1: always close in order 20→Bull.  Only score after all 7 closed.

| Strategy | lead_multiplier | use_extra_darts | use_chase |
|----------|-----------------|-----------------|-----------|
| S1       | None            | False           | False     |
| S2       | 0               | False           | False     |
| S3       | 3               | False           | False     |
| S4       | 6               | False           | False     |
| S5       | 9               | False           | False     |
| S6       | 0               | True            | False     |
| S7       | 3               | True            | False     |
| S8       | 6               | True            | False     |
| S9       | 9               | True            | False     |
| S10      | 0               | False           | True      |
| S11      | 3               | False           | True      |
| S12      | 6               | False           | True      |
| S13      | 9               | False           | True      |
| S14      | 0               | True            | True      |
| S15      | 3               | True            | True      |
| S16      | 6               | True            | True      |
| S17      | 9               | True            | True      |
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from agent import AIPlayer
from config import (
    BULL,
    BULL_HIT_TYPES,
    DARTS_PER_TURN,
    HIT_MULTIPLIER,
    HIT_TYPES,
    MARKS_TO_CLOSE,
    NUM_ACTIONS,
    TARGET_NAMES,
    TARGETS,
)
from game import DartsCricketGame

logger = logging.getLogger(__name__)


class StrategyBot(ABC):
    """Base class for rule-based strategy bots."""

    def __init__(self, player_id: int) -> None:
        self.player_id: int = player_id

    @abstractmethod
    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        """Choose a (target, hit_type) for the next dart."""

    def play_turn(self, game: DartsCricketGame) -> Dict:
        """Play a full turn (up to 3 darts). Returns summary dict matching AIPlayer format."""
        throws: List[Dict] = []
        total_points: int = 0
        triple_count: int = 0

        for _ in range(DARTS_PER_TURN):
            if game.game_over:
                break
            if game.current_player != self.player_id:
                break

            target, hit_type = self.choose_throw(game)
            action = AIPlayer.target_hit_to_action(target, hit_type)
            result = game.throw_dart(target, hit_type)

            if result.get("actual_hit_type") == "triple":
                triple_count += 1
            total_points += result["points_scored"]

            throws.append({
                "action": action,
                "target": target,
                "hit_type": hit_type,
                "result": result,
            })

        return {
            "throws": throws,
            "total_points": total_points,
            "triple_count": triple_count,
            "points_scored": total_points > 0,
        }

    def _best_hit_type(self, target: int) -> str:
        """Return the best hit type for a target (triple for numbers, double for bull)."""
        if target == BULL:
            return "double"
        return "triple"

    def _is_closed(self, game: DartsCricketGame, player: int, target_idx: int) -> bool:
        """Check if a player has closed a target."""
        return game.marks[player][target_idx] >= MARKS_TO_CLOSE

    def _unclosed_targets(self, game: DartsCricketGame, player: int) -> List[int]:
        """Return list of target values that the player has NOT closed."""
        unclosed = []
        for i, target in enumerate(TARGETS):
            if not self._is_closed(game, player, i):
                unclosed.append(target)
        return unclosed

    def _closed_targets(self, game: DartsCricketGame, player: int) -> List[int]:
        """Return list of target values that the player HAS closed."""
        closed = []
        for i, target in enumerate(TARGETS):
            if self._is_closed(game, player, i):
                closed.append(target)
        return closed

    def _scoreable_targets(self, game: DartsCricketGame) -> List[int]:
        """Return targets where self has closed but opponent has NOT (can score)."""
        opp = 1 - self.player_id
        scoreable = []
        for i, target in enumerate(TARGETS):
            if (self._is_closed(game, self.player_id, i)
                    and not self._is_closed(game, opp, i)):
                scoreable.append(target)
        return scoreable


# Value-descending order for targets (Bull sorted last since it's 25 but strategically less valuable)
_DESCENDING_TARGETS: List[int] = [20, 19, 18, 17, 16, 15, BULL]


def _target_value(target: int) -> int:
    """Face value for sorting (Bull=25)."""
    return 25 if target == BULL else target


class FrongelloStrategy(StrategyBot):
    """Parameterized Frongello strategy bot covering all 17 strategies.

    The decision flow per dart is a committed sequence within each turn:

    1. **Pick intended target** based on strategy phase (chase / score / cover).
    2. **Aim** with best hit type (triple for numbers, double for bull).
    3. **Resolve** via skill profile (hit/miss/downgrade).
    4. **Reassess for remaining darts** based on what actually happened:
       - Did the target close? → Score on it or move to next.
       - Can it still close this turn (assuming singles)? → Keep aiming.
       - Can't close? (extra-darts strategies) → Redirect to scoring.

    Parameters
    ----------
    player_id : int
        0 or 1.
    lead_multiplier : int | None
        Score-lead threshold multiplier.  ``None`` = always cover (S1 mode).
        ``0`` = score until ahead by any amount (S2 mode).
        ``N`` = score until ahead by N × highest-unblocked-value.
    use_extra_darts : bool
        If True, darts 2-3 within a turn redirect to scoring when the
        current closing target can't be finished (marks_needed > darts_left).
    use_chase : bool
        If True, first priority each dart is to close the highest-value
        target the opponent has closed that we haven't.
    """

    def __init__(
        self,
        player_id: int,
        lead_multiplier: Optional[int] = None,
        use_extra_darts: bool = False,
        use_chase: bool = False,
    ) -> None:
        super().__init__(player_id)
        self.lead_multiplier = lead_multiplier
        self.use_extra_darts = use_extra_darts
        self.use_chase = use_chase
        self._dart_in_turn: int = 0

    # ── Public API ────────────────────────────────────────────────────

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        # Track dart number within the turn
        if game.darts_remaining == DARTS_PER_TURN:
            self._dart_in_turn = 0
        self._dart_in_turn += 1

        # --- S1 mode: always close in fixed order, score only when all closed ---
        if self.lead_multiplier is None:
            return self._s1_decision(game)

        # --- S10-S17: chase first ---
        if self.use_chase:
            chase = self._get_chase_target(game)
            if chase is not None:
                return chase, self._best_hit_type(chase)

        # --- Core S2-S5 decision: score vs cover ---
        target, hit_type = self._score_or_cover(game)

        # --- Extra-darts redirect ---
        # Frongello: "if at any point the next open number can not be
        # completely covered assuming single hits with darts remaining in
        # hand, the strategy aims these 'extra darts' at the highest number
        # the strategy has closed and can point on."
        #
        # This only applies when the core decision chose to CLOSE a number
        # (target is unclosed by self).  If the core decision was to SCORE
        # on an already-closed target, no redirect needed.
        if self.use_extra_darts:
            redirected = self._apply_extra_darts(game, target)
            if redirected is not None:
                return redirected

        return target, hit_type

    def play_turn(self, game: DartsCricketGame) -> Dict:
        """Override to reset per-turn state properly."""
        self._dart_in_turn = 0
        return super().play_turn(game)

    # ── S1 logic ──────────────────────────────────────────────────────

    def _s1_decision(self, game: DartsCricketGame) -> Tuple[int, str]:
        """S1: close in descending order 20→Bull. Score only after all closed."""
        unclosed = self._unclosed_targets(game, self.player_id)
        if unclosed:
            # Close highest unclosed by self, in fixed descending order
            for target in _DESCENDING_TARGETS:
                if target in unclosed:
                    return target, self._best_hit_type(target)

        # All closed: if behind, score on highest pointable target
        opp = 1 - self.player_id
        if game.scores[self.player_id] < game.scores[opp]:
            scoreable = self._scoreable_targets(game)
            if scoreable:
                # Use strategic descending order (Bull last)
                for target in _DESCENDING_TARGETS:
                    if target in scoreable:
                        return target, self._best_hit_type(target)

        # Ahead or tied with all closed — throw at Bull
        return BULL, self._best_hit_type(BULL)

    # ── S2-S5 core logic ──────────────────────────────────────────────

    def _score_or_cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        """S2-S5: score until lead exceeds threshold, then cover.

        Scoring: aim at highest scoreable target (we closed, opponent hasn't).
                 If nothing scoreable, close highest unclosed by self
                 (to create a scoring opportunity).
        Covering: close the highest target NOT closed by opponent
                  (defensive — blocks their future scoring).
        """
        if self._lead_below_threshold(game):
            # --- SCORE phase ---
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)
            # Nothing scoreable — close highest unclosed by self to create opportunity
            return self._close_highest_unclosed_by_self(game)
        else:
            # --- COVER phase ---
            return self._cover_highest_unclosed_by_opponent(game)

    def _lead_below_threshold(self, game: DartsCricketGame) -> bool:
        """Return True if our lead is at or below the scoring threshold."""
        opp = 1 - self.player_id
        lead = game.scores[self.player_id] - game.scores[opp]
        threshold = self.lead_multiplier * self._highest_unblocked_value(game)
        return lead <= threshold

    def _close_highest_unclosed_by_self(self, game: DartsCricketGame) -> Tuple[int, str]:
        """Close the highest target WE haven't closed yet (offensive)."""
        unclosed = self._unclosed_targets(game, self.player_id)
        if unclosed:
            for target in _DESCENDING_TARGETS:
                if target in unclosed:
                    return target, self._best_hit_type(target)
        # All closed fallback — score or throw bull
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, self._best_hit_type(BULL)

    def _cover_highest_unclosed_by_opponent(self, game: DartsCricketGame) -> Tuple[int, str]:
        """Close the highest target the OPPONENT hasn't closed yet (defensive).

        This blocks the opponent's future scoring opportunities.  If we've
        already closed everything the opponent hasn't, fall back to closing
        our own remaining targets.
        """
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)

        # Among opponent's unclosed targets, find highest that WE haven't closed
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._best_hit_type(target)

        # We've closed everything the opponent hasn't — close our remaining
        return self._close_highest_unclosed_by_self(game)

    # ── Chase logic (S10-S17) ─────────────────────────────────────────

    def _get_chase_target(self, game: DartsCricketGame) -> Optional[int]:
        """Return the highest-value target where opponent is closed and we are not.

        This is "chasing" — closing targets the opponent can currently score on.
        """
        opp = 1 - self.player_id
        for target in _DESCENDING_TARGETS:
            idx = TARGETS.index(target)
            if (self._is_closed(game, opp, idx)
                    and not self._is_closed(game, self.player_id, idx)):
                return target
        return None

    # ── Extra-darts logic (S6-S9, S14-S17) ────────────────────────────

    def _apply_extra_darts(
        self, game: DartsCricketGame, core_target: int
    ) -> Optional[Tuple[int, str]]:
        """Redirect to scoring if the core closing target can't be finished.

        Frongello: "if at any point the next open number can not be completely
        covered assuming single hits with darts remaining in hand, the strategy
        aims these 'extra darts' at the highest number the strategy has closed
        and can point on.  If no targets are currently closed and pointable,
        the strategy works on closing/pointing on the highest number not closed
        by the opponent."

        Only fires when ``core_target`` is a number WE haven't closed yet
        (i.e., the core decision was to *close* something, not to score).

        Returns (target, hit_type) if redirecting, or None to use core decision.
        """
        idx = TARGETS.index(core_target)

        # If the core target is already closed by us, we're scoring — no redirect
        if self._is_closed(game, self.player_id, idx):
            return None

        # Core decision is to close core_target.  Can we finish it this turn?
        marks_have = game.marks[self.player_id][idx]
        marks_still_needed = MARKS_TO_CLOSE - marks_have
        darts_left = game.darts_remaining  # includes current dart

        if marks_still_needed <= darts_left:
            # Can close with remaining darts assuming singles — keep aiming
            return None

        # Can't close → redirect to scoring on highest pointable target
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)

        # "If no targets are currently closed and pointable, the strategy
        # works on closing/pointing on the highest number not closed by
        # the opponent."
        opp = 1 - self.player_id
        for target in _DESCENDING_TARGETS:
            t_idx = TARGETS.index(target)
            if not self._is_closed(game, opp, t_idx) and not self._is_closed(
                game, self.player_id, t_idx
            ):
                return target, self._best_hit_type(target)

        # Fallback: nothing to redirect to
        return None

    # ── Shared helpers ────────────────────────────────────────────────

    def _highest_unblocked_value(self, game: DartsCricketGame) -> int:
        """Return face value of highest target NOT closed by opponent.

        Used for computing the scoring threshold.
        """
        opp = 1 - self.player_id
        for target in _DESCENDING_TARGETS:
            idx = TARGETS.index(target)
            if not self._is_closed(game, opp, idx):
                return _target_value(target)
        return 0


# ── Strategy factory + registry ───────────────────────────────────────

_STRATEGY_PARAMS: Dict[str, Tuple[Optional[int], bool, bool]] = {
    "S1":  (None, False, False),
    "S2":  (0,    False, False),
    "S3":  (3,    False, False),
    "S4":  (6,    False, False),
    "S5":  (9,    False, False),
    "S6":  (0,    True,  False),
    "S7":  (3,    True,  False),
    "S8":  (6,    True,  False),
    "S9":  (9,    True,  False),
    "S10": (0,    False, True),
    "S11": (3,    False, True),
    "S12": (6,    False, True),
    "S13": (9,    False, True),
    "S14": (0,    True,  True),
    "S15": (3,    True,  True),
    "S16": (6,    True,  True),
    "S17": (9,    True,  True),
}


def _strategy_class(name: str, lead_multiplier: Optional[int],
                    use_extra_darts: bool, use_chase: bool) -> type:
    """Create a concrete subclass of FrongelloStrategy with baked-in params."""

    class _Strategy(FrongelloStrategy):
        def __init__(self, player_id: int) -> None:
            super().__init__(
                player_id,
                lead_multiplier=lead_multiplier,
                use_extra_darts=use_extra_darts,
                use_chase=use_chase,
            )

    _Strategy.__name__ = name
    _Strategy.__qualname__ = name
    return _Strategy


# Build the registry
STRATEGY_CLASSES: Dict[str, type] = {}
for _name, (_lm, _ed, _ch) in _STRATEGY_PARAMS.items():
    STRATEGY_CLASSES[_name] = _strategy_class(_name, _lm, _ed, _ch)


# ── Backward-compatible aliases ───────────────────────────────────────
SequentialCloser = STRATEGY_CLASSES["S1"]
LeadThenCover = STRATEGY_CLASSES["S2"]
LeadPlusExtraDarts = STRATEGY_CLASSES["S6"]
ChaseAndCover = STRATEGY_CLASSES["S10"]


# ══════════════════════════════════════════════════════════════════════
# Experimental strategies (E1-E4) — hypothetical alternatives to S2
# ══════════════════════════════════════════════════════════════════════

# Closing order with Bull moved after 17 (E1 uses this)
_EARLY_BULL_ORDER: List[int] = [20, 19, 18, 17, BULL, 16, 15]


class EarlyBullStrategy(StrategyBot):
    """E1: S2 logic but close Bull right after 17 instead of last.

    Closing order: 20→19→18→17→Bull→16→15
    Rationale: Bull is worth 25 points.  Getting it closed early means more
    scoring potential on the highest face-value target.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        lead = game.scores[self.player_id] - game.scores[opp]

        if lead <= 0:
            # SCORE phase — aim at highest scoreable
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)
            # Nothing scoreable — close next in early-bull order
            return self._close_next(game)
        else:
            # COVER phase — close highest opp-unclosed we haven't closed
            opp_unclosed = self._unclosed_targets(game, opp)
            for target in _EARLY_BULL_ORDER:
                if target in opp_unclosed and not self._is_closed(
                    game, self.player_id, TARGETS.index(target)
                ):
                    return target, self._best_hit_type(target)
            # Fall back to closing our own
            return self._close_next(game)

    def _close_next(self, game: DartsCricketGame) -> Tuple[int, str]:
        """Close next unclosed target in early-bull order."""
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _EARLY_BULL_ORDER:
            if target in unclosed:
                return target, self._best_hit_type(target)
        # All closed — score on highest scoreable or throw bull
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, self._best_hit_type(BULL)


class HoneypotStrategy(StrategyBot):
    """E2: S2 + Honeypot — leave opponent's high open number as a scoring trap.

    When ahead on points in cover phase, if the opponent has an unclosed high
    number that WE have already closed (i.e. we can score on it), skip closing
    that number defensively and close other numbers instead.  Save the
    opponent's open number as a future scoring opportunity.

    Falls back to standard S2 logic when behind or tied.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        lead = game.scores[self.player_id] - game.scores[opp]

        if lead <= 0:
            # SCORE phase — same as S2
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)
            return self._close_highest_unclosed(game)
        else:
            # COVER phase with honeypot twist
            return self._honeypot_cover(game)

    def _honeypot_cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        """Cover but skip the highest scoreable target we could point on later."""
        opp = 1 - self.player_id
        scoreable = set(self._scoreable_targets(game))
        opp_unclosed = self._unclosed_targets(game, opp)

        # Find the highest scoreable we want to preserve as honeypot
        honeypot = None
        if scoreable:
            for target in _DESCENDING_TARGETS:
                if target in scoreable:
                    honeypot = target
                    break

        # Cover highest opp-unclosed we haven't closed, SKIPPING the honeypot
        for target in _DESCENDING_TARGETS:
            if target == honeypot:
                continue  # leave this one open for future scoring
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._best_hit_type(target)

        # If only the honeypot is left uncovered, we need to close it or
        # close our own remaining targets
        return self._close_highest_unclosed(game)

    def _close_highest_unclosed(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        if unclosed:
            for target in _DESCENDING_TARGETS:
                if target in unclosed:
                    return target, self._best_hit_type(target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, self._best_hit_type(BULL)


class GreedyCloseAndScore(StrategyBot):
    """E3: Close-then-score within the same turn.

    Like S2 but exploits the moment a target closes.  When you close a target
    mid-turn, any remaining darts in that turn immediately score on it (since
    the opponent hasn't had a chance to respond yet).  Only after the turn
    ends does it switch to covering.

    Per dart: if the previous dart this turn just closed something, keep
    scoring on it.  Otherwise, use standard S2 score/cover logic.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self._just_closed: Optional[int] = None

    def play_turn(self, game: DartsCricketGame) -> Dict:
        self._just_closed = None
        throws: List[Dict] = []
        total_points: int = 0
        triple_count: int = 0

        for _ in range(DARTS_PER_TURN):
            if game.game_over or game.current_player != self.player_id:
                break

            target, hit_type = self.choose_throw(game)
            action = AIPlayer.target_hit_to_action(target, hit_type)
            result = game.throw_dart(target, hit_type)

            if result.get("actual_hit_type") == "triple":
                triple_count += 1
            total_points += result["points_scored"]

            # Track if we just closed something — score on it next dart
            if result["closed"]:
                self._just_closed = result["target"]
            elif self._just_closed is not None:
                # Check if our closed target is still scoreable
                opp = 1 - self.player_id
                idx = TARGETS.index(self._just_closed)
                if self._is_closed(game, opp, idx):
                    self._just_closed = None  # opponent closed it too

            throws.append({
                "action": action, "target": target,
                "hit_type": hit_type, "result": result,
            })

        return {
            "throws": throws, "total_points": total_points,
            "triple_count": triple_count, "points_scored": total_points > 0,
        }

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id

        # If we just closed a target, keep scoring on it
        if self._just_closed is not None:
            idx = TARGETS.index(self._just_closed)
            if (self._is_closed(game, self.player_id, idx)
                    and not self._is_closed(game, opp, idx)):
                return self._just_closed, self._best_hit_type(self._just_closed)
            self._just_closed = None

        # Standard S2 logic
        lead = game.scores[self.player_id] - game.scores[opp]
        if lead <= 0:
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)
            return self._close_highest(game)
        else:
            return self._cover(game)

    def _close_highest(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                return target, self._best_hit_type(target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, self._best_hit_type(BULL)

    def _cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._best_hit_type(target)
        return self._close_highest(game)


class AdaptiveThresholdStrategy(StrategyBot):
    """E4: S2 with a threshold that adapts to game progress.

    Early game (many targets unclosed): score more aggressively before
    covering, like S3 (threshold = 3× highest unblocked).
    Late game (few targets left): cover quickly like S2 (threshold = 0).

    The threshold scales linearly: threshold_multiplier = 3 × (unclosed/7).
    When 7/7 unclosed → mult=3.  When 1/7 unclosed → mult≈0.43.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id

        # Adaptive threshold
        unclosed_count = len(self._unclosed_targets(game, self.player_id))
        threshold_mult = 3.0 * (unclosed_count / len(TARGETS))

        lead = game.scores[self.player_id] - game.scores[opp]
        highest_unblocked = self._highest_unblocked(game)
        threshold = threshold_mult * highest_unblocked

        if lead <= threshold:
            # SCORE phase
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)
            return self._close_highest(game)
        else:
            # COVER phase
            return self._cover(game)

    def _highest_unblocked(self, game: DartsCricketGame) -> int:
        opp = 1 - self.player_id
        for target in _DESCENDING_TARGETS:
            idx = TARGETS.index(target)
            if not self._is_closed(game, opp, idx):
                return _target_value(target)
        return 0

    def _close_highest(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                return target, self._best_hit_type(target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, self._best_hit_type(BULL)

    def _cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._best_hit_type(target)
        return self._close_highest(game)


class SmartAimStrategy(StrategyBot):
    """E5: S2 logic but choose hit type based on situation.

    - SCORING: always aim triple (max expected points)
    - CLOSING with 1 mark needed: aim single (96% hit vs 86% triple at pro)
    - CLOSING with 2 marks needed, 1 dart left: aim double (only way to close)
    - CLOSING with 2 marks needed, 2+ darts left: aim triple (chance to close
      in 1 dart, and if downgraded to double still closes)
    - CLOSING with 3 marks needed: aim triple (only way to close in 1 dart)
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        lead = game.scores[self.player_id] - game.scores[opp]

        if lead <= 0:
            # SCORE phase
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)  # triple for max pts
            # Close to create scoring opportunity
            return self._smart_close(game)
        else:
            # COVER phase
            return self._smart_cover(game)

    def _smart_hit_type(self, game: DartsCricketGame, target: int) -> str:
        """Choose hit type based on marks needed to close."""
        if target == BULL:
            return "double"
        idx = TARGETS.index(target)
        marks = game.marks[self.player_id][idx]
        needed = MARKS_TO_CLOSE - marks
        if needed <= 1:
            return "single"  # high reliability to close
        elif needed == 2 and game.darts_remaining == 1:
            return "double"  # only shot, need exactly 2
        else:
            return "triple"  # need 3, or have multiple darts

    def _smart_close(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                return target, self._smart_hit_type(game, target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, "double"

    def _smart_cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._smart_hit_type(game, target)
        return self._smart_close(game)


class AlwaysSingleStrategy(StrategyBot):
    """E6: S2 logic but always aim singles (except Bull=double).

    Control experiment: what if we never aim for triples/doubles?
    Singles are the most reliable (96% hit at pro vs 86% triple).
    Trades scoring punch for consistency.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def _aim(self, target: int) -> str:
        return "double" if target == BULL else "single"

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        lead = game.scores[self.player_id] - game.scores[opp]

        if lead <= 0:
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._aim(best)
            return self._close(game)
        else:
            return self._cover(game)

    def _close(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                return target, self._aim(target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._aim(best)
        return BULL, "double"

    def _cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._aim(target)
        return self._close(game)


class DoubleOrNothingStrategy(StrategyBot):
    """E7: S2 logic but always aim doubles.

    Doubles are the middle ground: 2 marks on hit (vs 3 triple, 1 single),
    with moderate reliability (75% hit at pro).  Two doubles = close.
    Expected marks per dart: 1.15 (vs 1.88 triple, 0.96 single).
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def _aim(self, target: int) -> str:
        return "double" if target == BULL else "double"

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        lead = game.scores[self.player_id] - game.scores[opp]

        if lead <= 0:
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._aim(best)
            return self._close(game)
        else:
            return self._cover(game)

    def _close(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                return target, self._aim(target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._aim(best)
        return BULL, "double"

    def _cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._aim(target)
        return self._close(game)


class ScoreTripleCoverSingleStrategy(StrategyBot):
    """E8: Triples when scoring, singles when closing/covering.

    The sharpest version of the insight: when you NEED points, go big
    (triples = 37.6 expected pts on 20).  When you need to CLOSE numbers,
    go reliable (singles = 96% hit rate at pro).

    This maximizes scoring punch while minimizing wasted darts during
    the covering phase.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        lead = game.scores[self.player_id] - game.scores[opp]

        if lead <= 0:
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)  # TRIPLE for scoring
            return self._close_single(game)
        else:
            return self._cover_single(game)

    def _close_single(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                hit = "double" if target == BULL else "single"
                return target, hit
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, "double"

    def _cover_single(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                hit = "double" if target == BULL else "single"
                return target, hit
        return self._close_single(game)


class PhaseShiftStrategy(StrategyBot):
    """E9: Three-phase strategy — aggressive early, balanced mid, close-out late.

    Early (5+ unclosed by self): Score aggressively, threshold = 3× like S3.
      Build a point cushion while many scoring opportunities exist.
    Mid (3-4 unclosed): Standard S2 — score until any lead, then cover.
    Late (1-2 unclosed): Pure cover (like S1) — just close out and win.
      Points don't matter if you're already ahead; rush to finish.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        unclosed_count = len(self._unclosed_targets(game, self.player_id))
        lead = game.scores[self.player_id] - game.scores[opp]

        if unclosed_count >= 5:
            # EARLY: aggressive scoring (S3-like threshold)
            threshold = 3 * self._highest_unblocked(game)
            return self._score_or_cover(game, lead, threshold)
        elif unclosed_count >= 3:
            # MID: S2 logic (lead by any amount, then cover)
            return self._score_or_cover(game, lead, 0)
        else:
            # LATE: pure cover — just close out
            return self._cover(game)

    def _score_or_cover(
        self, game: DartsCricketGame, lead: int, threshold: float
    ) -> Tuple[int, str]:
        if lead <= threshold:
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)
            return self._close_highest(game)
        else:
            return self._cover(game)

    def _highest_unblocked(self, game: DartsCricketGame) -> int:
        opp = 1 - self.player_id
        for target in _DESCENDING_TARGETS:
            idx = TARGETS.index(target)
            if not self._is_closed(game, opp, idx):
                return _target_value(target)
        return 0

    def _close_highest(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                return target, self._best_hit_type(target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, self._best_hit_type(BULL)

    def _cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._best_hit_type(target)
        return self._close_highest(game)


class ScoreSurgeStrategy(StrategyBot):
    """E10: S2 but with a mid-game scoring surge.

    When 3-5 targets unclosed and behind or tied, temporarily become hyper-
    aggressive: score on EVERY scoreable target before covering anything.
    Only switch to covering when ahead by 2× highest unblocked.

    Early and late game: standard S2.

    Rationale: the mid-game is when the most scoreable targets exist
    (you've closed some, opponent hasn't caught up).  Exploit that window.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        unclosed_count = len(self._unclosed_targets(game, self.player_id))
        lead = game.scores[self.player_id] - game.scores[opp]

        if 3 <= unclosed_count <= 5:
            # MID-GAME SURGE: score until 2× threshold
            threshold = 2 * self._highest_unblocked(game)
            return self._score_or_cover(game, lead, threshold)
        else:
            # Early/late: standard S2
            return self._score_or_cover(game, lead, 0)

    def _score_or_cover(
        self, game: DartsCricketGame, lead: int, threshold: float
    ) -> Tuple[int, str]:
        if lead <= threshold:
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)
            return self._close_highest(game)
        else:
            return self._cover(game)

    def _highest_unblocked(self, game: DartsCricketGame) -> int:
        opp = 1 - self.player_id
        for target in _DESCENDING_TARGETS:
            idx = TARGETS.index(target)
            if not self._is_closed(game, opp, idx):
                return _target_value(target)
        return 0

    def _close_highest(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                return target, self._best_hit_type(target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, self._best_hit_type(BULL)

    def _cover(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._best_hit_type(target)
        return self._close_highest(game)


class KitchenSinkStrategy(StrategyBot):
    """E11: Best ideas combined — phase shifts + smart aim + honeypot.

    Combines:
    - Phase-shifting thresholds (E9): aggressive early, tight late
    - Smart aim (E5): singles to close, triples to score
    - Honeypot (E2): leave one scoring trap open when covering

    The everything-bagel strategy.
    """

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)

    def choose_throw(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        unclosed_count = len(self._unclosed_targets(game, self.player_id))
        lead = game.scores[self.player_id] - game.scores[opp]

        # Phase-based threshold
        if unclosed_count >= 5:
            threshold = 3 * self._highest_unblocked(game)
        elif unclosed_count >= 3:
            threshold = 0
        else:
            threshold = -9999  # always cover in endgame

        if lead <= threshold:
            # SCORE phase — use triples for max points
            scoreable = self._scoreable_targets(game)
            if scoreable:
                best = max(scoreable, key=_target_value)
                return best, self._best_hit_type(best)
            # Close to create opportunity — use smart aim
            return self._smart_close(game)
        else:
            # COVER phase — smart aim + honeypot
            return self._honeypot_cover_smart(game)

    def _smart_hit_type(self, game: DartsCricketGame, target: int) -> str:
        if target == BULL:
            return "double"
        idx = TARGETS.index(target)
        marks = game.marks[self.player_id][idx]
        needed = MARKS_TO_CLOSE - marks
        if needed <= 1:
            return "single"
        elif needed == 2 and game.darts_remaining == 1:
            return "double"
        else:
            return "triple"

    def _smart_close(self, game: DartsCricketGame) -> Tuple[int, str]:
        unclosed = self._unclosed_targets(game, self.player_id)
        for target in _DESCENDING_TARGETS:
            if target in unclosed:
                return target, self._smart_hit_type(game, target)
        scoreable = self._scoreable_targets(game)
        if scoreable:
            best = max(scoreable, key=_target_value)
            return best, self._best_hit_type(best)
        return BULL, "double"

    def _honeypot_cover_smart(self, game: DartsCricketGame) -> Tuple[int, str]:
        opp = 1 - self.player_id
        scoreable = set(self._scoreable_targets(game))

        # Find honeypot — highest scoreable to preserve
        honeypot = None
        if scoreable:
            for target in _DESCENDING_TARGETS:
                if target in scoreable:
                    honeypot = target
                    break

        opp_unclosed = self._unclosed_targets(game, opp)
        for target in _DESCENDING_TARGETS:
            if target == honeypot:
                continue
            if target in opp_unclosed and not self._is_closed(
                game, self.player_id, TARGETS.index(target)
            ):
                return target, self._smart_hit_type(game, target)
        return self._smart_close(game)

    def _highest_unblocked(self, game: DartsCricketGame) -> int:
        opp = 1 - self.player_id
        for target in _DESCENDING_TARGETS:
            idx = TARGETS.index(target)
            if not self._is_closed(game, opp, idx):
                return _target_value(target)
        return 0


EXPERIMENTAL_CLASSES: Dict[str, type] = {
    "E1": EarlyBullStrategy,
    "E2": HoneypotStrategy,
    "E3": GreedyCloseAndScore,
    "E4": AdaptiveThresholdStrategy,
    "E5": SmartAimStrategy,
    "E6": AlwaysSingleStrategy,
    "E7": DoubleOrNothingStrategy,
    "E8": ScoreTripleCoverSingleStrategy,
    "E9": PhaseShiftStrategy,
    "E10": ScoreSurgeStrategy,
    "E11": KitchenSinkStrategy,
}
