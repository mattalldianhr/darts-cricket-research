"""DartsCricketGame class: pure game logic, no AI concerns."""

import logging
import random
from typing import Dict, List, Optional, Tuple

from config import (
    BULL,
    BULL_DIFFICULTY_MULTIPLIER,
    BULL_HIT_TYPES,
    DARTS_PER_TURN,
    HIT_MULTIPLIER,
    HIT_PROBABILITY,
    HIT_TYPES,
    MARKS_TO_CLOSE,
    SCORE_BUCKET_THRESHOLDS,
    SKILL_PROFILE_LEGACY,
    SkillProfile,
    TARGET_NAMES,
    TARGETS,
)

logger = logging.getLogger(__name__)


class DartsCricketGame:
    """Cricket darts game engine for two players."""

    def __init__(
        self,
        miss_enabled: bool = False,
        skill_profiles: Optional[List[SkillProfile]] = None,
    ) -> None:
        # skill_profiles takes precedence; miss_enabled is backward compat
        if skill_profiles is not None:
            self.skill_profiles: Optional[List[SkillProfile]] = list(skill_profiles)
        elif miss_enabled:
            self.skill_profiles = [SKILL_PROFILE_LEGACY, SKILL_PROFILE_LEGACY]
        else:
            self.skill_profiles = None
        self.targets: List[int] = list(TARGETS)
        self.scores: List[int] = [0, 0]
        self.marks: List[List[int]] = [[0] * len(TARGETS), [0] * len(TARGETS)]
        self.current_player: int = 0
        self.darts_remaining: int = DARTS_PER_TURN
        self.total_darts_thrown: int = 0
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.reset()

    @property
    def miss_enabled(self) -> bool:
        """Backward-compat property: True when any skill profile is active."""
        return self.skill_profiles is not None

    def reset(self) -> None:
        """Zero all state. Player 0 starts with 3 darts."""
        self.scores = [0, 0]
        self.marks = [[0] * len(self.targets), [0] * len(self.targets)]
        self.current_player = 0
        self.darts_remaining = DARTS_PER_TURN
        self.total_darts_thrown = 0
        self.game_over = False
        self.winner = None

    @staticmethod
    def bucket_score_diff(self_score: int, opp_score: int) -> int:
        """Bucket the score differential into strategic categories.

        Returns: -2 (far behind), -1 (behind), 0 (tied), 1 (ahead), 2 (far ahead)
        Threshold is SCORE_BUCKET_THRESHOLDS[1] (default 40 points).
        """
        diff = self_score - opp_score
        if diff == 0:
            return 0
        threshold = SCORE_BUCKET_THRESHOLDS[1]
        if diff > 0:
            return 2 if diff >= threshold else 1
        else:
            return -2 if diff <= -threshold else -1

    @staticmethod
    def _encode_mark(my_marks: int, opp_marks: int) -> int:
        """Encode a single target's marks as a strategic category.

        Returns:
          0 = untouched (0 marks)
          1 = in progress (1-2 marks)
          2 = closed, opponent open (scoreable)
          3 = closed, opponent also closed (dead)
        """
        if my_marks == 0:
            return 0
        elif my_marks < MARKS_TO_CLOSE:
            return 1
        elif opp_marks < MARKS_TO_CLOSE:
            return 2
        else:
            return 3

    def get_state(self, perspective: int) -> Tuple:
        """Return hashable state tuple from the given player's perspective.

        Returns (score_bucket, (marks_self_tuple, marks_opp_tuple))
        where score_bucket is a bucketed differential (-2 to +2),
        and marks are encoded as strategic categories (0-3).

        Each mark is encoded as:
          0 = untouched (0 marks)
          1 = in progress (1-2 marks)
          2 = closed (3+ marks), opponent open (scoreable)
          3 = closed, opponent also closed (dead)
        """
        opp = 1 - perspective
        marks_self: List[int] = []
        marks_opp: List[int] = []
        for i in range(len(self.targets)):
            my_m = self.marks[perspective][i]
            opp_m = self.marks[opp][i]
            marks_self.append(self._encode_mark(my_m, opp_m))
            marks_opp.append(self._encode_mark(opp_m, my_m))
        score_bucket = self.bucket_score_diff(
            self.scores[perspective], self.scores[opp]
        )
        return (
            score_bucket,
            (tuple(marks_self), tuple(marks_opp)),
        )

    def throw_dart(self, target: int, hit_type: str) -> Dict:
        """Process a single dart throw and return result details."""
        if self.game_over:
            logger.warning("Attempted throw after game over")
            return {
                "target": target,
                "hit_type": hit_type,
                "actual_hit_type": None,
                "marks_added": 0,
                "closed": False,
                "points_scored": 0,
                "missed": True,
                "game_over": True,
                "winner": self.winner,
            }

        target_idx = self.targets.index(target)
        player = self.current_player
        opponent = 1 - player

        missed = False
        actual_hit_type = hit_type

        if self.skill_profiles is not None:
            target, actual_hit_type, missed = self._apply_miss(target, hit_type)

        if missed:
            marks_added = 0
        else:
            marks_added = HIT_MULTIPLIER[actual_hit_type]

        old_marks = self.marks[player][target_idx]
        self.marks[player][target_idx] += marks_added

        closed = (
            self.marks[player][target_idx] >= MARKS_TO_CLOSE
            and old_marks < MARKS_TO_CLOSE
        )

        points_scored = 0
        if self.marks[player][target_idx] >= MARKS_TO_CLOSE and not self.is_closed(
            opponent, target_idx
        ):
            excess_marks = self.marks[player][target_idx] - max(
                old_marks, MARKS_TO_CLOSE
            )
            if excess_marks > 0:
                face_value = target
                points_scored = face_value * excess_marks
                self.scores[player] += points_scored

        self.total_darts_thrown += 1
        self.darts_remaining -= 1

        self.check_winner()

        if self.darts_remaining <= 0 and not self.game_over:
            self.switch_player()

        result = {
            "target": target,
            "hit_type": hit_type,
            "actual_hit_type": actual_hit_type if not missed else None,
            "marks_added": marks_added,
            "closed": closed,
            "points_scored": points_scored,
            "missed": missed,
            "game_over": self.game_over,
            "winner": self.winner,
        }

        logger.debug(
            "Player %d threw %s %s -> marks_added=%d, closed=%s, points=%d",
            player,
            hit_type,
            TARGET_NAMES.get(target, str(target)),
            marks_added,
            closed,
            points_scored,
        )

        return result

    def _apply_miss(
        self, target: int, hit_type: str
    ) -> Tuple[int, str, bool]:
        """Apply miss probability using the current player's SkillProfile.

        Returns (target, actual_hit_type, missed).
        """
        profile = self.skill_profiles[self.current_player]

        # Select the outcome distribution for the aimed hit type
        if hit_type == "triple":
            outcomes = dict(profile.triple_outcomes)
        elif hit_type == "double":
            outcomes = dict(profile.double_outcomes)
        else:
            outcomes = dict(profile.single_outcomes)

        # Bull target: triple is impossible — redistribute its mass to other outcomes
        if target == BULL and "triple" in outcomes and outcomes["triple"] > 0:
            triple_mass = outcomes.pop("triple")
            remaining = sum(outcomes.values())
            if remaining > 0:
                for key in outcomes:
                    outcomes[key] += triple_mass * (outcomes[key] / remaining)
            else:
                outcomes["miss"] = outcomes.get("miss", 0.0) + triple_mass

        # Bull target: reduce primary hit probability (smaller circular target)
        if target == BULL:
            primary = hit_type
            if primary in outcomes and outcomes[primary] > 0:
                original = outcomes[primary]
                reduced = original * BULL_DIFFICULTY_MULTIPLIER
                freed = original - reduced
                outcomes[primary] = reduced
                outcomes["miss"] = outcomes.get("miss", 0.0) + freed

        labels = list(outcomes.keys())
        weights = list(outcomes.values())
        result = random.choices(labels, weights=weights, k=1)[0]

        if result == "miss":
            return target, hit_type, True
        return target, result, False

    def check_winner(self) -> Optional[int]:
        """Check if either player has won. Sets game_over and winner."""
        for player in range(2):
            opponent = 1 - player
            if self.all_closed(player) and self.scores[player] >= self.scores[opponent]:
                self.game_over = True
                self.winner = player
                logger.info(
                    "Player %d wins! Score: %d-%d",
                    player,
                    self.scores[player],
                    self.scores[opponent],
                )
                return player
        return None

    def is_closed(self, player: int, target_idx: int) -> bool:
        """Check if a player has closed a specific target."""
        return self.marks[player][target_idx] >= MARKS_TO_CLOSE

    def all_closed(self, player: int) -> bool:
        """Check if a player has closed all targets."""
        return all(
            self.marks[player][i] >= MARKS_TO_CLOSE for i in range(len(self.targets))
        )

    def get_valid_actions(self) -> List[Tuple[int, str]]:
        """Return all valid (target, hit_type) action combinations.

        Bullseye only allows single and double (no triple on a real dartboard).
        """
        actions = []
        for target in self.targets:
            hit_types = BULL_HIT_TYPES if target == BULL else HIT_TYPES
            for hit_type in hit_types:
                actions.append((target, hit_type))
        return actions

    def switch_player(self) -> None:
        """Toggle current player and reset darts remaining."""
        self.current_player = 1 - self.current_player
        self.darts_remaining = DARTS_PER_TURN

    def get_board_display(self) -> str:
        """Generate ASCII board showing marks and scores for both players."""
        mark_symbols = {0: " ", 1: "/", 2: "X", 3: "O"}
        lines = []
        lines.append("=" * 40)
        lines.append(f"  P1 Score: {self.scores[0]:>5}    P2 Score: {self.scores[1]:>5}")
        lines.append("-" * 40)
        lines.append(f"  {'Target':^8} {'P1':^6} {'P2':^6}")
        lines.append("-" * 40)

        for i, target in enumerate(self.targets):
            name = TARGET_NAMES[target]
            p1_marks = min(self.marks[0][i], MARKS_TO_CLOSE)
            p2_marks = min(self.marks[1][i], MARKS_TO_CLOSE)
            p1_sym = mark_symbols[p1_marks]
            p2_sym = mark_symbols[p2_marks]
            lines.append(f"  {name:^8} {p1_sym:^6} {p2_sym:^6}")

        lines.append("=" * 40)

        if self.game_over:
            lines.append(f"  WINNER: Player {self.winner + 1}!")
        else:
            lines.append(
                f"  Current: Player {self.current_player + 1}"
                f"  Darts: {self.darts_remaining}"
            )

        return "\n".join(lines)
