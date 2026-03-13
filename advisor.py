"""Throw recommendation engine — get strategic advice from a trained Q-table."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from agent import AIPlayer
from config import (
    BULL,
    BULL_HIT_TYPES,
    HIT_MULTIPLIER,
    HIT_TYPES,
    MARKS_TO_CLOSE,
    NUM_ACTIONS,
    TARGET_NAMES,
    TARGETS,
)

if TYPE_CHECKING:
    from game import DartsCricketGame

logger = logging.getLogger(__name__)


@dataclass
class ThrowRecommendation:
    """A single throw recommendation with context."""

    target: int
    hit_type: str
    q_value: float
    rank: int
    explanation: str


@dataclass
class TurnAdvice:
    """Full turn advice: ranked recommendations + strategic context."""

    recommendations: List[ThrowRecommendation] = field(default_factory=list)
    strategic_context: str = ""


def state_from_board(
    my_score: int,
    opp_score: int,
    my_marks: List[int],
    opp_marks: List[int],
) -> tuple:
    """Convert board state to Q-table lookup format.

    marks lists are in order: [15, 16, 17, 18, 19, 20, Bull].
    Each mark is encoded as a strategic category:
      0 = untouched, 1 = in progress, 2 = closed (scoreable), 3 = both closed
    Score is bucketed into differential categories (-2 to +2).
    """
    from game import DartsCricketGame

    encoded_my = []
    encoded_opp = []
    for i in range(7):
        my_m = my_marks[i]
        opp_m = opp_marks[i]
        if my_m == 0:
            encoded_my.append(0)
        elif my_m < MARKS_TO_CLOSE:
            encoded_my.append(1)
        elif opp_m < MARKS_TO_CLOSE:
            encoded_my.append(2)
        else:
            encoded_my.append(3)
        if opp_m == 0:
            encoded_opp.append(0)
        elif opp_m < MARKS_TO_CLOSE:
            encoded_opp.append(1)
        elif my_m < MARKS_TO_CLOSE:
            encoded_opp.append(2)
        else:
            encoded_opp.append(3)
    score_bucket = DartsCricketGame.bucket_score_diff(my_score, opp_score)
    return (score_bucket, (tuple(encoded_my), tuple(encoded_opp)))


def get_advice(
    agent: AIPlayer,
    game: DartsCricketGame,
    player_perspective: int,
    top_n: int = 5,
) -> TurnAdvice:
    """Get ranked throw recommendations for the current board state.

    Returns a TurnAdvice with the top_n best actions and strategic context.
    """
    state = game.get_state(player_perspective)
    opp = 1 - player_perspective

    # Gather Q-values for all actions
    action_qvalues = []
    for action_idx in range(NUM_ACTIONS):
        q_value = agent.q_table[state][action_idx]
        action_qvalues.append((action_idx, q_value))

    # Sort by Q-value descending
    action_qvalues.sort(key=lambda x: x[1], reverse=True)

    # Build recommendations
    recommendations: List[ThrowRecommendation] = []
    for rank, (action_idx, q_value) in enumerate(action_qvalues[:top_n], start=1):
        explanation = explain_recommendation(
            action_idx, q_value, state, game, player_perspective
        )
        target, hit_type = AIPlayer.action_to_target_hit(action_idx)
        recommendations.append(
            ThrowRecommendation(
                target=target,
                hit_type=hit_type,
                q_value=q_value,
                rank=rank,
                explanation=explanation,
            )
        )

    # Build strategic context
    my_score = game.scores[player_perspective]
    opp_score = game.scores[opp]
    my_closed = sum(
        1 for i in range(len(TARGETS))
        if game.marks[player_perspective][i] >= MARKS_TO_CLOSE
    )
    opp_closed = sum(
        1 for i in range(len(TARGETS))
        if game.marks[opp][i] >= MARKS_TO_CLOSE
    )
    score_diff = my_score - opp_score

    context_parts = []
    if score_diff > 0:
        context_parts.append(f"Leading by {score_diff} points")
    elif score_diff < 0:
        context_parts.append(f"Trailing by {abs(score_diff)} points")
    else:
        context_parts.append("Scores tied")

    context_parts.append(f"You have closed {my_closed}/7, opponent {opp_closed}/7")

    if my_closed > opp_closed and score_diff >= 0:
        context_parts.append("In a strong position -- focus on closing remaining targets")
    elif my_closed < opp_closed:
        context_parts.append("Behind on closures -- prioritize catching up")
    elif score_diff < 0:
        context_parts.append("Need points -- consider scoring on closed targets")

    strategic_context = ". ".join(context_parts) + "."

    return TurnAdvice(recommendations=recommendations, strategic_context=strategic_context)


def explain_recommendation(
    action_idx: int,
    q_value: float,
    state: tuple,
    game: DartsCricketGame,
    perspective: int,
) -> str:
    """Generate human-readable explanation for a recommended action."""
    target, hit_type = AIPlayer.action_to_target_hit(action_idx)
    target_name = TARGET_NAMES.get(target, str(target))
    target_idx = TARGETS.index(target)
    opp = 1 - perspective

    my_marks = game.marks[perspective][target_idx]
    opp_marks = game.marks[opp][target_idx]
    marks_needed = max(0, MARKS_TO_CLOSE - my_marks)
    multiplier = HIT_MULTIPLIER[hit_type]

    parts = [f"{hit_type.capitalize()} {target_name} (Q={q_value:.2f})"]

    if my_marks >= MARKS_TO_CLOSE:
        if opp_marks >= MARKS_TO_CLOSE:
            parts.append(
                f"Target {target_name} is closed by both players. No benefit from this throw."
            )
        else:
            parts.append(
                f"Target {target_name} is closed by you but open for opponent. "
                f"Scores {target * multiplier} points."
            )
    else:
        parts.append(
            f"Target {target_name} is unclosed with {my_marks} marks ({marks_needed} needed to close)."
        )
        if multiplier >= marks_needed:
            parts.append(
                f"Aiming {hit_type} adds {multiplier} marks -- closes immediately."
            )
            excess = multiplier - marks_needed
            if excess > 0 and opp_marks < MARKS_TO_CLOSE:
                parts.append(
                    f"Excess {excess} marks score {target * excess} points."
                )
        else:
            parts.append(
                f"Aiming {hit_type} adds {multiplier} mark(s) toward closing."
            )

    return " ".join(parts)


def load_advisor(q_table_path: str) -> AIPlayer:
    """Load Q-table into an AIPlayer set for pure exploitation (epsilon=0)."""
    agent = AIPlayer(player_id=0, epsilon=0.0)
    agent.load_q_table(q_table_path)
    agent.epsilon = 0.0  # ensure pure exploitation after load
    return agent


def load_dqn_advisor(model_path: str):
    """Load DQN model into a DQNPlayer set for pure exploitation (epsilon=0).

    Returns a DQNPlayer whose q_table proxy makes it compatible with
    get_advice() and analysis functions without modification.
    """
    from dqn_agent import DQNPlayer
    agent = DQNPlayer(player_id=0, epsilon=0.0)
    agent.load_q_table(model_path)
    agent.epsilon = 0.0
    return agent


def load_a2c_advisor(model_path: str):
    """Load A2C model into an A2CPlayer set for pure exploitation.

    Returns an A2CPlayer whose q_table proxy makes it compatible with
    get_advice() and analysis functions without modification.
    """
    from a2c_agent import A2CPlayer
    agent = A2CPlayer(player_id=0, epsilon=0.0)
    agent.load_q_table(model_path)
    agent.epsilon = 0.0
    return agent
