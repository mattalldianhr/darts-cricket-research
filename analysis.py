"""Q-table introspection and strategy visualization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    MARKS_TO_CLOSE,
    NUM_ACTIONS,
    TARGET_NAMES,
    TARGETS,
)
from game import DartsCricketGame

if TYPE_CHECKING:
    from agent import AIPlayer

logger = logging.getLogger(__name__)


def build_state(
    my_score: int = 0,
    opp_score: int = 0,
    my_marks: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0),
    opp_marks: Tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0),
) -> tuple:
    """Build a Q-table state tuple from human-readable parameters.

    marks are in order: [15, 16, 17, 18, 19, 20, Bull].
    Each mark is encoded as a strategic category:
      0 = untouched (0 raw marks)
      1 = in progress (1-2 raw marks)
      2 = closed, opponent open (scoreable)
      3 = both closed (dead)
    Score is bucketed into differential categories (-2 to +2).
    """
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


def analyze_opening_preferences(agent: AIPlayer) -> List[Tuple[int, str, float]]:
    """Return ranked (action_idx, label, q_value) for the clean starting state.

    Sorted by Q-value descending.
    """
    from agent import AIPlayer as _AP

    starting_state = build_state()
    results: List[Tuple[int, str, float]] = []
    for action_idx in range(NUM_ACTIONS):
        q_value = agent.q_table[starting_state][action_idx]
        label = _AP.get_action_label(action_idx)
        results.append((action_idx, label, q_value))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def analyze_state_policy(
    agent: AIPlayer, state: tuple
) -> List[Tuple[int, str, float]]:
    """Return ranked (action_idx, label, q_value) for an arbitrary state.

    Sorted by Q-value descending.
    """
    from agent import AIPlayer as _AP

    results: List[Tuple[int, str, float]] = []
    for action_idx in range(NUM_ACTIONS):
        q_value = agent.q_table[state][action_idx]
        label = _AP.get_action_label(action_idx)
        results.append((action_idx, label, q_value))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def analyze_chase_behavior(agent: AIPlayer) -> Dict:
    """Check if agent prioritizes chasing numbers the opponent has closed.

    Constructs states where opponent has closed specific numbers we haven't,
    then checks the agent's top action preference.
    """
    from agent import AIPlayer as _AP

    results: Dict = {}
    # Check each non-bull target (indices 0-5 = 15-20)
    for target_idx, target in enumerate(TARGETS):
        target_name = TARGET_NAMES[target]
        # Opponent closed this number, we have 0 marks on it
        opp_marks = [0] * 7
        opp_marks[target_idx] = MARKS_TO_CLOSE
        my_marks = [0] * 7

        state = build_state(
            my_score=0,
            opp_score=0,
            my_marks=tuple(my_marks),
            opp_marks=tuple(opp_marks),
        )

        policy = analyze_state_policy(agent, state)
        top_action_idx, top_label, top_q = policy[0]
        top_target, top_hit = _AP.action_to_target_hit(top_action_idx)

        chases = top_target == target
        results[f"opp_closed_{target_name}"] = {
            "top_action": top_label,
            "q_value": round(top_q, 4),
            "chases_target": chases,
        }

    return results


def analyze_scoring_vs_closing(agent: AIPlayer) -> Dict:
    """At various score differentials, check if agent prefers scoring or closing.

    Tests scenarios: behind by 40, tied, ahead by 40, ahead by 80.
    Uses a state where we have closed 3 of 7 targets and opponent has closed 2.
    """
    from agent import AIPlayer as _AP

    # We closed 15, 16, 17 (indices 0-2); opponent closed 15, 16 (indices 0-1)
    my_marks_base = (3, 3, 3, 0, 0, 0, 0)
    opp_marks_base = (3, 3, 0, 0, 0, 0, 0)

    scenarios = {
        "behind_40": (-40, 0),
        "tied": (0, 0),
        "ahead_40": (40, 0),
        "ahead_80": (80, 0),
    }

    results: Dict = {}
    for name, (my_score, opp_score) in scenarios.items():
        state = build_state(
            my_score=max(my_score, 0),
            opp_score=max(-my_score, 0) if my_score < 0 else opp_score,
            my_marks=my_marks_base,
            opp_marks=opp_marks_base,
        )

        policy = analyze_state_policy(agent, state)
        top_action_idx, top_label, top_q = policy[0]
        top_target, top_hit = _AP.action_to_target_hit(top_action_idx)

        # Check if target is already closed by us (scoring) or unclosed (closing)
        target_idx = TARGETS.index(top_target)
        is_scoring = my_marks_base[target_idx] >= MARKS_TO_CLOSE
        results[name] = {
            "score_diff": my_score,
            "top_action": top_label,
            "q_value": round(top_q, 4),
            "strategy": "scoring" if is_scoring else "closing",
        }

    return results


def generate_strategy_heatmap(agent: AIPlayer, output_path: str) -> None:
    """Create a heatmap: X=score differential, Y=targets closed, color=strategy type.

    Cell color: 0=close unclosed number, 1=score on closed number.
    """
    from agent import AIPlayer as _AP

    score_diffs = list(range(-100, 101, 20))
    targets_closed_range = list(range(8))  # 0-7

    data = np.zeros((len(targets_closed_range), len(score_diffs)))

    for yi, num_closed in enumerate(targets_closed_range):
        my_marks = [MARKS_TO_CLOSE] * num_closed + [0] * (7 - num_closed)
        opp_marks = [0] * 7

        for xi, diff in enumerate(score_diffs):
            my_score = max(diff, 0)
            opp_score = max(-diff, 0)

            state = build_state(
                my_score=my_score,
                opp_score=opp_score,
                my_marks=tuple(my_marks),
                opp_marks=tuple(opp_marks),
            )

            policy = analyze_state_policy(agent, state)
            top_action_idx = policy[0][0]
            top_target, _ = _AP.action_to_target_hit(top_action_idx)
            target_idx = TARGETS.index(top_target)

            # Is the top target already closed by us?
            is_scoring = my_marks[target_idx] >= MARKS_TO_CLOSE
            data[yi, xi] = 1.0 if is_scoring else 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(2)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")

    ax.set_xticks(range(len(score_diffs)))
    ax.set_xticklabels([str(d) for d in score_diffs])
    ax.set_yticks(range(len(targets_closed_range)))
    ax.set_yticklabels([str(c) for c in targets_closed_range])

    ax.set_xlabel("Score Differential (self - opponent)")
    ax.set_ylabel("Number of Targets Closed (self)")
    ax.set_title("Strategy Heatmap: Close (green) vs Score (red)")

    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["Close", "Score"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Strategy heatmap saved to %s", output_path)


def generate_validation_plots(report: object, output_dir: str) -> None:
    """Generate validation visualizations from a ValidationReport.

    Creates:
    1. Round-robin win rate matrix heatmap
    2. Principle pass/fail bar chart
    3. Opening preference bar chart
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # 1. Round-robin win rate matrix heatmap
    if hasattr(report, "round_robin_results") and report.round_robin_results:
        names = list(report.round_robin_results.keys())
        n = len(names)
        matrix = np.zeros((n, n))
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if name_j in report.round_robin_results[name_i]:
                    matrix[i, j] = report.round_robin_results[name_i][name_j]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(names)
        ax.set_title("Round-Robin Win Rate Matrix")
        ax.set_xlabel("Opponent")
        ax.set_ylabel("Player")
        fig.colorbar(im, ax=ax, label="Win Rate")

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "round_robin_matrix.png"), dpi=150)
        plt.close(fig)

    # 2. Principle pass/fail bar chart
    if hasattr(report, "principle_results") and report.principle_results:
        names = list(report.principle_results.keys())
        passes = [1 if report.principle_results[n] else 0 for n in names]
        colors = ["green" if p else "red" for p in passes]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(names)), [1] * len(names), color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_title("Frongello Principle Validation")
        ax.set_ylabel("Pass / Fail")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "principle_results.png"), dpi=150)
        plt.close(fig)

    # 3. Opening preference bar chart
    if hasattr(report, "opening_preferences") and report.opening_preferences:
        labels = [item[1] for item in report.opening_preferences[:10]]
        values = [item[2] for item in report.opening_preferences[:10]]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(labels)), values, color="steelblue")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Q-Value")
        ax.set_title("Opening Move Preferences (Top 10)")
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "opening_preferences.png"), dpi=150)
        plt.close(fig)

    logger.info("Validation plots saved to %s", output_dir)
