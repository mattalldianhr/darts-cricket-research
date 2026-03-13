"""Tournament runner and Frongello principle validation for darts cricket."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent import AIPlayer
from config import (
    DARTS_PER_TURN,
    FRONGELLO_WIN_RATE_THRESHOLD,
    NUM_ACTIONS,
    SkillProfile,
    TARGETS,
    VALIDATION_GAMES_PER_MATCHUP,
)
from game import DartsCricketGame
from strategies import (
    ChaseAndCover,
    LeadThenCover,
    SequentialCloser,
    StrategyBot,
    STRATEGY_CLASSES,
)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class MatchupResult:
    """Result of a head-to-head matchup between two players."""

    player_a_name: str
    player_b_name: str
    wins_a: int
    wins_b: int
    total_games: int
    win_rate_a: float
    win_rate_b: float
    avg_score_a: float
    avg_score_b: float
    avg_game_length: float


@dataclass
class RoundRobinResult:
    """Result of a round-robin tournament."""

    matchups: List[MatchupResult]
    rankings: List[Tuple[str, float]]  # (name, overall_win_rate) sorted descending
    win_rate_matrix: Dict[str, Dict[str, float]]


@dataclass
class PrincipleResult:
    """Result of testing a single Frongello principle."""

    name: str
    passed: bool
    details: str
    evidence: Dict[str, Any]


@dataclass
class ValidationReport:
    """Full validation report with principle results and matchup data."""

    principles: List[PrincipleResult]
    matchups: List[MatchupResult]
    summary: str


# ── Core matchup runner ──────────────────────────────────────────────


def run_matchup(
    player_a: Any,
    player_b: Any,
    num_games: int,
    player_a_name: str = "PlayerA",
    player_b_name: str = "PlayerB",
    miss_enabled: bool = False,
    skill_profiles: Optional[List[SkillProfile]] = None,
) -> MatchupResult:
    """Play num_games between two players, alternating who goes first.

    Both players must have play_turn(game) and player_id attributes.
    """
    game = DartsCricketGame(miss_enabled=miss_enabled, skill_profiles=skill_profiles)
    wins_a = 0
    wins_b = 0
    total_score_a = 0
    total_score_b = 0
    total_darts = 0

    players = [player_a, player_b]

    for game_num in range(num_games):
        game.reset()

        # Alternate first player
        if game_num % 2 == 1:
            game.current_player = 1

        # Assign player IDs for this game
        player_a.player_id = 0
        player_b.player_id = 1

        max_turns = 200
        turns = 0
        while not game.game_over and turns < max_turns:
            current_player = players[game.current_player]
            current_player.play_turn(game)
            turns += 1

        if game.winner == 0:
            wins_a += 1
        elif game.winner == 1:
            wins_b += 1

        total_score_a += game.scores[0]
        total_score_b += game.scores[1]
        total_darts += game.total_darts_thrown

    return MatchupResult(
        player_a_name=player_a_name,
        player_b_name=player_b_name,
        wins_a=wins_a,
        wins_b=wins_b,
        total_games=num_games,
        win_rate_a=wins_a / max(1, num_games),
        win_rate_b=wins_b / max(1, num_games),
        avg_score_a=total_score_a / max(1, num_games),
        avg_score_b=total_score_b / max(1, num_games),
        avg_game_length=total_darts / max(1, num_games),
    )


def run_round_robin(
    players: Dict[str, Any],
    num_games: int,
) -> RoundRobinResult:
    """Run all-pairs matchups among named players.

    players: dict mapping name -> player object with play_turn and player_id.
    """
    names = list(players.keys())
    matchups: List[MatchupResult] = []
    win_rate_matrix: Dict[str, Dict[str, float]] = {n: {} for n in names}
    total_wins: Dict[str, int] = {n: 0 for n in names}
    total_games_played: Dict[str, int] = {n: 0 for n in names}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]
            result = run_matchup(
                players[name_a],
                players[name_b],
                num_games,
                player_a_name=name_a,
                player_b_name=name_b,
            )
            matchups.append(result)
            win_rate_matrix[name_a][name_b] = result.win_rate_a
            win_rate_matrix[name_b][name_a] = result.win_rate_b

            total_wins[name_a] += result.wins_a
            total_wins[name_b] += result.wins_b
            total_games_played[name_a] += result.total_games
            total_games_played[name_b] += result.total_games

    # Compute rankings by overall win rate
    rankings: List[Tuple[str, float]] = []
    for name in names:
        overall_wr = total_wins[name] / max(1, total_games_played[name])
        rankings.append((name, overall_wr))
    rankings.sort(key=lambda x: x[1], reverse=True)

    return RoundRobinResult(
        matchups=matchups,
        rankings=rankings,
        win_rate_matrix=win_rate_matrix,
    )


# ── Frongello principle tests ────────────────────────────────────────


def _test_principle_a(
    agent: AIPlayer,
    num_games: int,
    threshold: float,
    miss_enabled: bool = False,
    skill_profiles: Optional[List[SkillProfile]] = None,
) -> PrincipleResult:
    """Principle A: Score first.

    Run Q-agent vs S1 (SequentialCloser). Check that Q-agent wins > threshold
    AND that the top opening Q-value action targets 20.
    """
    s1 = SequentialCloser(player_id=1)
    result = run_matchup(agent, s1, num_games, "Q-Agent", "S1-SequentialCloser",
                         miss_enabled=miss_enabled, skill_profiles=skill_profiles)

    # Check starting state Q-values for action targeting 20
    game = DartsCricketGame()
    starting_state = game.get_state(perspective=0)
    q_vals = agent.get_starting_state_q_values(starting_state)
    top_action = max(q_vals, key=q_vals.get)
    top_target, top_hit = AIPlayer.action_to_target_hit(top_action)

    win_check = result.win_rate_a >= threshold
    target_check = top_target == 20

    passed = win_check and target_check
    details_parts = []
    details_parts.append(
        f"Win rate vs S1: {result.win_rate_a:.3f} "
        f"({'PASS' if win_check else 'FAIL'}, threshold={threshold})"
    )
    details_parts.append(
        f"Top opening action: {AIPlayer.get_action_label(top_action)} "
        f"({'PASS' if target_check else 'FAIL'}, expected target 20)"
    )

    return PrincipleResult(
        name="A: Score First",
        passed=passed,
        details="; ".join(details_parts),
        evidence={
            "win_rate_vs_s1": result.win_rate_a,
            "top_opening_action": AIPlayer.get_action_label(top_action),
            "top_opening_target": top_target,
            "matchup": result,
        },
    )


def _test_principle_b(
    agent: AIPlayer,
    threshold: float,
) -> PrincipleResult:
    """Principle B: Target 20.

    Check starting-state Q-value ranking; action targeting 20 should be in top 2.
    """
    game = DartsCricketGame()
    starting_state = game.get_state(perspective=0)
    q_vals = agent.get_starting_state_q_values(starting_state)

    # Sort actions by Q-value descending
    sorted_actions = sorted(q_vals.keys(), key=lambda a: q_vals[a], reverse=True)
    top_2_targets = []
    for a in sorted_actions[:2]:
        t, h = AIPlayer.action_to_target_hit(a)
        top_2_targets.append(t)

    has_20_in_top2 = 20 in top_2_targets
    top_labels = [AIPlayer.get_action_label(a) for a in sorted_actions[:5]]

    return PrincipleResult(
        name="B: Target 20",
        passed=has_20_in_top2,
        details=(
            f"Top 5 opening actions: {top_labels}; "
            f"20 in top 2: {'PASS' if has_20_in_top2 else 'FAIL'}"
        ),
        evidence={
            "top_5_actions": top_labels,
            "top_2_targets": top_2_targets,
            "q_values_top5": {
                AIPlayer.get_action_label(a): q_vals[a] for a in sorted_actions[:5]
            },
        },
    )


def _test_principle_c(
    agent: AIPlayer,
    num_games: int,
    threshold: float,
    miss_enabled: bool = False,
    skill_profiles: Optional[List[SkillProfile]] = None,
) -> PrincipleResult:
    """Principle C: Never chase.

    Run Q-agent vs S10 (ChaseAndCover). Check Q-agent wins > threshold
    AND in a chase-test state (opponent has closed 20, we haven't),
    the agent doesn't choose to target 20.
    """
    s10 = ChaseAndCover(player_id=1)
    result = run_matchup(agent, s10, num_games, "Q-Agent", "S10-ChaseAndCover",
                         miss_enabled=miss_enabled, skill_profiles=skill_profiles)

    # Chase test: construct state where opponent closed 20, we haven't
    game = DartsCricketGame()
    idx_20 = TARGETS.index(20)
    game.marks[1][idx_20] = 3  # opponent closed 20
    # Leave our marks at 0 for 20
    chase_state = game.get_state(perspective=0)

    # Set epsilon to 0 for greedy action
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    action = agent.choose_action(chase_state)
    agent.epsilon = old_epsilon

    chosen_target, chosen_hit = AIPlayer.action_to_target_hit(action)
    no_chase = chosen_target != 20

    win_check = result.win_rate_a >= threshold
    passed = win_check and no_chase

    details_parts = []
    details_parts.append(
        f"Win rate vs S10: {result.win_rate_a:.3f} "
        f"({'PASS' if win_check else 'FAIL'}, threshold={threshold})"
    )
    details_parts.append(
        f"Chase test action: {AIPlayer.get_action_label(action)} "
        f"({'PASS' if no_chase else 'FAIL'}, should NOT target 20)"
    )

    return PrincipleResult(
        name="C: Never Chase",
        passed=passed,
        details="; ".join(details_parts),
        evidence={
            "win_rate_vs_s10": result.win_rate_a,
            "chase_test_action": AIPlayer.get_action_label(action),
            "chase_test_target": chosen_target,
            "matchup": result,
        },
    )


def _test_principle_d(
    q_table_path: str,
    model_type: str = "qtable",
) -> PrincipleResult:
    """Principle D: Skill adaptation.

    For qtable: Load two Q-tables (p1 and p2) if available and compare sizes.
    For DQN: Load two models and verify both produce non-trivial Q-values.
    If only one model is available, report as inconclusive.
    """
    # Try to find both models
    dir_path = os.path.dirname(q_table_path)
    base_name = os.path.basename(q_table_path)

    p1_path = q_table_path
    p2_path = None
    if "p1" in base_name:
        p2_candidate = os.path.join(dir_path, base_name.replace("p1", "p2"))
        if os.path.exists(p2_candidate):
            p2_path = p2_candidate
    elif "p2" in base_name:
        p1_candidate = os.path.join(dir_path, base_name.replace("p2", "p1"))
        if os.path.exists(p1_candidate):
            p1_path = p1_candidate
            p2_path = q_table_path

    if p2_path is None:
        return PrincipleResult(
            name="D: Skill Adaptation",
            passed=False,
            details="Only one model found; cannot compare skill adaptation.",
            evidence={"q_table_path": q_table_path},
        )

    if model_type in ("dqn", "a2c"):
        if model_type == "dqn":
            from dqn_agent import DQNPlayer
            agent_p1 = DQNPlayer(player_id=0, epsilon=0.0)
            agent_p2 = DQNPlayer(player_id=1, epsilon=0.0)
        else:
            from a2c_agent import A2CPlayer
            agent_p1 = A2CPlayer(player_id=0, epsilon=0.0)
            agent_p2 = A2CPlayer(player_id=1, epsilon=0.0)
        agent_p1.load_q_table(p1_path)
        agent_p2.load_q_table(p2_path)

        # Check that both models produce varied Q-values (not all zeros)
        game = DartsCricketGame()
        starting_state = game.get_state(perspective=0)
        q_vals_p1 = agent_p1.get_starting_state_q_values(starting_state)
        q_vals_p2 = agent_p2.get_starting_state_q_values(starting_state)
        # Filter out -inf values for A2C (Bull+triple)
        p1_finite = [v for v in q_vals_p1.values() if v != float("-inf")]
        p2_finite = [v for v in q_vals_p2.values() if v != float("-inf")]
        p1_range = max(p1_finite) - min(p1_finite) if p1_finite else 0
        p2_range = max(p2_finite) - min(p2_finite) if p2_finite else 0
        both_learned = p1_range > 0.01 and p2_range > 0.01

        return PrincipleResult(
            name="D: Skill Adaptation",
            passed=both_learned,
            details=(
                f"P1 Q-value range: {p1_range:.4f}, P2 Q-value range: {p2_range:.4f}. "
                f"Both have non-trivial Q-values: {'PASS' if both_learned else 'FAIL'}"
            ),
            evidence={
                "p1_q_range": p1_range,
                "p2_q_range": p2_range,
                "p1_path": p1_path,
                "p2_path": p2_path,
            },
        )

    agent_p1 = AIPlayer(player_id=0, epsilon=0.0)
    agent_p1.load_q_table(p1_path)
    agent_p2 = AIPlayer(player_id=1, epsilon=0.0)
    agent_p2.load_q_table(p2_path)

    # Compare Q-table sizes as proxy for learning
    p1_size = len(agent_p1.q_table)
    p2_size = len(agent_p2.q_table)
    both_learned = p1_size > 100 and p2_size > 100

    return PrincipleResult(
        name="D: Skill Adaptation",
        passed=both_learned,
        details=(
            f"P1 Q-table: {p1_size} states, P2 Q-table: {p2_size} states. "
            f"Both have >100 states: {'PASS' if both_learned else 'FAIL'}"
        ),
        evidence={
            "p1_q_table_size": p1_size,
            "p2_q_table_size": p2_size,
            "p1_path": p1_path,
            "p2_path": p2_path,
        },
    )


# ── Main validation entry point ──────────────────────────────────────


def _load_agent(q_table_path: str, model_type: str = "qtable"):
    """Load an agent from a Q-table, DQN, or A2C model file."""
    if model_type == "dqn":
        from dqn_agent import DQNPlayer
        agent = DQNPlayer(player_id=0, epsilon=0.0)
        agent.load_q_table(q_table_path)
        logger.info("Loaded DQN model from %s", q_table_path)
        return agent
    if model_type == "a2c":
        from a2c_agent import A2CPlayer
        agent = A2CPlayer(player_id=0, epsilon=0.0)
        agent.load_q_table(q_table_path)
        logger.info("Loaded A2C model from %s", q_table_path)
        return agent
    agent = AIPlayer(player_id=0, epsilon=0.0)
    agent.load_q_table(q_table_path)
    logger.info("Loaded Q-table from %s (%d states)", q_table_path, len(agent.q_table))
    return agent


def validate_frongello_principles(
    q_table_path: str,
    num_games: int = VALIDATION_GAMES_PER_MATCHUP,
    output_dir: str = "output",
    threshold: float = FRONGELLO_WIN_RATE_THRESHOLD,
    model_type: str = "qtable",
    miss_enabled: bool = False,
    skill_profiles: Optional[List[SkillProfile]] = None,
) -> ValidationReport:
    """Load a Q-table or DQN model, run all 4 Frongello principle tests, return report."""
    os.makedirs(output_dir, exist_ok=True)

    # Load agent
    agent = _load_agent(q_table_path, model_type)

    # Run principle tests
    logger.info("Testing Principle A: Score First (%d games)...", num_games)
    principle_a = _test_principle_a(agent, num_games, threshold, miss_enabled=miss_enabled, skill_profiles=skill_profiles)
    logger.info("  Result: %s — %s", "PASS" if principle_a.passed else "FAIL", principle_a.details)

    logger.info("Testing Principle B: Target 20...")
    principle_b = _test_principle_b(agent, threshold)
    logger.info("  Result: %s — %s", "PASS" if principle_b.passed else "FAIL", principle_b.details)

    logger.info("Testing Principle C: Never Chase (%d games)...", num_games)
    principle_c = _test_principle_c(agent, num_games, threshold, miss_enabled=miss_enabled, skill_profiles=skill_profiles)
    logger.info("  Result: %s — %s", "PASS" if principle_c.passed else "FAIL", principle_c.details)

    logger.info("Testing Principle D: Skill Adaptation...")
    principle_d = _test_principle_d(q_table_path, model_type=model_type)
    logger.info("  Result: %s — %s", "PASS" if principle_d.passed else "FAIL", principle_d.details)

    principles = [principle_a, principle_b, principle_c, principle_d]

    # Run strategy bot matchups for additional context
    matchups: List[MatchupResult] = []
    strategy_bots = {
        "S1": SequentialCloser(player_id=1),
        "S2": LeadThenCover(player_id=1),
        "S10": ChaseAndCover(player_id=1),
    }

    for name, bot in strategy_bots.items():
        logger.info("Running Q-Agent vs %s (%d games)...", name, num_games)
        m = run_matchup(agent, bot, num_games, "Q-Agent", name, miss_enabled=miss_enabled, skill_profiles=skill_profiles)
        matchups.append(m)
        logger.info("  Q-Agent win rate: %.3f", m.win_rate_a)

    # Build summary
    passed_count = sum(1 for p in principles if p.passed)
    total_count = len(principles)
    summary_lines = [
        f"Frongello Validation: {passed_count}/{total_count} principles passed",
        "",
    ]
    for p in principles:
        status = "PASS" if p.passed else "FAIL"
        summary_lines.append(f"  [{status}] {p.name}: {p.details}")

    summary_lines.append("")
    summary_lines.append("Strategy Matchups:")
    for m in matchups:
        summary_lines.append(
            f"  {m.player_a_name} vs {m.player_b_name}: "
            f"{m.win_rate_a:.3f} / {m.win_rate_b:.3f}"
        )

    summary = "\n".join(summary_lines)
    logger.info("\n%s", summary)

    return ValidationReport(
        principles=principles,
        matchups=matchups,
        summary=summary,
    )
