"""Full 22x22 round-robin tournament at 11 MPR skill levels.

Strategies:
  S1-S17  — All 17 Frongello strategies
  E1-E4   — Experimental strategies (EarlyBull, Honeypot, GreedyClose, Adaptive)
  PS      — PhaseSwitchCombo (threshold_mult=13)

For each MPR level, runs every ordered pair once (upper triangle + mirror),
saves results as JSON to docs/data/tournament_mpr_{mpr}.json.
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List

from config import SKILL_PROFILES, TARGETS, SkillProfile
from game import DartsCricketGame
from mpr_sweep import compute_mpr, make_profile_for_mpr, run_matchup_with_stats
from refined_grid import PhaseSwitchCombo
from strategies import STRATEGY_CLASSES, EXPERIMENTAL_CLASSES, FrongelloStrategy
from validation import run_matchup


# ── Constants ─────────────────────────────────────────────────────────

MPR_LEVELS = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.6, 4.0, 4.9, 5.6]

STRATEGY_NAMES = [
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
    "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
    "E1", "E2", "E3", "E4",
    "PS",
]

NUM_STRATEGIES = len(STRATEGY_NAMES)  # 22


# ── Strategy factory ─────────────────────────────────────────────────


def make_strategy(name: str, player_id: int) -> Any:
    """Instantiate a strategy bot by name."""
    if name == "PS":
        return PhaseSwitchCombo(player_id=player_id, threshold_mult=13)
    elif name in STRATEGY_CLASSES:
        return STRATEGY_CLASSES[name](player_id=player_id)
    elif name in EXPERIMENTAL_CLASSES:
        return EXPERIMENTAL_CLASSES[name](player_id=player_id)
    else:
        raise ValueError(f"Unknown strategy: {name}")


# ── Tournament runner ─────────────────────────────────────────────────


def run_tournament_at_mpr(
    target_mpr: float,
    num_games: int,
) -> Dict:
    """Run the full 22x22 round-robin at one MPR level. Returns result dict."""

    # Build skill profile
    profile = make_profile_for_mpr(target_mpr)
    theoretical_mpr = compute_mpr(profile)
    skill_profiles = [profile, profile]

    n = NUM_STRATEGIES
    # Number of upper-triangle matchups: n*(n-1)/2
    num_matchups = n * (n - 1) // 2

    # Initialize matrix with 50.0 on diagonal
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 50.0

    # Track aggregate stats
    total_avg_turns = 0.0
    total_avg_darts = 0.0
    total_empirical_mpr = 0.0
    matchup_count = 0

    print(f"\n{'=' * 80}")
    print(f"MPR {target_mpr:.1f} — Profile: {profile.name} — "
          f"Theoretical MPR: {theoretical_mpr:.2f}")
    print(f"{'=' * 80}")

    # Run upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            matchup_count += 1
            name_a = STRATEGY_NAMES[i]
            name_b = STRATEGY_NAMES[j]

            t0 = time.time()

            player_a = make_strategy(name_a, player_id=0)
            player_b = make_strategy(name_b, player_id=1)

            win_rate_a, win_rate_b, avg_turns, avg_darts, empirical_mpr = (
                run_matchup_with_stats(
                    player_a, player_b, num_games,
                    player_a_name=name_a,
                    player_b_name=name_b,
                    skill_profiles=skill_profiles,
                )
            )

            elapsed = time.time() - t0

            # Fill matrix
            matrix[i][j] = round(win_rate_a * 100, 1)
            matrix[j][i] = round((1.0 - win_rate_a) * 100, 1)

            # Accumulate stats
            total_avg_turns += avg_turns
            total_avg_darts += avg_darts
            total_empirical_mpr += empirical_mpr

            print(f"MPR {target_mpr:.1f} — Matchup {matchup_count}/{num_matchups}: "
                  f"{name_a} vs {name_b} ... {win_rate_a * 100:.1f}% ({elapsed:.1f}s)")

    # Compute averages per strategy (excluding diagonal)
    averages = {}
    for i, name in enumerate(STRATEGY_NAMES):
        row_vals = [matrix[i][j] for j in range(n) if j != i]
        averages[name] = round(sum(row_vals) / len(row_vals), 1)

    # Build rankings sorted by average descending
    rankings = sorted(
        [{"name": name, "avg": avg} for name, avg in averages.items()],
        key=lambda x: x["avg"],
        reverse=True,
    )

    # Aggregate stats across all matchups
    overall_avg_turns = total_avg_turns / max(1, matchup_count)
    overall_avg_darts = total_avg_darts / max(1, matchup_count)
    overall_empirical_mpr = total_empirical_mpr / max(1, matchup_count)

    result = {
        "mpr_target": target_mpr,
        "mpr_empirical": round(overall_empirical_mpr, 2),
        "skill_profile_name": profile.name,
        "games_per_matchup": num_games,
        "avg_turns": round(overall_avg_turns, 1),
        "avg_darts": round(overall_avg_darts, 1),
        "strategies": STRATEGY_NAMES,
        "matrix": matrix,
        "averages": averages,
        "rankings": rankings,
    }

    # Save JSON
    out_dir = os.path.join("docs", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tournament_mpr_{target_mpr}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print summary table
    print(f"\n--- Rankings at MPR {target_mpr:.1f} ---")
    print(f"{'Rank':>4}  {'Strategy':>8}  {'Avg Win%':>8}")
    print("-" * 26)
    for rank, entry in enumerate(rankings, 1):
        print(f"{rank:>4}  {entry['name']:>8}  {entry['avg']:>7.1f}%")
    print()

    return result


# ── Main ──────────────────────────────────────────────────────────────


def main():
    logging.disable(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run full 22x22 tournament at all MPR levels"
    )
    parser.add_argument(
        "--mpr", type=float, help="Run single MPR level"
    )
    parser.add_argument(
        "--games", type=int, default=20000, help="Games per matchup"
    )
    args = parser.parse_args()

    # Determine which MPR levels to run
    if args.mpr is not None:
        mpr_levels = [args.mpr]
    else:
        mpr_levels = MPR_LEVELS

    num_games = args.games

    print(f"Full 22x22 Round-Robin Tournament")
    print(f"Strategies: {', '.join(STRATEGY_NAMES)}")
    print(f"Games per matchup: {num_games:,}")
    print(f"MPR levels: {mpr_levels}")
    print(f"Matchups per level: {NUM_STRATEGIES * (NUM_STRATEGIES - 1) // 2}")

    t_start = time.time()

    for mpr in mpr_levels:
        run_tournament_at_mpr(mpr, num_games)

    total_elapsed = time.time() - t_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    print(f"\nTotal elapsed time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
