"""Full 22x22 round-robin tournament across ALL skill pairings (unequal skill).

For each pair of MPR levels (p1_mpr, p2_mpr), runs every strategy combination.
Unlike the equal-skill tournament, the matrix is NOT symmetric — P1 at MPR 5.6
using S2 vs P2 at MPR 3.0 using S6 is a different matchup from the reverse.

Output: docs/data/unequal/tournament_{p1_mpr}v{p2_mpr}.json

Usage:
    # Run everything (121 pairings, ~53k matchups, ~12-18h on 20 threads)
    python run_unequal_tournament.py

    # Run a single pairing
    python run_unequal_tournament.py --p1-mpr 5.6 --p2-mpr 3.0

    # Skip equal-skill pairings (already have them)
    python run_unequal_tournament.py --skip-equal

    # Fewer games for a quick test
    python run_unequal_tournament.py --games 1000

    # Control parallelism
    python run_unequal_tournament.py --workers 16
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any, Dict, List, Tuple

from config import TARGETS, SkillProfile
from game import DartsCricketGame
from mpr_sweep import compute_mpr, make_profile_for_mpr
from refined_grid import PhaseSwitchCombo
from strategies import STRATEGY_CLASSES, EXPERIMENTAL_CLASSES, FrongelloStrategy


# ── Constants ─────────────────────────────────────────────────────────

MPR_LEVELS = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.6, 4.0, 4.9, 5.6]

STRATEGY_NAMES = [
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
    "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
    "E1", "E2", "E3", "E4",
    "PS",
]

NUM_STRATEGIES = len(STRATEGY_NAMES)  # 22

OUT_DIR = os.path.join("docs", "data", "unequal")


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


# ── Single matchup runner (picklable for multiprocessing) ────────────


def run_single_matchup(args: Tuple) -> Dict:
    """Run one matchup and return results. Designed for ProcessPoolExecutor."""
    (
        strat_a_name, strat_b_name,
        row_idx, col_idx,
        profile_a_dict, profile_b_dict,
        num_games,
        bull_mult,
    ) = args

    # Patch bull difficulty in worker process
    import config as _cfg
    import game as _game
    _cfg.BULL_DIFFICULTY_MULTIPLIER = bull_mult
    _game.BULL_DIFFICULTY_MULTIPLIER = bull_mult

    # Reconstruct SkillProfiles from dicts (can't pickle SkillProfile directly
    # if it's a dataclass, but our SkillProfile is simple enough)
    profile_a = SkillProfile(**profile_a_dict)
    profile_b = SkillProfile(**profile_b_dict)
    skill_profiles = [profile_a, profile_b]

    player_a = make_strategy(strat_a_name, player_id=0)
    player_b = make_strategy(strat_b_name, player_id=1)

    game = DartsCricketGame(skill_profiles=skill_profiles)
    wins_a = 0
    wins_b = 0
    total_darts = 0
    total_turns = 0
    total_marks_p1 = 0
    total_marks_p2 = 0
    players = [player_a, player_b]

    for game_num in range(num_games):
        game.reset()
        if game_num % 2 == 1:
            game.current_player = 1
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

        total_darts += game.total_darts_thrown
        total_turns += turns

        for i in range(len(TARGETS)):
            total_marks_p1 += game.marks[0][i]
            total_marks_p2 += game.marks[1][i]

    win_rate_a = wins_a / max(1, num_games)
    avg_turns = total_turns / max(1, num_games)
    avg_darts = total_darts / max(1, num_games)

    total_rounds = total_darts / 3.0
    total_marks = total_marks_p1 + total_marks_p2
    empirical_mpr = total_marks / max(1, total_rounds) if total_rounds > 0 else 0.0

    return {
        "row": row_idx,
        "col": col_idx,
        "win_rate_a": round(win_rate_a * 100, 1),
        "avg_turns": avg_turns,
        "avg_darts": avg_darts,
        "empirical_mpr": empirical_mpr,
    }


# ── Tournament for one skill pairing ────────────────────────────────


def run_pairing(
    p1_mpr: float,
    p2_mpr: float,
    num_games: int,
    max_workers: int,
    bull_mult: float = 0.75,
) -> Dict:
    """Run full 22x22 tournament for one skill pairing. Returns result dict."""

    profile_a = make_profile_for_mpr(p1_mpr)
    profile_b = make_profile_for_mpr(p2_mpr)

    # Convert to dicts for pickling across processes
    profile_a_dict = {
        "name": profile_a.name,
        "triple_outcomes": dict(profile_a.triple_outcomes),
        "double_outcomes": dict(profile_a.double_outcomes),
        "single_outcomes": dict(profile_a.single_outcomes),
    }
    profile_b_dict = {
        "name": profile_b.name,
        "triple_outcomes": dict(profile_b.triple_outcomes),
        "double_outcomes": dict(profile_b.double_outcomes),
        "single_outcomes": dict(profile_b.single_outcomes),
    }

    n = NUM_STRATEGIES
    is_equal = (p1_mpr == p2_mpr)

    # For equal skill, only compute upper triangle (symmetric).
    # For unequal skill, compute ALL n*n cells (not symmetric).
    if is_equal:
        matchup_list = []
        for i in range(n):
            for j in range(i + 1, n):
                matchup_list.append((
                    STRATEGY_NAMES[i], STRATEGY_NAMES[j],
                    i, j,
                    profile_a_dict, profile_b_dict,
                    num_games,
                    bull_mult,
                ))
        total_matchups = len(matchup_list)
    else:
        matchup_list = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Same strategy, different skill — still interesting!
                    pass  # include it
                matchup_list.append((
                    STRATEGY_NAMES[i], STRATEGY_NAMES[j],
                    i, j,
                    profile_a_dict, profile_b_dict,
                    num_games,
                    bull_mult,
                ))
        total_matchups = len(matchup_list)

    # Initialize matrix
    matrix = [[0.0] * n for _ in range(n)]

    print(f"\n{'=' * 80}")
    print(f"P1 MPR {p1_mpr:.1f} vs P2 MPR {p2_mpr:.1f} — "
          f"{'EQUAL' if is_equal else 'UNEQUAL'} — "
          f"{total_matchups} matchups")
    print(f"{'=' * 80}")

    t0 = time.time()
    completed = 0

    total_avg_turns = 0.0
    total_avg_darts = 0.0
    total_empirical_mpr = 0.0

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_matchup, m): m for m in matchup_list}

        for future in as_completed(futures):
            result = future.result()
            r, c = result["row"], result["col"]

            if is_equal:
                matrix[r][c] = result["win_rate_a"]
                matrix[c][r] = round(100.0 - result["win_rate_a"], 1)
            else:
                matrix[r][c] = result["win_rate_a"]

            total_avg_turns += result["avg_turns"]
            total_avg_darts += result["avg_darts"]
            total_empirical_mpr += result["empirical_mpr"]

            completed += 1
            if completed % 50 == 0 or completed == total_matchups:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_matchups - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total_matchups}] "
                      f"{elapsed:.0f}s elapsed, "
                      f"~{eta:.0f}s remaining")

    # For equal skill, set diagonal to 50
    if is_equal:
        for i in range(n):
            matrix[i][i] = 50.0

    # Compute averages per strategy
    averages_p1 = {}  # P1's avg win rate using each strategy (across all P2 strategies)
    averages_p2 = {}  # P2's avg win rate using each strategy (across all P1 strategies)

    for i, name in enumerate(STRATEGY_NAMES):
        # Row average: strategy i as P1, across all P2 strategies
        if is_equal:
            row_vals = [matrix[i][j] for j in range(n) if j != i]
        else:
            row_vals = [matrix[i][j] for j in range(n)]
        averages_p1[name] = round(sum(row_vals) / len(row_vals), 1) if row_vals else 50.0

        # Column average: strategy i as P2, across all P1 strategies
        if is_equal:
            col_vals = [100.0 - matrix[j][i] for j in range(n) if j != i]
        else:
            col_vals = [100.0 - matrix[j][i] for j in range(n)]
        averages_p2[name] = round(sum(col_vals) / len(col_vals), 1) if col_vals else 50.0

    # Rankings for P1 (stronger/weaker player depending on pairing)
    rankings_p1 = sorted(
        [{"name": name, "avg": avg} for name, avg in averages_p1.items()],
        key=lambda x: x["avg"],
        reverse=True,
    )
    rankings_p2 = sorted(
        [{"name": name, "avg": avg} for name, avg in averages_p2.items()],
        key=lambda x: x["avg"],
        reverse=True,
    )

    overall_avg_turns = total_avg_turns / max(1, total_matchups)
    overall_avg_darts = total_avg_darts / max(1, total_matchups)
    overall_empirical_mpr = total_empirical_mpr / max(1, total_matchups)

    elapsed_total = time.time() - t0

    result_dict = {
        "p1_mpr_target": p1_mpr,
        "p2_mpr_target": p2_mpr,
        "p1_mpr_empirical": round(compute_mpr(profile_a), 2),
        "p2_mpr_empirical": round(compute_mpr(profile_b), 2),
        "is_equal_skill": is_equal,
        "games_per_matchup": num_games,
        "total_matchups": total_matchups,
        "avg_turns": round(overall_avg_turns, 1),
        "avg_darts": round(overall_avg_darts, 1),
        "strategies": STRATEGY_NAMES,
        "matrix": matrix,
        "averages_p1": averages_p1,
        "averages_p2": averages_p2,
        "rankings_p1": rankings_p1,
        "rankings_p2": rankings_p2,
        "elapsed_seconds": round(elapsed_total, 1),
    }

    # Save JSON
    os.makedirs(OUT_DIR, exist_ok=True)
    filename = f"tournament_{p1_mpr}v{p2_mpr}.json"
    out_path = os.path.join(OUT_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print top-5 rankings
    print(f"\n--- P1 (MPR {p1_mpr}) Best Strategies ---")
    for rank, entry in enumerate(rankings_p1[:5], 1):
        print(f"  {rank}. {entry['name']:>4}  {entry['avg']:.1f}%")

    if not is_equal:
        print(f"\n--- P2 (MPR {p2_mpr}) Best Strategies ---")
        for rank, entry in enumerate(rankings_p2[:5], 1):
            print(f"  {rank}. {entry['name']:>4}  {entry['avg']:.1f}%")

    print(f"\nCompleted in {elapsed_total:.0f}s")

    return result_dict


# ── Main ──────────────────────────────────────────────────────────────


def main():
    logging.disable(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run full 22x22 tournament across all skill pairings (unequal skill)"
    )
    parser.add_argument(
        "--p1-mpr", type=float, default=None,
        help="P1 MPR level (run single pairing with --p2-mpr)"
    )
    parser.add_argument(
        "--p2-mpr", type=float, default=None,
        help="P2 MPR level (run single pairing with --p1-mpr)"
    )
    parser.add_argument(
        "--games", type=int, default=20000,
        help="Games per matchup (default: 20000)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--skip-equal", action="store_true",
        help="Skip equal-skill pairings (diagonal)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip pairings that already have output files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: docs/data/unequal)"
    )
    parser.add_argument(
        "--bull-mult", type=float, default=0.75,
        help="Bull difficulty multiplier (default: 0.75)"
    )
    args = parser.parse_args()

    max_workers = args.workers or os.cpu_count()

    # Override output directory if specified
    global OUT_DIR
    if args.output_dir:
        OUT_DIR = args.output_dir

    # Patch bull difficulty multiplier in main process
    import config as _cfg
    import game as _game
    _cfg.BULL_DIFFICULTY_MULTIPLIER = args.bull_mult
    _game.BULL_DIFFICULTY_MULTIPLIER = args.bull_mult

    # Determine which pairings to run
    if args.p1_mpr is not None and args.p2_mpr is not None:
        pairings = [(args.p1_mpr, args.p2_mpr)]
    elif args.p1_mpr is not None or args.p2_mpr is not None:
        parser.error("Must specify both --p1-mpr and --p2-mpr, or neither")
    else:
        # All 121 pairings (11 x 11)
        pairings = list(product(MPR_LEVELS, MPR_LEVELS))

    if args.skip_equal:
        pairings = [(a, b) for a, b in pairings if a != b]

    # Resume: skip pairings with existing output
    if args.resume:
        os.makedirs(OUT_DIR, exist_ok=True)
        remaining = []
        for p1, p2 in pairings:
            filename = f"tournament_{p1}v{p2}.json"
            path = os.path.join(OUT_DIR, filename)
            if os.path.exists(path):
                print(f"  Skipping {p1} vs {p2} (already exists)")
            else:
                remaining.append((p1, p2))
        pairings = remaining

    num_pairings = len(pairings)
    total_matchups_est = sum(
        NUM_STRATEGIES * (NUM_STRATEGIES - 1) // 2 if p1 == p2
        else NUM_STRATEGIES * NUM_STRATEGIES
        for p1, p2 in pairings
    )
    total_games_est = total_matchups_est * args.games

    print(f"Unequal-Skill 22x22 Round-Robin Tournament")
    print(f"{'=' * 60}")
    print(f"Bull multiplier:  {args.bull_mult}")
    print(f"Strategies:       {', '.join(STRATEGY_NAMES)}")
    print(f"Skill pairings:   {num_pairings}")
    print(f"Games per matchup: {args.games:,}")
    print(f"Total matchups:   {total_matchups_est:,}")
    print(f"Total games:      {total_games_est:,}")
    print(f"Workers:          {max_workers}")
    print(f"Output dir:       {OUT_DIR}")
    print(f"{'=' * 60}")

    t_start = time.time()

    for idx, (p1_mpr, p2_mpr) in enumerate(pairings, 1):
        print(f"\n>>> Pairing {idx}/{num_pairings}: "
              f"P1 MPR {p1_mpr} vs P2 MPR {p2_mpr}")
        run_pairing(p1_mpr, p2_mpr, args.games, max_workers, args.bull_mult)

    total_elapsed = time.time() - t_start
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — {num_pairings} pairings in {hours}h {minutes}m {seconds}s")
    print(f"Output: {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
