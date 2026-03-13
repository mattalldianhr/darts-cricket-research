"""MPR sweep: Phase Switch (13x combo) vs S2 across a wide range of skill levels.

Also includes S2 vs S1 at each level for reference.

Creates synthetic SkillProfile objects by uniformly scaling ALL hit probabilities
from the amateur profile, redistributing freed probability mass to miss.
"""

import logging
import sys
import time
from typing import Dict, List, Tuple

from config import (
    SKILL_PROFILES,
    SkillProfile,
    SKILL_PROFILE_AMATEUR,
    SKILL_PROFILE_GOOD,
    SKILL_PROFILE_PRO,
    TARGETS,
)
from game import DartsCricketGame
from refined_grid import PhaseSwitchCombo
from strategies import FrongelloStrategy, STRATEGY_CLASSES
from validation import run_matchup


# ── Helpers ──────────────────────────────────────────────────────────


def compute_mpr(profile: SkillProfile) -> float:
    """Compute the theoretical MPR (marks per round) for a profile aiming triples.

    MPR = 3 * marks_per_dart_when_aiming_triple
    marks_per_dart = T*3 + D*2 + S*1 + M*0
    """
    t = profile.triple_outcomes
    mpd = t.get("triple", 0) * 3 + t.get("double", 0) * 2 + t.get("single", 0) * 1
    return mpd * 3


def scale_profile_uniform(base: SkillProfile, factor: float, name: str = "") -> SkillProfile:
    """Scale ALL hit probabilities uniformly by factor, redistributing to miss.

    Unlike config.scale_profile which only scales the primary hit type,
    this scales every non-miss outcome proportionally and gives the freed
    mass to miss. This preserves the relative mix of outcomes.

    factor=1.0 returns the original profile.
    factor=0.5 halves all hit probabilities.
    factor>1.0 boosts all hit probabilities (capped so miss >= 0).
    """
    def _scale_outcomes(outcomes: Dict[str, float]) -> Dict[str, float]:
        scaled = {}
        miss = outcomes.get("miss", 0.0)
        # Scale all non-miss outcomes
        for key, val in outcomes.items():
            if key == "miss":
                continue
            new_val = val * factor
            scaled[key] = new_val
        # Compute new miss = whatever is left
        total_hits = sum(scaled.values())
        new_miss = 1.0 - total_hits
        if new_miss < 0:
            # Cap: can't have negative miss. Normalize hits to sum to 1.0.
            for key in scaled:
                scaled[key] /= total_hits
            new_miss = 0.0
        scaled["miss"] = new_miss
        return scaled

    label = name or f"{base.name}_x{factor:.3f}"
    return SkillProfile(
        name=label,
        triple_outcomes=_scale_outcomes(dict(base.triple_outcomes)),
        double_outcomes=_scale_outcomes(dict(base.double_outcomes)),
        single_outcomes=_scale_outcomes(dict(base.single_outcomes)),
    )


def make_profile_for_mpr(target_mpr: float) -> SkillProfile:
    """Create a SkillProfile that achieves approximately the target MPR.

    Strategy: pick the closest built-in profile as base, compute its MPR,
    then uniformly scale to hit the target.
    """
    # Built-in profiles with their MPRs
    builtins = [
        (SKILL_PROFILE_AMATEUR, compute_mpr(SKILL_PROFILE_AMATEUR)),
        (SKILL_PROFILE_GOOD, compute_mpr(SKILL_PROFILE_GOOD)),
        (SKILL_PROFILE_PRO, compute_mpr(SKILL_PROFILE_PRO)),
    ]

    # Find closest base profile
    best_base = None
    best_mpr = None
    best_dist = float("inf")
    for profile, mpr in builtins:
        dist = abs(mpr - target_mpr)
        if dist < best_dist:
            best_dist = dist
            best_base = profile
            best_mpr = mpr

    # If we're very close to a built-in, just use it
    if abs(best_mpr - target_mpr) < 0.05:
        return best_base

    # Compute scale factor
    factor = target_mpr / best_mpr
    label = f"MPR{target_mpr:.1f}"
    return scale_profile_uniform(best_base, factor, name=label)


def run_matchup_with_stats(
    player_a,
    player_b,
    num_games: int,
    player_a_name: str,
    player_b_name: str,
    skill_profiles: List[SkillProfile],
) -> Tuple[float, float, float, float]:
    """Run a matchup and return (win_rate_a, win_rate_b, avg_turns, empirical_mpr).

    We need per-game stats so we run manually instead of using validation.run_matchup.
    """
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

        # Count total marks achieved by both players (uncapped, matching real-world MPR measurement)
        for i in range(len(TARGETS)):
            total_marks_p1 += game.marks[0][i]
            total_marks_p2 += game.marks[1][i]

    win_rate_a = wins_a / max(1, num_games)
    win_rate_b = wins_b / max(1, num_games)
    avg_turns = total_turns / max(1, num_games)
    avg_darts = total_darts / max(1, num_games)

    # Empirical MPR: total marks / total rounds (a round = 3 darts)
    total_rounds = total_darts / 3.0
    total_marks = total_marks_p1 + total_marks_p2
    empirical_mpr = total_marks / max(1, total_rounds) if total_rounds > 0 else 0.0

    return win_rate_a, win_rate_b, avg_turns, avg_darts, empirical_mpr


def main():
    logging.disable(logging.INFO)

    NUM_GAMES = 20_000

    # Target MPRs to sweep
    target_mprs = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.6, 4.0, 4.9, 5.6]

    # Build profiles
    profiles = {}
    for mpr in target_mprs:
        p = make_profile_for_mpr(mpr)
        actual_theoretical = compute_mpr(p)
        profiles[mpr] = (p, actual_theoretical)

    # Print profile summary
    print("=" * 80)
    print("MPR SWEEP: Phase Switch (13x combo) vs S2 and S2 vs S1")
    print(f"Games per matchup: {NUM_GAMES:,}")
    print("=" * 80)
    print()
    print("Profile summary:")
    print(f"  {'Target MPR':>12}  {'Base':>10}  {'Theoretical':>12}  {'Triple%':>8}  {'Double%':>8}  {'Single%':>8}  {'Miss%':>8}")
    print("-" * 88)
    for mpr in target_mprs:
        p, theoretical = profiles[mpr]
        t = p.triple_outcomes
        print(f"  {mpr:>10.1f}    {p.name:>10}  {theoretical:>10.2f}    "
              f"{t.get('triple',0)*100:>6.1f}%  {t.get('double',0)*100:>6.1f}%  "
              f"{t.get('single',0)*100:>6.1f}%  {t.get('miss',0)*100:>6.1f}%")
    print()

    # Run all matchups
    results = []
    total_matchups = len(target_mprs) * 2  # PhaseSwitch vs S2 + S2 vs S1
    matchup_num = 0

    print("Running matchups...")
    print("-" * 80)

    for mpr in target_mprs:
        p, theoretical = profiles[mpr]
        skill_profiles = [p, p]

        # PhaseSwitch (13x combo) vs S2
        matchup_num += 1
        t0 = time.time()
        ps = PhaseSwitchCombo(player_id=0, threshold_mult=13)
        s2 = STRATEGY_CLASSES["S2"](player_id=1)
        ps_wr, s2_wr_vs_ps, ps_turns, ps_darts, ps_emp_mpr = run_matchup_with_stats(
            ps, s2, NUM_GAMES, "PS(13x,combo)", "S2", skill_profiles
        )
        elapsed1 = time.time() - t0

        # S2 vs S1
        matchup_num += 1
        t0 = time.time()
        s2_a = STRATEGY_CLASSES["S2"](player_id=0)
        s1 = STRATEGY_CLASSES["S1"](player_id=1)
        s2_wr_vs_s1, s1_wr, s2_turns, s2_darts, s2_emp_mpr = run_matchup_with_stats(
            s2_a, s1, NUM_GAMES, "S2", "S1", skill_profiles
        )
        elapsed2 = time.time() - t0

        results.append({
            "target_mpr": mpr,
            "theoretical_mpr": theoretical,
            "empirical_mpr_ps": ps_emp_mpr,
            "empirical_mpr_s2": s2_emp_mpr,
            "ps_win_rate": ps_wr,
            "s2_win_rate_vs_ps": s2_wr_vs_ps,
            "ps_avg_turns": ps_turns,
            "ps_avg_darts": ps_darts,
            "s2_win_rate_vs_s1": s2_wr_vs_s1,
            "s1_win_rate": s1_wr,
            "s2_avg_turns": s2_turns,
            "s2_avg_darts": s2_darts,
        })

        print(f"  [{matchup_num:2d}/{total_matchups}] MPR {mpr:.1f}  "
              f"PS vs S2: {ps_wr*100:5.1f}% / {s2_wr_vs_ps*100:5.1f}%  "
              f"S2 vs S1: {s2_wr_vs_s1*100:5.1f}% / {s1_wr*100:5.1f}%  "
              f"({elapsed1 + elapsed2:.1f}s)")

    # Print final table
    print()
    print("=" * 130)
    print("RESULTS")
    print("=" * 130)
    print()
    print(f"  {'MPR':>4}  {'MPR':>5}  {'Avg':>5}  {'Avg':>6}  "
          f"{'PhaseSwitch':>11}  {'S2':>6}  "
          f"{'S2':>6}  {'S1':>6}  "
          f"{'PS Edge':>8}  {'S2 Edge':>8}")
    print(f"  {'(tgt)':>4}  {'(emp)':>5}  {'Turns':>5}  {'Darts':>6}  "
          f"{'Win%':>11}  {'Win%':>6}  "
          f"{'Win%':>6}  {'Win%':>6}  "
          f"{'vs S2':>8}  {'vs S1':>8}")
    print("-" * 130)

    for r in results:
        ps_edge = r["ps_win_rate"] - r["s2_win_rate_vs_ps"]
        s2_edge = r["s2_win_rate_vs_s1"] - r["s1_win_rate"]
        emp_mpr = (r["empirical_mpr_ps"] + r["empirical_mpr_s2"]) / 2  # average of both matchups
        print(f"  {r['target_mpr']:>4.1f}  {emp_mpr:>5.2f}  {r['ps_avg_turns']:>5.1f}  {r['ps_avg_darts']:>6.1f}  "
              f"{r['ps_win_rate']*100:>10.1f}%  {r['s2_win_rate_vs_ps']*100:>5.1f}%  "
              f"{r['s2_win_rate_vs_s1']*100:>5.1f}%  {r['s1_win_rate']*100:>5.1f}%  "
              f"{ps_edge*100:>+7.1f}pp  {s2_edge*100:>+7.1f}pp")

    print()
    print("=" * 130)
    print()

    # Key insight summary
    print("KEY INSIGHTS:")
    print("-" * 60)

    # Find where PS advantage peaks and where it disappears
    max_ps_edge = 0
    max_ps_mpr = 0
    for r in results:
        edge = r["ps_win_rate"] - r["s2_win_rate_vs_ps"]
        if edge > max_ps_edge:
            max_ps_edge = edge
            max_ps_mpr = r["target_mpr"]

    print(f"  Phase Switch peak advantage: {max_ps_edge*100:+.1f}pp at MPR {max_ps_mpr:.1f}")

    # Find crossover point (where PS edge <= 0)
    crossover = None
    for r in results:
        edge = r["ps_win_rate"] - r["s2_win_rate_vs_ps"]
        if edge <= 0:
            crossover = r["target_mpr"]
            break

    if crossover:
        print(f"  Phase Switch loses advantage at MPR {crossover:.1f}")
    else:
        print("  Phase Switch maintains advantage across all tested MPRs")

    # S2 vs S1 trend
    max_s2_edge = 0
    max_s2_mpr = 0
    for r in results:
        edge = r["s2_win_rate_vs_s1"] - r["s1_win_rate"]
        if edge > max_s2_edge:
            max_s2_edge = edge
            max_s2_mpr = r["target_mpr"]
    print(f"  S2 vs S1 peak advantage: {max_s2_edge*100:+.1f}pp at MPR {max_s2_mpr:.1f}")

    # Game length trend
    shortest = min(results, key=lambda r: r["ps_avg_turns"])
    longest = min(results, key=lambda r: -r["ps_avg_turns"])
    print(f"  Shortest games: {shortest['ps_avg_turns']:.1f} turns at MPR {shortest['target_mpr']:.1f}")
    print(f"  Longest games: {longest['ps_avg_turns']:.1f} turns at MPR {longest['target_mpr']:.1f}")


if __name__ == "__main__":
    main()
