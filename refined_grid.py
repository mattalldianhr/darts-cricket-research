"""Refined phase-switch grid search: PhaseSwitch vs S2 (pro skill profile).

Explores finer threshold multipliers around the sweet spot and a combo
condition variant that requires BOTH unclosed <= 3 AND marks_remaining <= 9.
"""

import sys
import time
from typing import List, Optional, Tuple

from config import MARKS_TO_CLOSE, SKILL_PROFILES, TARGETS
from game import DartsCricketGame
from strategies import FrongelloStrategy, STRATEGY_CLASSES
from validation import run_matchup


class PhaseSwitch(FrongelloStrategy):
    """Phase-switching strategy: aggressive scoring early, S2 late.

    Uses a high lead_multiplier (threshold_mult x highest unblocked value)
    until the player has switch_at or fewer unclosed targets, then switches
    to S2 behavior (lead_multiplier=0).
    """

    def __init__(
        self,
        player_id: int,
        threshold_mult: int,
        switch_at: int,
    ) -> None:
        # Start with the high threshold
        super().__init__(
            player_id,
            lead_multiplier=threshold_mult,
            use_extra_darts=False,
            use_chase=False,
        )
        self._threshold_mult = threshold_mult
        self._switch_at = switch_at

    def _lead_below_threshold(self, game: DartsCricketGame) -> bool:
        """Override: check phase, then delegate to parent with the right multiplier."""
        my_unclosed = len(self._unclosed_targets(game, self.player_id))

        if my_unclosed <= self._switch_at:
            # Late phase: S2 behavior (threshold = 0)
            self.lead_multiplier = 0
        else:
            # Early phase: aggressive scoring
            self.lead_multiplier = self._threshold_mult

        return super()._lead_below_threshold(game)


class PhaseSwitchCombo(FrongelloStrategy):
    """Phase-switching with combo condition: switch when BOTH
    my_unclosed <= 3 AND my_marks_remaining <= 9.

    marks_remaining = total marks still needed to close all unclosed targets.
    """

    def __init__(
        self,
        player_id: int,
        threshold_mult: int,
    ) -> None:
        super().__init__(
            player_id,
            lead_multiplier=threshold_mult,
            use_extra_darts=False,
            use_chase=False,
        )
        self._threshold_mult = threshold_mult

    def _lead_below_threshold(self, game: DartsCricketGame) -> bool:
        """Override: switch to S2 when both conditions met."""
        my_unclosed = len(self._unclosed_targets(game, self.player_id))

        # Compute marks remaining to close all unclosed targets
        marks_remaining = 0
        for i, target in enumerate(TARGETS):
            marks_have = game.marks[self.player_id][i]
            if marks_have < MARKS_TO_CLOSE:
                marks_remaining += (MARKS_TO_CLOSE - marks_have)

        if my_unclosed <= 3 and marks_remaining <= 9:
            self.lead_multiplier = 0
        else:
            self.lead_multiplier = self._threshold_mult

        return super()._lead_below_threshold(game)


def run_grid():
    """Run the refined grid and print results."""
    NUM_GAMES = 50_000
    pro = SKILL_PROFILES["pro"]
    skill_profiles = [pro, pro]

    # Grid parameters
    thresholds = [10, 11, 12, 13, 14, 15, 16, 18, 20, 25]
    switch_ats = [2, 3, 4]

    # Results storage
    results = {}  # (threshold, switch_at) -> win_rate
    combo_results = {}  # threshold -> win_rate
    best_wr = 0.0
    best_key = None

    total_cells = len(thresholds) * len(switch_ats) + len(thresholds)
    cell_num = 0

    print(f"Refined Phase-Switch Grid Search vs S2 (pro skill, {NUM_GAMES:,} games/cell)")
    print(f"Total cells: {total_cells}")
    print("=" * 70)
    print()

    # Run the main grid
    for thresh in thresholds:
        for sw in switch_ats:
            cell_num += 1
            t0 = time.time()

            ps = PhaseSwitch(player_id=0, threshold_mult=thresh, switch_at=sw)
            s2 = STRATEGY_CLASSES["S2"](player_id=1)

            result = run_matchup(
                ps, s2, NUM_GAMES,
                player_a_name=f"PS({thresh}x,sw@{sw})",
                player_b_name="S2",
                skill_profiles=skill_profiles,
            )

            wr = result.win_rate_a
            results[(thresh, sw)] = wr
            elapsed = time.time() - t0

            if wr > best_wr:
                best_wr = wr
                best_key = (thresh, sw, "standard")

            print(f"  [{cell_num:2d}/{total_cells}] thresh={thresh:2d}x  sw@{sw}  "
                  f"=> {wr*100:5.1f}%  ({elapsed:.1f}s)")

    # Run combo condition grid
    print()
    print("-" * 70)
    print("Combo condition: switch when unclosed<=3 AND marks_remaining<=9")
    print("-" * 70)

    for thresh in thresholds:
        cell_num += 1
        t0 = time.time()

        ps = PhaseSwitchCombo(player_id=0, threshold_mult=thresh)
        s2 = STRATEGY_CLASSES["S2"](player_id=1)

        result = run_matchup(
            ps, s2, NUM_GAMES,
            player_a_name=f"PSCombo({thresh}x)",
            player_b_name="S2",
            skill_profiles=skill_profiles,
        )

        wr = result.win_rate_a
        combo_results[thresh] = wr
        elapsed = time.time() - t0

        if wr > best_wr:
            best_wr = wr
            best_key = (thresh, None, "combo")

        print(f"  [{cell_num:2d}/{total_cells}] thresh={thresh:2d}x  combo  "
              f"=> {wr*100:5.1f}%  ({elapsed:.1f}s)")

    # Print formatted grid
    print()
    print("=" * 70)
    print("RESULTS GRID (win rate % vs S2)")
    print("=" * 70)
    print()

    # Header
    header = f"{'thresh':>8}"
    for sw in switch_ats:
        header += f"  {'sw@'+str(sw):>7}"
    header += f"  {'combo':>7}"
    print(header)
    print("-" * (8 + (len(switch_ats) + 1) * 9))

    for thresh in thresholds:
        row = f"{thresh:>5}x  "
        for sw in switch_ats:
            wr = results[(thresh, sw)]
            marker = "*" if (thresh, sw, "standard") == best_key else " "
            row += f"  {wr*100:5.1f}%{marker}"

        cwr = combo_results[thresh]
        marker = "*" if (thresh, None, "combo") == best_key else " "
        row += f"  {cwr*100:5.1f}%{marker}"

        print(row)

    print()
    if best_key[2] == "standard":
        print(f"BEST: thresh={best_key[0]}x  sw@{best_key[1]}  => {best_wr*100:.1f}% *")
    else:
        print(f"BEST: thresh={best_key[0]}x  combo  => {best_wr*100:.1f}% *")

    # Also print the top 5 results
    print()
    print("Top 10 configurations:")
    all_results = []
    for (t, s), wr in results.items():
        all_results.append((wr, f"thresh={t:2d}x  sw@{s}"))
    for t, wr in combo_results.items():
        all_results.append((wr, f"thresh={t:2d}x  combo"))
    all_results.sort(reverse=True)
    for i, (wr, desc) in enumerate(all_results[:10]):
        print(f"  {i+1:2d}. {desc}  => {wr*100:.2f}%")


if __name__ == "__main__":
    import logging
    logging.disable(logging.INFO)  # suppress game/validation INFO spam
    run_grid()
