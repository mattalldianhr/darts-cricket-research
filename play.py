"""Human vs AI CLI mode with board display."""

from typing import List, Optional, Tuple

from agent import AIPlayer
from config import BULL, BULL_HIT_TYPES, DEFAULT_SKILL_LEVEL, HIT_TYPES, NUM_ACTIONS, SkillProfile, TARGET_NAMES, TARGETS
from game import DartsCricketGame


def get_human_input(game: DartsCricketGame) -> Tuple[int, str]:
    """Prompt for target and hit_type. Validates input, loops until valid."""
    valid_targets = {str(t): t for t in TARGETS if t != 25}
    valid_targets["bull"] = 25
    valid_targets["bullseye"] = 25
    valid_targets["25"] = 25

    valid_hits = {
        "single": "single",
        "s": "single",
        "double": "double",
        "d": "double",
        "triple": "triple",
        "t": "triple",
    }

    while True:
        raw_target = input("  Target (15-20, bull): ").strip().lower()
        if raw_target not in valid_targets:
            print(f"  Invalid target. Choose from: 15, 16, 17, 18, 19, 20, bull")
            continue
        target = valid_targets[raw_target]

        allowed_hits = BULL_HIT_TYPES if target == BULL else HIT_TYPES
        raw_hit = input("  Hit type (s/d/t or single/double/triple): ").strip().lower()
        if raw_hit not in valid_hits:
            print(f"  Invalid hit type. Choose from: s, d, t (or single, double, triple)")
            continue
        hit_type = valid_hits[raw_hit]

        if hit_type not in allowed_hits:
            print(f"  No triple on bullseye! Choose single (s) or double (d).")
            continue

        return target, hit_type


def print_board(game: DartsCricketGame) -> None:
    """Print the ASCII board."""
    print(game.get_board_display())


def print_throw_result(result: dict) -> None:
    """Print what happened on a throw."""
    target_name = TARGET_NAMES.get(result["target"], str(result["target"]))

    if result["missed"]:
        print(f"  -> Missed! (aimed at {target_name})")
        return

    actual = result.get("actual_hit_type", result["hit_type"])
    print(f"  -> Hit {actual} {target_name} (+{result['marks_added']} marks)")

    if result["closed"]:
        print(f"     Closed {target_name}!")
    if result["points_scored"] > 0:
        print(f"     Scored {result['points_scored']} points!")


def play_against_ai(
    q_table_path: str,
    miss_enabled: bool = False,
    skill_level: int = DEFAULT_SKILL_LEVEL,
    skill_profiles: Optional[List[SkillProfile]] = None,
) -> None:
    """Run a human vs AI game in the terminal."""
    game = DartsCricketGame(miss_enabled=miss_enabled, skill_profiles=skill_profiles)
    ai = AIPlayer(player_id=1, skill_level=skill_level)
    ai.load_q_table(q_table_path)
    ai.epsilon = 0.0  # no exploration during play

    print("\n" + "=" * 40)
    print("  DARTS CRICKET — Human vs AI")
    print(f"  You are Player 1. AI is Player 2 (skill {skill_level}).")
    print("=" * 40)
    print_board(game)

    while not game.game_over:
        if game.current_player == 0:
            # Human turn
            print(f"\n--- Your Turn (darts remaining: {game.darts_remaining}) ---")
            while game.darts_remaining > 0 and not game.game_over:
                if game.current_player != 0:
                    break
                print(f"\n  Dart {4 - game.darts_remaining}/3:")
                target, hit_type = get_human_input(game)
                result = game.throw_dart(target, hit_type)
                print_throw_result(result)
            print_board(game)
        else:
            # AI turn
            print(f"\n--- AI's Turn ---")
            state_before = game.get_state(ai.player_id)
            turn_info = ai.play_turn(game)

            for throw in turn_info["throws"]:
                action = throw["action"]
                result = throw["result"]
                target_name = TARGET_NAMES.get(throw["target"], str(throw["target"]))
                print(f"\n  AI aimed at {throw['hit_type']} {target_name}")
                print_throw_result(result)

                # Show top 3 Q-value actions (educational)
                q_vals = ai.get_starting_state_q_values(state_before)
                sorted_actions = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)[:3]
                q_display = ", ".join(
                    f"{ai.get_action_label(a)}: {v:.1f}" for a, v in sorted_actions
                )
                print(f"  [Top Q-values: {q_display}]")

            print_board(game)

    # Game over
    print("\n" + "=" * 40)
    if game.winner == 0:
        print("  YOU WIN! Congratulations!")
    else:
        print("  AI WINS! Better luck next time.")
    print(f"  Final Score — You: {game.scores[0]}  AI: {game.scores[1]}")
    print("=" * 40 + "\n")
