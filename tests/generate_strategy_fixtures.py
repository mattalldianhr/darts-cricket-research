"""Generate test fixtures for JS strategy parity validation.

Creates random game states and runs all 22 Python strategies against them,
outputting results as JSON. Use this to verify the JS port produces identical
recommendations.

Usage:
    python tests/generate_strategy_fixtures.py [num_states]
    # Outputs: docs/data/strategy_test_fixtures.json
"""

import json
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BULL, MARKS_TO_CLOSE, TARGETS
from game import DartsCricketGame
from strategies import (
    EXPERIMENTAL_CLASSES,
    STRATEGY_CLASSES,
    FrongelloStrategy,
)
from refined_grid import PhaseSwitchCombo


def random_game_state():
    """Generate a random but valid game state."""
    marks = [[random.randint(0, 3) for _ in range(7)] for _ in range(2)]
    scores = [random.randint(0, 120), random.randint(0, 120)]
    darts_remaining = random.choice([1, 2, 3])
    current_player = random.choice([0, 1])
    return {
        "marks": marks,
        "scores": scores,
        "darts_remaining": darts_remaining,
        "current_player": current_player,
    }


def create_game_from_state(state, player):
    """Create a DartsCricketGame matching the given state."""
    game = DartsCricketGame()
    game.marks = [list(state["marks"][0]), list(state["marks"][1])]
    game.scores = list(state["scores"])
    game.darts_remaining = state["darts_remaining"]
    game.current_player = player
    return game


def run_strategy(name, game, player):
    """Run a single strategy and return (target, hit_type)."""
    if name == "PS":
        bot = PhaseSwitchCombo(player_id=player, threshold_mult=13)
    elif name in STRATEGY_CLASSES:
        bot = STRATEGY_CLASSES[name](player_id=player)
    elif name in EXPERIMENTAL_CLASSES:
        bot = EXPERIMENTAL_CLASSES[name](player_id=player)
    else:
        raise ValueError(f"Unknown strategy: {name}")

    # For Frongello strategies, set dart_in_turn based on darts_remaining
    if hasattr(bot, '_dart_in_turn'):
        bot._dart_in_turn = 4 - game.darts_remaining

    target, hit_type = bot.choose_throw(game)
    return target, hit_type


def main():
    num_states = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    random.seed(42)  # Reproducible

    all_strategies = list(STRATEGY_CLASSES.keys()) + list(EXPERIMENTAL_CLASSES.keys()) + ["PS"]

    fixtures = []
    for i in range(num_states):
        state = random_game_state()
        player = state["current_player"]

        results = {}
        for name in all_strategies:
            game = create_game_from_state(state, player)
            try:
                target, hit_type = run_strategy(name, game, player)
                results[name] = {"target": target, "hitType": hit_type}
            except Exception as e:
                results[name] = {"error": str(e)}

        fixtures.append({
            "state": state,
            "player": player,
            "results": results,
        })

    output_path = Path(__file__).parent.parent / "docs" / "data" / "strategy_test_fixtures.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Generated {len(fixtures)} fixtures -> {output_path}")


if __name__ == "__main__":
    main()
