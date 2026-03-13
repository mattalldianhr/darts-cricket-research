"""CLI entry point for Darts Cricket system."""

import argparse
import logging
import sys

from config import (
    A2C_DEFAULT_ENTROPY_COEFF,
    A2C_DEFAULT_ENTROPY_MIN,
    A2C_DEFAULT_GAE_LAMBDA,
    A2C_DEFAULT_GAMMA,
    A2C_DEFAULT_GRAD_CLIP,
    A2C_DEFAULT_LR,
    A2C_DEFAULT_NUM_GAMES,
    A2C_DEFAULT_VALUE_COEFF,
    A2C_PRETRAIN_BATCH_SIZE,
    A2C_PRETRAIN_EPOCHS,
    A2C_PRETRAIN_GAMES_PER_BOT,
    A2C_PRETRAIN_LR,
    CLOSE_NUMBER_REWARD_FACTOR,
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_LOG_LEVEL,
    DEFAULT_NUM_GAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SKILL_LEVEL,
    DQN_DEFAULT_BATCH_SIZE,
    DQN_DEFAULT_GAMMA,
    DQN_DEFAULT_GRAD_CLIP,
    DQN_DEFAULT_LR,
    DQN_DEFAULT_MIN_REPLAY,
    DQN_DEFAULT_NUM_GAMES,
    DQN_DEFAULT_REPLAY_CAPACITY,
    DQN_DEFAULT_TAU,
    LOG_LEVEL_SILENT,
    LOG_LEVEL_SUMMARY,
    LOG_LEVEL_VERBOSE,
    SCORE_POINTS_REWARD_FACTOR,
    SKILL_PROFILES,
    scale_profile,
)


def _setup_logging(log_level: str) -> None:
    """Configure root logging based on the requested level."""
    level_map = {
        LOG_LEVEL_VERBOSE: logging.DEBUG,
        LOG_LEVEL_SUMMARY: logging.INFO,
        LOG_LEVEL_SILENT: logging.WARNING,
    }
    level = level_map.get(log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_skill_profile_args(parser: argparse.ArgumentParser) -> None:
    """Add --skill-profile, --skill-profile-p2, --accuracy-scale to a subparser."""
    profile_names = ", ".join(SKILL_PROFILES.keys())
    parser.add_argument(
        "--skill-profile", type=str, default=None,
        help=f"Skill profile for both players ({profile_names})",
    )
    parser.add_argument(
        "--skill-profile-p2", type=str, default=None,
        help="Override skill profile for player 2 only",
    )
    parser.add_argument(
        "--accuracy-scale", type=float, default=None,
        help="Scale P2 hit probabilities (e.g., 0.95 = 95%% of P1 accuracy)",
    )


def _build_skill_profiles(args: argparse.Namespace):
    """Resolve CLI args into a [P1_profile, P2_profile] list or None."""
    profile_name = getattr(args, "skill_profile", None)
    profile_p2_name = getattr(args, "skill_profile_p2", None)
    accuracy_scale = getattr(args, "accuracy_scale", None)
    miss_enabled = getattr(args, "miss_enabled", False)

    if profile_name is None and not miss_enabled:
        return None

    # Start from the named profile or legacy (for --miss-enabled)
    if profile_name is not None:
        if profile_name not in SKILL_PROFILES:
            raise ValueError(f"Unknown skill profile: {profile_name}")
        p1_profile = SKILL_PROFILES[profile_name]
    else:
        # --miss-enabled without --skill-profile → legacy
        from config import SKILL_PROFILE_LEGACY
        p1_profile = SKILL_PROFILE_LEGACY

    # Determine P2 profile
    if profile_p2_name is not None:
        if profile_p2_name not in SKILL_PROFILES:
            raise ValueError(f"Unknown skill profile: {profile_p2_name}")
        p2_profile = SKILL_PROFILES[profile_p2_name]
    else:
        p2_profile = p1_profile

    # Apply accuracy scaling to P2
    if accuracy_scale is not None:
        p2_profile = scale_profile(p2_profile, accuracy_scale)

    return [p1_profile, p2_profile]


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Darts Cricket — Q-learning AI training and play system",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    train_parser = subparsers.add_parser("train", help="Train Q-learning agents")
    train_parser.add_argument(
        "--games", type=int, default=DEFAULT_NUM_GAMES,
        help=f"Number of training games (default: {DEFAULT_NUM_GAMES})",
    )
    train_parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Learning rate (default: {DEFAULT_ALPHA})",
    )
    train_parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Discount factor (default: {DEFAULT_GAMMA})",
    )
    train_parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON,
        help=f"Initial exploration rate (default: {DEFAULT_EPSILON})",
    )
    train_parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    train_parser.add_argument(
        "--log-level", type=str, default=DEFAULT_LOG_LEVEL,
        choices=[LOG_LEVEL_VERBOSE, LOG_LEVEL_SUMMARY, LOG_LEVEL_SILENT],
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL})",
    )
    train_parser.add_argument(
        "--miss-enabled", action="store_true",
        help="Enable realistic miss probabilities",
    )
    train_parser.add_argument(
        "--skill-p1", type=int, default=DEFAULT_SKILL_LEVEL,
        help="Skill level for player 1 (3=beginner, 5=intermediate, 6=solid, 9=expert)",
    )
    train_parser.add_argument(
        "--skill-p2", type=int, default=DEFAULT_SKILL_LEVEL,
        help="Skill level for player 2 (3=beginner, 5=intermediate, 6=solid, 9=expert)",
    )
    train_parser.add_argument(
        "--curriculum", action="store_true",
        help="Train against rotating strategy bots (S1, S2, S10) + self-play",
    )
    _add_skill_profile_args(train_parser)

    # train-dqn
    dqn_parser = subparsers.add_parser("train-dqn", help="Train DQN agents")
    dqn_parser.add_argument(
        "--games", type=int, default=DQN_DEFAULT_NUM_GAMES,
        help=f"Number of training games (default: {DQN_DEFAULT_NUM_GAMES})",
    )
    dqn_parser.add_argument(
        "--gamma", type=float, default=DQN_DEFAULT_GAMMA,
        help=f"Discount factor (default: {DQN_DEFAULT_GAMMA})",
    )
    dqn_parser.add_argument(
        "--lr", type=float, default=DQN_DEFAULT_LR,
        help=f"Learning rate (default: {DQN_DEFAULT_LR})",
    )
    dqn_parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON,
        help=f"Initial exploration rate (default: {DEFAULT_EPSILON})",
    )
    dqn_parser.add_argument(
        "--batch-size", type=int, default=DQN_DEFAULT_BATCH_SIZE,
        help=f"Replay batch size (default: {DQN_DEFAULT_BATCH_SIZE})",
    )
    dqn_parser.add_argument(
        "--replay-capacity", type=int, default=DQN_DEFAULT_REPLAY_CAPACITY,
        help=f"Replay buffer capacity (default: {DQN_DEFAULT_REPLAY_CAPACITY})",
    )
    dqn_parser.add_argument(
        "--min-replay", type=int, default=DQN_DEFAULT_MIN_REPLAY,
        help=f"Min replay size before training (default: {DQN_DEFAULT_MIN_REPLAY})",
    )
    dqn_parser.add_argument(
        "--tau", type=float, default=DQN_DEFAULT_TAU,
        help=f"Target network soft update rate (default: {DQN_DEFAULT_TAU})",
    )
    dqn_parser.add_argument(
        "--grad-clip", type=float, default=DQN_DEFAULT_GRAD_CLIP,
        help=f"Gradient clipping max norm (default: {DQN_DEFAULT_GRAD_CLIP})",
    )
    dqn_parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    dqn_parser.add_argument(
        "--log-level", type=str, default=DEFAULT_LOG_LEVEL,
        choices=[LOG_LEVEL_VERBOSE, LOG_LEVEL_SUMMARY, LOG_LEVEL_SILENT],
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL})",
    )
    dqn_parser.add_argument(
        "--miss-enabled", action="store_true",
        help="Enable realistic miss probabilities",
    )
    dqn_parser.add_argument(
        "--skill-p1", type=int, default=DEFAULT_SKILL_LEVEL,
        help="Skill level for player 1",
    )
    dqn_parser.add_argument(
        "--skill-p2", type=int, default=DEFAULT_SKILL_LEVEL,
        help="Skill level for player 2",
    )
    dqn_parser.add_argument(
        "--curriculum", action="store_true",
        help="Train against rotating strategy bots (S1, S2, S10) + self-play",
    )
    dqn_parser.add_argument(
        "--close-reward", type=float, default=CLOSE_NUMBER_REWARD_FACTOR,
        help=f"Close number reward factor (default: {CLOSE_NUMBER_REWARD_FACTOR})",
    )
    dqn_parser.add_argument(
        "--score-reward", type=float, default=SCORE_POINTS_REWARD_FACTOR,
        help=f"Score points reward factor (default: {SCORE_POINTS_REWARD_FACTOR})",
    )
    _add_skill_profile_args(dqn_parser)

    # train-a2c
    a2c_parser = subparsers.add_parser("train-a2c", help="Train A2C branching agents")
    a2c_parser.add_argument(
        "--games", type=int, default=A2C_DEFAULT_NUM_GAMES,
        help=f"Number of training games (default: {A2C_DEFAULT_NUM_GAMES})",
    )
    a2c_parser.add_argument(
        "--gamma", type=float, default=A2C_DEFAULT_GAMMA,
        help=f"Discount factor (default: {A2C_DEFAULT_GAMMA})",
    )
    a2c_parser.add_argument(
        "--lr", type=float, default=A2C_DEFAULT_LR,
        help=f"Learning rate (default: {A2C_DEFAULT_LR})",
    )
    a2c_parser.add_argument(
        "--gae-lambda", type=float, default=A2C_DEFAULT_GAE_LAMBDA,
        help=f"GAE lambda (default: {A2C_DEFAULT_GAE_LAMBDA})",
    )
    a2c_parser.add_argument(
        "--value-coeff", type=float, default=A2C_DEFAULT_VALUE_COEFF,
        help=f"Value loss coefficient (default: {A2C_DEFAULT_VALUE_COEFF})",
    )
    a2c_parser.add_argument(
        "--entropy-coeff", type=float, default=A2C_DEFAULT_ENTROPY_COEFF,
        help=f"Entropy bonus coefficient (default: {A2C_DEFAULT_ENTROPY_COEFF})",
    )
    a2c_parser.add_argument(
        "--entropy-min", type=float, default=A2C_DEFAULT_ENTROPY_MIN,
        help=f"Minimum entropy coefficient (default: {A2C_DEFAULT_ENTROPY_MIN})",
    )
    a2c_parser.add_argument(
        "--grad-clip", type=float, default=A2C_DEFAULT_GRAD_CLIP,
        help=f"Gradient clipping max norm (default: {A2C_DEFAULT_GRAD_CLIP})",
    )
    a2c_parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    a2c_parser.add_argument(
        "--log-level", type=str, default=DEFAULT_LOG_LEVEL,
        choices=[LOG_LEVEL_VERBOSE, LOG_LEVEL_SUMMARY, LOG_LEVEL_SILENT],
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL})",
    )
    a2c_parser.add_argument(
        "--miss-enabled", action="store_true",
        help="Enable realistic miss probabilities",
    )
    a2c_parser.add_argument(
        "--skill-p1", type=int, default=DEFAULT_SKILL_LEVEL,
        help="Skill level for player 1",
    )
    a2c_parser.add_argument(
        "--skill-p2", type=int, default=DEFAULT_SKILL_LEVEL,
        help="Skill level for player 2",
    )
    a2c_parser.add_argument(
        "--curriculum", action="store_true",
        help="Train with curriculum (progressive opponents)",
    )
    a2c_parser.add_argument(
        "--close-reward", type=float, default=CLOSE_NUMBER_REWARD_FACTOR,
        help=f"Close number reward factor (default: {CLOSE_NUMBER_REWARD_FACTOR})",
    )
    a2c_parser.add_argument(
        "--score-reward", type=float, default=SCORE_POINTS_REWARD_FACTOR,
        help=f"Score points reward factor (default: {SCORE_POINTS_REWARD_FACTOR})",
    )
    a2c_parser.add_argument(
        "--pretrained-model", type=str, default=None,
        help="Path to pre-trained A2C model (.pt) to initialize from",
    )
    a2c_parser.add_argument(
        "--opponent-rotation", action="store_true",
        help="Train against rotating pool of all strategy bots",
    )
    a2c_parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from --pretrained-model checkpoint (preserves optimizer & entropy)",
    )
    a2c_parser.add_argument(
        "--bot-pool", type=str, default=None,
        help="Comma-separated list of bot names for opponent rotation (e.g., S2,S4,S6,E3). "
             "If omitted with --opponent-rotation, uses all bots.",
    )
    _add_skill_profile_args(a2c_parser)

    # pretrain-a2c
    pretrain_parser = subparsers.add_parser(
        "pretrain-a2c", help="Supervised pre-training for A2C network",
    )
    pretrain_parser.add_argument(
        "--output", type=str, default="output/a2c_pretrained.pt",
        help="Output path for pre-trained model (default: output/a2c_pretrained.pt)",
    )
    pretrain_parser.add_argument(
        "--games-per-bot", type=int, default=A2C_PRETRAIN_GAMES_PER_BOT,
        help=f"Games per bot for data generation (default: {A2C_PRETRAIN_GAMES_PER_BOT})",
    )
    pretrain_parser.add_argument(
        "--epochs", type=int, default=A2C_PRETRAIN_EPOCHS,
        help=f"Training epochs (default: {A2C_PRETRAIN_EPOCHS})",
    )
    pretrain_parser.add_argument(
        "--lr", type=float, default=A2C_PRETRAIN_LR,
        help=f"Learning rate (default: {A2C_PRETRAIN_LR})",
    )
    pretrain_parser.add_argument(
        "--batch-size", type=int, default=A2C_PRETRAIN_BATCH_SIZE,
        help=f"Batch size (default: {A2C_PRETRAIN_BATCH_SIZE})",
    )
    pretrain_parser.add_argument(
        "--log-level", type=str, default=DEFAULT_LOG_LEVEL,
        choices=[LOG_LEVEL_VERBOSE, LOG_LEVEL_SUMMARY, LOG_LEVEL_SILENT],
        help=f"Logging level (default: {DEFAULT_LOG_LEVEL})",
    )
    pretrain_parser.add_argument(
        "--miss-enabled", action="store_true",
        help="Enable realistic miss probabilities during data generation",
    )
    _add_skill_profile_args(pretrain_parser)

    # play
    play_parser = subparsers.add_parser("play", help="Play against a trained AI")
    play_parser.add_argument(
        "--q-table", type=str, required=True,
        help="Path to a saved Q-table pickle file",
    )
    play_parser.add_argument(
        "--miss-enabled", action="store_true",
        help="Enable realistic miss probabilities",
    )
    play_parser.add_argument(
        "--skill", type=int, default=DEFAULT_SKILL_LEVEL,
        help="AI opponent skill level (3=beginner, 5=intermediate, 6=solid, 9=expert)",
    )
    _add_skill_profile_args(play_parser)

    # grid-search
    grid_parser = subparsers.add_parser(
        "grid-search", help="Run hyperparameter grid search",
    )
    grid_parser.add_argument(
        "--config", type=str, default="config.json",
        help="Path to grid search config JSON (default: config.json)",
    )
    grid_parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    # plot
    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from saved training data",
    )
    plot_parser.add_argument(
        "--metrics-dir", type=str, required=True,
        help="Directory containing training_metrics.pkl",
    )
    plot_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: metrics_dir/plots)",
    )

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze Q-table strategy",
    )
    analyze_parser.add_argument(
        "--q-table", type=str, required=True,
        help="Path to Q-table or DQN model",
    )
    analyze_parser.add_argument(
        "--q-table-2", type=str, default=None,
        help="Optional second Q-table for comparison",
    )
    analyze_parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    analyze_parser.add_argument(
        "--model-type", type=str, default="qtable", choices=["qtable", "dqn", "a2c"],
        help="Model type: qtable (pickle), dqn (.pt), or a2c (.pt) (default: qtable)",
    )

    # advise
    advise_parser = subparsers.add_parser(
        "advise", help="Get throw advice",
    )
    advise_parser.add_argument(
        "--q-table", type=str, required=True,
        help="Path to Q-table",
    )
    advise_parser.add_argument(
        "--my-score", type=int, required=True,
        help="Your current score",
    )
    advise_parser.add_argument(
        "--opp-score", type=int, required=True,
        help="Opponent's current score",
    )
    advise_parser.add_argument(
        "--my-marks", type=str, required=True,
        help="Your marks as comma-separated (7 values for 15,16,17,18,19,20,Bull)",
    )
    advise_parser.add_argument(
        "--opp-marks", type=str, required=True,
        help="Opponent's marks as comma-separated",
    )
    advise_parser.add_argument(
        "--model-type", type=str, default="qtable", choices=["qtable", "dqn", "a2c"],
        help="Model type: qtable (pickle), dqn (.pt), or a2c (.pt) (default: qtable)",
    )

    # validate
    validate_parser = subparsers.add_parser(
        "validate", help="Run Frongello validation suite",
    )
    validate_parser.add_argument(
        "--q-table", type=str, required=True,
        help="Path to trained Q-table",
    )
    validate_parser.add_argument(
        "--games", type=int, default=None,
        help="Games per matchup (default: from config)",
    )
    validate_parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    validate_parser.add_argument(
        "--model-type", type=str, default="qtable", choices=["qtable", "dqn", "a2c"],
        help="Model type: qtable (pickle), dqn (.pt), or a2c (.pt) (default: qtable)",
    )
    validate_parser.add_argument(
        "--miss-enabled", action="store_true",
        help="Enable realistic miss probabilities during validation games",
    )
    _add_skill_profile_args(validate_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        _setup_logging(args.log_level)
        from training import train_agents
        train_agents(
            num_games=args.games,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            output_dir=args.output_dir,
            log_level=args.log_level,
            miss_enabled=args.miss_enabled,
            skill_level_p1=args.skill_p1,
            skill_level_p2=args.skill_p2,
            curriculum=args.curriculum,
            skill_profiles=_build_skill_profiles(args),
        )

    elif args.command == "train-dqn":
        _setup_logging(args.log_level)
        from dqn_training import train_dqn_agents
        train_dqn_agents(
            num_games=args.games,
            gamma=args.gamma,
            lr=args.lr,
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            min_replay=args.min_replay,
            tau=args.tau,
            grad_clip=args.grad_clip,
            output_dir=args.output_dir,
            log_level=args.log_level,
            miss_enabled=args.miss_enabled,
            skill_level_p1=args.skill_p1,
            skill_level_p2=args.skill_p2,
            curriculum=args.curriculum,
            close_reward_factor=args.close_reward,
            score_reward_factor=args.score_reward,
            skill_profiles=_build_skill_profiles(args),
        )

    elif args.command == "train-a2c":
        _setup_logging(args.log_level)
        from a2c_training import train_a2c_agents
        train_a2c_agents(
            num_games=args.games,
            gamma=args.gamma,
            lr=args.lr,
            gae_lambda=args.gae_lambda,
            value_coeff=args.value_coeff,
            entropy_coeff=args.entropy_coeff,
            entropy_min=args.entropy_min,
            grad_clip=args.grad_clip,
            output_dir=args.output_dir,
            log_level=args.log_level,
            miss_enabled=args.miss_enabled,
            skill_level_p1=args.skill_p1,
            skill_level_p2=args.skill_p2,
            curriculum=args.curriculum,
            close_reward_factor=args.close_reward,
            score_reward_factor=args.score_reward,
            pretrained_model=args.pretrained_model,
            opponent_rotation=args.opponent_rotation,
            skill_profiles=_build_skill_profiles(args),
            resume=getattr(args, "resume", False),
            bot_pool_names=[s.strip() for s in args.bot_pool.split(",")] if args.bot_pool else None,
        )

    elif args.command == "pretrain-a2c":
        _setup_logging(args.log_level)
        from a2c_pretrain import run_pretrain
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        run_pretrain(
            output_path=args.output,
            num_games_per_bot=args.games_per_bot,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            miss_enabled=args.miss_enabled,
            skill_profiles=_build_skill_profiles(args),
        )

    elif args.command == "play":
        _setup_logging(LOG_LEVEL_SUMMARY)
        from play import play_against_ai
        play_against_ai(
            q_table_path=args.q_table,
            miss_enabled=args.miss_enabled,
            skill_level=args.skill,
            skill_profiles=_build_skill_profiles(args),
        )

    elif args.command == "grid-search":
        _setup_logging(LOG_LEVEL_SUMMARY)
        from hyperparams import run_grid_search
        run_grid_search(
            config_path=args.config,
            output_dir=args.output_dir,
        )

    elif args.command == "plot":
        _setup_logging(LOG_LEVEL_SUMMARY)
        from metrics import TrainingMetrics, generate_all_plots
        import os
        output_dir = args.output_dir or os.path.join(args.metrics_dir, "plots")
        metrics = TrainingMetrics.load(args.metrics_dir)
        generate_all_plots(metrics, output_dir)
        logging.getLogger(__name__).info("Plots saved to %s", output_dir)

    elif args.command == "analyze":
        _setup_logging(LOG_LEVEL_SUMMARY)
        from advisor import load_advisor, load_dqn_advisor, load_a2c_advisor
        from analysis import (
            analyze_opening_preferences,
            analyze_chase_behavior,
            analyze_scoring_vs_closing,
            generate_strategy_heatmap,
        )
        import os

        if args.model_type == "dqn":
            agent = load_dqn_advisor(args.q_table)
        elif args.model_type == "a2c":
            agent = load_a2c_advisor(args.q_table)
        else:
            agent = load_advisor(args.q_table)
        os.makedirs(args.output_dir, exist_ok=True)

        # Print opening preferences
        prefs = analyze_opening_preferences(agent)
        print("\nOpening Preferences (top 10):")
        for action_idx, label, qv in prefs[:10]:
            print(f"  {label:>15s}  Q={qv:.2f}")

        # Chase behavior
        chase = analyze_chase_behavior(agent)
        print("\nChase Behavior Analysis:")
        for key, val in chase.items():
            print(f"  {key}: {val}")

        # Scoring vs closing
        svc = analyze_scoring_vs_closing(agent)
        print("\nScoring vs Closing Preferences:")
        for scenario, info in svc.items():
            print(f"  {scenario}: {info}")

        # Generate heatmap
        heatmap_path = os.path.join(args.output_dir, "strategy_heatmap.png")
        generate_strategy_heatmap(agent, heatmap_path)
        print(f"\nHeatmap saved to {heatmap_path}")

    elif args.command == "advise":
        _setup_logging(LOG_LEVEL_SUMMARY)
        from advisor import load_advisor, load_dqn_advisor, load_a2c_advisor, get_advice
        from config import TARGET_NAMES
        from game import DartsCricketGame

        if args.model_type == "dqn":
            agent = load_dqn_advisor(args.q_table)
        elif args.model_type == "a2c":
            agent = load_a2c_advisor(args.q_table)
        else:
            agent = load_advisor(args.q_table)
        my_marks = [int(x) for x in args.my_marks.split(",")]
        opp_marks = [int(x) for x in args.opp_marks.split(",")]

        # Create game with the board state
        game = DartsCricketGame()
        game.scores = [args.my_score, args.opp_score]
        game.marks = [my_marks[:], opp_marks[:]]

        advice = get_advice(agent, game, player_perspective=0, top_n=5)
        print(f"\nStrategic Context: {advice.strategic_context}")
        print("\nRecommendations:")
        for rec in advice.recommendations:
            target_name = TARGET_NAMES.get(rec.target, str(rec.target))
            print(f"  #{rec.rank}: {rec.hit_type} {target_name} (Q={rec.q_value:.2f})")
            print(f"         {rec.explanation}")

    elif args.command == "validate":
        _setup_logging(LOG_LEVEL_SUMMARY)
        from validation import validate_frongello_principles
        from config import VALIDATION_GAMES_PER_MATCHUP
        num_games = args.games or VALIDATION_GAMES_PER_MATCHUP
        report = validate_frongello_principles(
            args.q_table,
            num_games=num_games,
            output_dir=args.output_dir,
            model_type=args.model_type,
            miss_enabled=args.miss_enabled,
            skill_profiles=_build_skill_profiles(args),
        )
        passed = sum(1 for p in report.principles if p.passed)
        total = len(report.principles)
        print(f"\nFrongello Validation: {passed}/{total} principles passed\n")
        print(report.summary)


if __name__ == "__main__":
    main()
