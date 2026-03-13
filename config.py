"""Constants, default hyperparameters, reward values, and logging levels."""

from dataclasses import dataclass
from typing import Dict, List

# Game Constants
TARGETS: List[int] = [15, 16, 17, 18, 19, 20, 25]
TARGET_NAMES: Dict[int, str] = {
    15: "15",
    16: "16",
    17: "17",
    18: "18",
    19: "19",
    20: "20",
    25: "Bull",
}
HIT_TYPES: List[str] = ["single", "double", "triple"]
HIT_MULTIPLIER: Dict[str, int] = {"single": 1, "double": 2, "triple": 3}
MARKS_TO_CLOSE: int = 3
DARTS_PER_TURN: int = 3
NUM_ACTIONS: int = 20  # 6 targets x 3 hit types + Bull x 2 (no triple bull)
BULL: int = 25  # Bullseye target value
BULL_HIT_TYPES: List[str] = ["single", "double"]  # No triple for bullseye
BULL_DIFFICULTY_MULTIPLIER: float = 0.75  # Bull is geometrically harder (smaller circular target)

# Skill Levels — cap total marks an agent can attempt per turn
DEFAULT_SKILL_LEVEL: int = 9  # unconstrained (max possible = 9 = 3 triples)
SKILL_LEVELS: Dict[str, int] = {
    "beginner": 3,
    "intermediate": 5,
    "solid": 6,
    "expert": 9,
}

# Miss Probabilities
HIT_PROBABILITY: Dict[str, float] = {
    "single": 0.85,
    "double": 0.75,
    "triple": 0.70,
}
MISS_ENABLED: bool = False

# Q-Learning Defaults
DEFAULT_ALPHA: float = 0.1
DEFAULT_GAMMA: float = 0.95
DEFAULT_EPSILON: float = 1.0
DEFAULT_EPSILON_MIN: float = 0.01
DEFAULT_EPSILON_DECAY: float = 0.99995
DEFAULT_NUM_GAMES: int = 50000

# Reward Values
WIN_REWARD: float = 100.0
LOSS_PENALTY: float = -100.0
CLOSE_NUMBER_REWARD_FACTOR: float = 0.3  # multiplied by target face value (e.g., closing 20 = 6.0, closing 15 = 4.5)
SCORE_POINTS_REWARD_FACTOR: float = 0.1

# Logging
LOG_LEVEL_VERBOSE: str = "verbose"
LOG_LEVEL_SUMMARY: str = "summary"
LOG_LEVEL_SILENT: str = "silent"
DEFAULT_LOG_LEVEL: str = LOG_LEVEL_SUMMARY

# State Representation — score bucketing to reduce state space
# Instead of raw (score_self, score_opp), use bucketed score differential
# Buckets: far_behind=-2, behind=-1, tied=0, ahead=1, far_ahead=2
SCORE_BUCKET_THRESHOLDS: List[int] = [0, 40]  # boundaries for near/far

# Output
DEFAULT_OUTPUT_DIR: str = "output"
CHECKPOINT_INTERVAL: int = 10000
SMOOTHING_WINDOW: int = 1000

# Validation
VALIDATION_GAMES_PER_MATCHUP: int = 20000
FRONGELLO_WIN_RATE_THRESHOLD: float = 0.55
STRATEGY_NAMES: Dict[str, str] = {
    "S1":  "S1 (Cover only)",
    "S2":  "S2 (Lead×0)",
    "S3":  "S3 (Lead×3)",
    "S4":  "S4 (Lead×6)",
    "S5":  "S5 (Lead×9)",
    "S6":  "S6 (Lead×0 +Extra)",
    "S7":  "S7 (Lead×3 +Extra)",
    "S8":  "S8 (Lead×6 +Extra)",
    "S9":  "S9 (Lead×9 +Extra)",
    "S10": "S10 (Lead×0 +Chase)",
    "S11": "S11 (Lead×3 +Chase)",
    "S12": "S12 (Lead×6 +Chase)",
    "S13": "S13 (Lead×9 +Chase)",
    "S14": "S14 (Lead×0 +Extra+Chase)",
    "S15": "S15 (Lead×3 +Extra+Chase)",
    "S16": "S16 (Lead×6 +Extra+Chase)",
    "S17": "S17 (Lead×9 +Extra+Chase)",
}

# DQN Defaults
DQN_DEFAULT_GAMMA: float = 0.99
DQN_DEFAULT_LR: float = 1e-4
DQN_DEFAULT_BATCH_SIZE: int = 64
DQN_DEFAULT_REPLAY_CAPACITY: int = 100_000
DQN_DEFAULT_MIN_REPLAY: int = 1000
DQN_DEFAULT_TAU: float = 0.005
DQN_DEFAULT_GRAD_CLIP: float = 1.0
DQN_DEFAULT_NUM_GAMES: int = 50_000
DQN_STATE_SIZE: int = 15
DQN_HIDDEN_SIZE: int = 128

# A2C Defaults
A2C_DEFAULT_LR: float = 1e-4
A2C_DEFAULT_GAMMA: float = 0.99
A2C_DEFAULT_GAE_LAMBDA: float = 0.95
A2C_DEFAULT_VALUE_COEFF: float = 0.5
A2C_DEFAULT_ENTROPY_COEFF: float = 0.05
A2C_DEFAULT_ENTROPY_MIN: float = 0.005
A2C_DEFAULT_GRAD_CLIP: float = 0.5
A2C_DEFAULT_NUM_GAMES: int = 200_000
A2C_TRUNK_HIDDEN: int = 128
A2C_BRANCH_HIDDEN: int = 64
A2C_STATE_SIZE: int = 19
A2C_PRETRAIN_LR: float = 1e-3
A2C_PRETRAIN_EPOCHS: int = 5
A2C_PRETRAIN_BATCH_SIZE: int = 256
A2C_PRETRAIN_GAMES_PER_BOT: int = 50_000


# ── Skill Profiles ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillProfile:
    """Per-player throw outcome distribution.

    Each *_outcomes dict maps outcome type ("triple", "double", "single", "miss")
    to a probability.  Values must sum to 1.0.  One ``random.choices()`` call
    using these weights resolves the throw — no separate hit/miss step.
    """

    name: str
    triple_outcomes: Dict[str, float]
    double_outcomes: Dict[str, float]
    single_outcomes: Dict[str, float]


# Pre-built profiles ─────────────────────────────────────────────────

SKILL_PROFILE_PERFECT = SkillProfile(
    name="perfect",
    triple_outcomes={"triple": 1.0, "double": 0.0, "single": 0.0, "miss": 0.0},
    double_outcomes={"double": 1.0, "single": 0.0, "miss": 0.0},
    single_outcomes={"single": 1.0, "miss": 0.0},
)

SKILL_PROFILE_PRO = SkillProfile(
    name="pro",
    triple_outcomes={"triple": 0.41, "double": 0.20, "single": 0.25, "miss": 0.14},
    double_outcomes={"double": 0.40, "single": 0.35, "miss": 0.25},
    single_outcomes={"single": 0.96, "miss": 0.04},
)

SKILL_PROFILE_GOOD = SkillProfile(
    name="good",
    triple_outcomes={"triple": 0.30, "double": 0.22, "single": 0.30, "miss": 0.18},
    double_outcomes={"double": 0.30, "single": 0.40, "miss": 0.30},
    single_outcomes={"single": 0.90, "miss": 0.10},
)

SKILL_PROFILE_AMATEUR = SkillProfile(
    name="amateur",
    triple_outcomes={"triple": 0.15, "double": 0.20, "single": 0.35, "miss": 0.30},
    double_outcomes={"double": 0.20, "single": 0.40, "miss": 0.40},
    single_outcomes={"single": 0.80, "miss": 0.20},
)

SKILL_PROFILE_LEGACY = SkillProfile(
    name="legacy",
    triple_outcomes={"triple": 0.70, "double": 0.10, "single": 0.10, "miss": 0.10},
    double_outcomes={"double": 0.75, "single": 0.125, "miss": 0.125},
    single_outcomes={"single": 0.85, "miss": 0.15},
)

SKILL_PROFILES: Dict[str, SkillProfile] = {
    "perfect": SKILL_PROFILE_PERFECT,
    "pro": SKILL_PROFILE_PRO,
    "good": SKILL_PROFILE_GOOD,
    "amateur": SKILL_PROFILE_AMATEUR,
    "legacy": SKILL_PROFILE_LEGACY,
}


def scale_profile(base: SkillProfile, factor: float) -> SkillProfile:
    """Scale a profile's hit probabilities by *factor*, redistributing to miss.

    ``scale_profile(pro, 0.95)`` models Frongello's "one player 95% as
    accurate" scenario.  The primary hit probability for each aimed type
    is multiplied by *factor*; any freed probability mass shifts to "miss".
    """
    def _scale_outcomes(outcomes: Dict[str, float], primary: str) -> Dict[str, float]:
        scaled = dict(outcomes)
        original_hit = outcomes[primary]
        new_hit = original_hit * factor
        delta = original_hit - new_hit
        scaled[primary] = new_hit
        scaled["miss"] = outcomes.get("miss", 0.0) + delta
        return scaled

    return SkillProfile(
        name=f"{base.name}×{factor}",
        triple_outcomes=_scale_outcomes(base.triple_outcomes, "triple"),
        double_outcomes=_scale_outcomes(base.double_outcomes, "double"),
        single_outcomes=_scale_outcomes(base.single_outcomes, "single"),
    )
