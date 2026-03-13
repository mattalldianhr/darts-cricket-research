"""Tests for SkillProfile system: presets, scaling, game integration."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    SKILL_PROFILES,
    SKILL_PROFILE_AMATEUR,
    SKILL_PROFILE_GOOD,
    SKILL_PROFILE_LEGACY,
    SKILL_PROFILE_PERFECT,
    SKILL_PROFILE_PRO,
    SkillProfile,
    scale_profile,
)
from game import DartsCricketGame


# ── Profile validation ────────────────────────────────────────────────


def test_all_presets_sum_to_one():
    """Every preset's outcome dicts must sum to 1.0 (within float tolerance)."""
    for name, profile in SKILL_PROFILES.items():
        for label, outcomes in [
            ("triple", profile.triple_outcomes),
            ("double", profile.double_outcomes),
            ("single", profile.single_outcomes),
        ]:
            total = sum(outcomes.values())
            assert math.isclose(total, 1.0, abs_tol=1e-9), (
                f"{name}.{label}_outcomes sums to {total}"
            )


def test_all_presets_non_negative():
    """No negative probabilities in any preset."""
    for name, profile in SKILL_PROFILES.items():
        for outcomes in [
            profile.triple_outcomes,
            profile.double_outcomes,
            profile.single_outcomes,
        ]:
            for key, val in outcomes.items():
                assert val >= 0.0, f"{name} has negative prob: {key}={val}"


def test_preset_names_match_dict_keys():
    """Each preset's .name matches its dict key."""
    for key, profile in SKILL_PROFILES.items():
        assert profile.name == key


def test_five_presets_exist():
    """We should have exactly 5 named presets."""
    expected = {"perfect", "pro", "good", "amateur", "legacy"}
    assert set(SKILL_PROFILES.keys()) == expected


# ── Scale function ────────────────────────────────────────────────────


def test_scale_identity():
    """scale(pro, 1.0) returns equivalent probabilities."""
    scaled = scale_profile(SKILL_PROFILE_PRO, 1.0)
    for key in SKILL_PROFILE_PRO.triple_outcomes:
        assert math.isclose(
            scaled.triple_outcomes[key],
            SKILL_PROFILE_PRO.triple_outcomes[key],
            abs_tol=1e-9,
        )


def test_scale_halves_hit():
    """scale(pro, 0.5) halves the primary hit probability."""
    scaled = scale_profile(SKILL_PROFILE_PRO, 0.5)
    assert math.isclose(
        scaled.triple_outcomes["triple"],
        SKILL_PROFILE_PRO.triple_outcomes["triple"] * 0.5,
        abs_tol=1e-9,
    )
    assert math.isclose(
        scaled.double_outcomes["double"],
        SKILL_PROFILE_PRO.double_outcomes["double"] * 0.5,
        abs_tol=1e-9,
    )
    assert math.isclose(
        scaled.single_outcomes["single"],
        SKILL_PROFILE_PRO.single_outcomes["single"] * 0.5,
        abs_tol=1e-9,
    )


def test_scale_preserves_sum():
    """Scaled profiles still sum to 1.0."""
    for factor in [0.1, 0.5, 0.95, 1.0]:
        scaled = scale_profile(SKILL_PROFILE_PRO, factor)
        for outcomes in [
            scaled.triple_outcomes,
            scaled.double_outcomes,
            scaled.single_outcomes,
        ]:
            total = sum(outcomes.values())
            assert math.isclose(total, 1.0, abs_tol=1e-9), (
                f"factor={factor}: sum={total}"
            )


def test_scale_name():
    """Scaled profile has descriptive name."""
    scaled = scale_profile(SKILL_PROFILE_PRO, 0.95)
    assert "pro" in scaled.name
    assert "0.95" in scaled.name


# ── Game integration ──────────────────────────────────────────────────


def test_perfect_profile_always_hits():
    """With perfect profile, all throws hit exactly as aimed."""
    game = DartsCricketGame(skill_profiles=[SKILL_PROFILE_PERFECT, SKILL_PROFILE_PERFECT])
    for _ in range(100):
        game.reset()
        result = game.throw_dart(20, "triple")
        assert result["missed"] is False
        assert result["actual_hit_type"] == "triple"
        assert result["marks_added"] == 3


def test_legacy_profile_triple_hit_rate():
    """Legacy profile hits triples ~70% of the time (statistical)."""
    game = DartsCricketGame(skill_profiles=[SKILL_PROFILE_LEGACY, SKILL_PROFILE_LEGACY])
    hits = 0
    n = 10000
    for _ in range(n):
        game.reset()
        result = game.throw_dart(20, "triple")
        if result["actual_hit_type"] == "triple" and not result["missed"]:
            hits += 1
    hit_rate = hits / n
    # 70% expected, allow reasonable statistical margin
    assert 0.65 < hit_rate < 0.75, f"Legacy triple hit rate: {hit_rate:.3f}"


def test_backward_compat_miss_enabled():
    """DartsCricketGame(miss_enabled=True) uses legacy profile."""
    game = DartsCricketGame(miss_enabled=True)
    assert game.miss_enabled is True
    assert game.skill_profiles is not None
    assert len(game.skill_profiles) == 2
    assert game.skill_profiles[0].name == "legacy"
    assert game.skill_profiles[1].name == "legacy"


def test_backward_compat_no_miss():
    """DartsCricketGame() has no skill profiles and miss_enabled=False."""
    game = DartsCricketGame()
    assert game.miss_enabled is False
    assert game.skill_profiles is None


def test_skill_profiles_takes_precedence():
    """skill_profiles param overrides miss_enabled."""
    game = DartsCricketGame(
        miss_enabled=True,
        skill_profiles=[SKILL_PROFILE_PERFECT, SKILL_PROFILE_PERFECT],
    )
    # skill_profiles should win — perfect accuracy
    result = game.throw_dart(20, "triple")
    assert result["missed"] is False
    assert result["actual_hit_type"] == "triple"


# ── Bull filtering ────────────────────────────────────────────────────


def test_bull_never_produces_triple():
    """When targeting Bull, no throw should ever produce a triple outcome."""
    game = DartsCricketGame(skill_profiles=[SKILL_PROFILE_PRO, SKILL_PROFILE_PRO])
    for _ in range(5000):
        game.reset()
        result = game.throw_dart(25, "double")
        if not result["missed"]:
            assert result["actual_hit_type"] != "triple", (
                "Bull double produced triple outcome"
            )


# ── Bull difficulty multiplier ────────────────────────────────────────


def test_bull_double_accuracy_penalty():
    """Pro double bull hit rate should be reduced by bull difficulty multiplier.

    Pro double_outcomes has double=0.40. With 0.75 multiplier → ~0.30.
    """
    game = DartsCricketGame(skill_profiles=[SKILL_PROFILE_PRO, SKILL_PROFILE_PRO])
    N = 5000
    hits = 0
    for _ in range(N):
        game.reset()
        result = game.throw_dart(25, "double")
        if not result["missed"] and result["actual_hit_type"] == "double":
            hits += 1
    rate = hits / N
    assert 0.20 <= rate <= 0.40, f"Bull double hit rate {rate:.3f} outside expected 0.20-0.40"


def test_bull_single_accuracy_penalty():
    """Pro single bull hit rate should be reduced by bull difficulty multiplier.

    Pro single_outcomes has single=0.96. After triple redistribution for bull
    and 0.75 multiplier, effective rate is lower.
    """
    game = DartsCricketGame(skill_profiles=[SKILL_PROFILE_PRO, SKILL_PROFILE_PRO])
    N = 5000
    hits = 0
    for _ in range(N):
        game.reset()
        result = game.throw_dart(25, "single")
        if not result["missed"] and result["actual_hit_type"] == "single":
            hits += 1
    rate = hits / N
    assert 0.60 <= rate <= 0.82, f"Bull single hit rate {rate:.3f} outside expected 0.60-0.82"


def test_bull_penalty_regular_target_unaffected():
    """Regular triple 20 should NOT be affected by bull difficulty multiplier."""
    game = DartsCricketGame(skill_profiles=[SKILL_PROFILE_PRO, SKILL_PROFILE_PRO])
    N = 5000
    hits = 0
    for _ in range(N):
        game.reset()
        result = game.throw_dart(20, "triple")
        if not result["missed"] and result["actual_hit_type"] == "triple":
            hits += 1
    rate = hits / N
    assert 0.37 <= rate <= 0.45, f"Triple 20 hit rate {rate:.3f} outside expected 0.37-0.45"


def test_bull_penalty_outcomes_sum_to_one():
    """After bull difficulty penalty, outcome probabilities must still sum to 1.0."""
    from config import BULL_DIFFICULTY_MULTIPLIER

    # Simulate what _apply_miss does for bull double with pro profile
    outcomes = dict(SKILL_PROFILE_PRO.double_outcomes)

    # Triple redistribution (bull has no triple)
    if "triple" in outcomes and outcomes["triple"] > 0:
        triple_mass = outcomes.pop("triple")
        remaining = sum(outcomes.values())
        if remaining > 0:
            for key in outcomes:
                outcomes[key] += triple_mass * (outcomes[key] / remaining)

    # Bull difficulty penalty
    primary = "double"
    if primary in outcomes and outcomes[primary] > 0:
        original = outcomes[primary]
        reduced = original * BULL_DIFFICULTY_MULTIPLIER
        freed = original - reduced
        outcomes[primary] = reduced
        outcomes["miss"] = outcomes.get("miss", 0.0) + freed

    total = sum(outcomes.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9), f"Outcomes sum to {total}, expected 1.0"


# ── Per-player profiles ──────────────────────────────────────────────


def test_per_player_different_profiles():
    """P0=perfect, P1=amateur → P0 always hits, P1 misses sometimes."""
    game = DartsCricketGame(
        skill_profiles=[SKILL_PROFILE_PERFECT, SKILL_PROFILE_AMATEUR]
    )

    # P0 always hits triples
    for _ in range(50):
        game.reset()
        result = game.throw_dart(20, "triple")
        assert result["missed"] is False
        assert result["actual_hit_type"] == "triple"

    # P1 should miss some triples (amateur triple hit = 15%)
    misses = 0
    n = 500
    for _ in range(n):
        game.reset()
        # Burn P0's darts to get to P1
        game.throw_dart(15, "single")
        game.throw_dart(15, "single")
        game.throw_dart(15, "single")
        assert game.current_player == 1
        result = game.throw_dart(20, "triple")
        if result["missed"] or result["actual_hit_type"] != "triple":
            misses += 1
    miss_rate = misses / n
    # Amateur triple hit is 15%, so miss rate ~85%
    assert miss_rate > 0.5, f"Amateur miss rate too low: {miss_rate:.3f}"


def test_miss_result_zero_marks():
    """When a throw misses, marks_added should be 0."""
    game = DartsCricketGame(skill_profiles=[SKILL_PROFILE_AMATEUR, SKILL_PROFILE_AMATEUR])
    found_miss = False
    for _ in range(200):
        game.reset()
        result = game.throw_dart(20, "triple")
        if result["missed"]:
            assert result["marks_added"] == 0
            found_miss = True
    assert found_miss, "No misses observed in 200 throws with amateur profile"


def test_downgrade_adds_correct_marks():
    """When a triple downgrades to single, marks_added should be 1."""
    game = DartsCricketGame(skill_profiles=[SKILL_PROFILE_AMATEUR, SKILL_PROFILE_AMATEUR])
    found_single = False
    for _ in range(500):
        game.reset()
        result = game.throw_dart(20, "triple")
        if not result["missed"] and result["actual_hit_type"] == "single":
            assert result["marks_added"] == 1
            found_single = True
    assert found_single, "No triple->single downgrades observed"
