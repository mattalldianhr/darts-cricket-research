# Optimal Strategy in Darts Cricket

A computational study extending [Frongello (2018)](https://www.frongello.com/cricket/) with realistic skill profiles, bull accuracy modeling, and new experimental strategies. Over **18 billion simulated games** across 22 strategies, 11 skill levels, and 16 bull accuracy conditions.

**Live site:** [https://darts.mattalldian.com](https://darts.mattalldian.com)

## Key Findings

1. **The right strategy depends on your skill level.** There is no single best strategy. At low MPR (beginners), strategy differences are small and luck dominates. At mid MPR (league players), E2 (Honeypot) and S2 (Score Then Cover) compete closely. At high MPR (pros), Phase Switch pulls ahead. The optimal approach shifts across the full skill spectrum.

2. **Bull accuracy is a hidden dimension that changes everything.** When we modeled bull as harder to hit than numbers (which it is on a real board), the entire strategy landscape shifted. E1 (Early Bull) -- previously the #1 strategy at 9 of 11 skill levels -- collapsed to dead last at MPR 4.0+. A single parameter change reshuffled the competitive field more than any other factor in the study.

3. **Your optimal strategy depends on your opponent's stats too.** Asymmetric bull accuracy between players creates win rate spreads of 16-39 percentage points (depending on skill level) and changes which strategy is optimal. Strategy recommendations change in ~77% of asymmetric slots.

4. **Frongello's core principles hold across all conditions.** Score first, then cover. Never chase. Weaker players want shorter games. These findings replicate across 11 skill levels and 16 bull conditions. An RL agent independently rediscovered these principles through pure trial and error, providing strong confirmation they reflect genuine strategic truth.

5. **A new strategy beats everything at high skill.** Phase Switch -- score aggressively early, then irreversibly commit to closing -- ranks #1 from MPR 3.6 upward and beats S2 (Score Then Cover) head-to-head from ~3.0. The key innovation is eliminating oscillation: once you decide to close out, you never go back. Its advantage grows with skill, reaching +5.0pp vs S2 at pro level.

6. **An RL agent discovered something no human designed.** A branching actor-critic agent, trained from scratch against 13 hard bots, learned to adapt its opening based on turn order: 20 when going first, 18 when going second. None of the 28 hand-crafted strategies make this distinction. The agent also independently confirmed Frongello's "never chase" and "score before cover" principles.

## Methodology

- **22 strategies**: 17 from Frongello (S1-S17), 4 experimental (E1-E4, E2 being "Honeypot"), and Phase Switch (PS)
- **11 skill levels**: MPR 0.8 through 5.6, calibrated to real player statistics
- **16 bull conditions**: 4 equal-bull multipliers (0.25x, 0.50x, 0.75x, 1.00x) plus 12 asymmetric pairings
- **20,000 games per matchup**: Each of the 22x22 = 484 strategy pairs plays 20,000 games at each condition
- **Skill profiles**: Realistic miss models where hit probability varies by dart type (single/double/triple) and skill level, with bull treated as geometrically harder due to its smaller target area
- **1,936 total tournaments** generating 18B+ individual game simulations

See the [Methodology page](https://darts.mattalldian.com/methodology.html) on the live site for full details.

## Repository Structure

```
darts-cricket-research/
├── docs/
│   ├── site/              # Cloudflare Pages deployment root
│   │   ├── *.html         # 8 pages (index, strategies, results, etc.)
│   │   ├── css/           # Stylesheet
│   │   ├── js/            # Interactive tools (advisor, bull advisor)
│   │   └── data/          # Tournament data (served to site)
│   ├── data/              # Same tournament data (for local audit)
│   │   ├── post_bull/     # 11 tournaments with bull difficulty enabled
│   │   ├── pre_bull/      # 11 tournaments without bull difficulty
│   │   ├── unequal/       # 121+ unequal-skill matchup files
│   │   ├── bull_lookup.json           # Precomputed bull advisor lookup
│   │   └── strategy_test_fixtures.json
│   └── scripts/
│       └── build_bull_lookup.py       # Generates bull_lookup.json
├── tests/                 # Test suite
├── game.py                # Core cricket game engine
├── config.py              # Constants, skill profiles, defaults
├── strategies.py          # All 22 strategy bots (S1-S17, E1-E4, PS)
├── validation.py          # Tournament runner and principle tests
├── analysis.py            # Strategy introspection tools
├── advisor.py             # Throw recommendation engine
├── mpr_sweep.py           # MPR calibration and matchup runner
├── run_full_tournament.py # Generates equal-skill tournament data
├── run_unequal_tournament.py  # Generates unequal-skill data
├── refined_grid.py        # Phase Switch grid search
├── play.py                # Human vs AI play mode
├── main.py                # CLI entry point
├── requirements.txt
└── LICENSE
```

## Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run a full tournament at pro skill level

```bash
python run_full_tournament.py
```

This runs all 22x22 strategy matchups at each of 11 skill levels with 20,000 games per matchup. Output is written as JSON to `docs/data/`.

### Run unequal-skill matchups

```bash
python run_unequal_tournament.py
```

Generates matchup data for all MPR pairings across bull conditions.

### Run tests

```bash
pytest tests/
```

### Phase Switch grid search

```bash
python refined_grid.py
```

Searches the (threshold, min_closed) parameter space to find the optimal Phase Switch configuration at each skill level.

## Verification

To audit the published results against the simulation code:

1. **Run a tournament** using `run_full_tournament.py` and compare the output JSON against the files in `docs/data/post_bull/` or `docs/data/pre_bull/`.
2. **Check specific matchups**: Each tournament JSON contains the full 22x22 win-rate matrix, head-to-head records, and rankings for that skill level.
3. **Run the test suite**: `pytest tests/` validates game logic, strategy behavior, skill profiles, and analytical tools against known fixtures.
4. **Strategy fixtures**: `docs/data/strategy_test_fixtures.json` contains deterministic test cases for strategy decision verification.

Note: Due to the stochastic nature of simulation, regenerated results will differ slightly from published data but should converge to the same rankings and conclusions at 20,000 games per matchup.

## What's Not Included

The reinforcement learning agent code (A2C, DQN, Q-learning) is not included in this repository. The RL agent's findings -- including its turn-order-dependent strategy and independent confirmation of Frongello's principles -- are discussed on the [AI Agent page](https://darts.mattalldian.com/ai-agent.html) of the live site.

## Credits

- **Andrew Frongello** (2018) -- [Optimal Strategy in Darts Cricket](https://www.frongello.com/cricket/). The foundational work that defined the 17 base strategies (S1-S17) and established the analytical framework for cricket strategy evaluation.
- **Matt Alldian** (2026) -- Extended Frongello's work with realistic skill profiles, bull accuracy modeling, experimental strategies (E1-E4), Phase Switch, unequal-skill matchups, and reinforcement learning validation.

## License

MIT License. See [LICENSE](LICENSE).
