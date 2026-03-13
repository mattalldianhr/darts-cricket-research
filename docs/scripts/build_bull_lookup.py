#!/usr/bin/env python3
"""Build bull_lookup.json from tournament data files.

Reads all tournament JSON files from the unequal data directory and produces
docs/site/data/bull_lookup.json with strategy recommendations keyed by
p1_mpr|p2_mpr|p1_bull|p2_bull.
"""

import json
import os
import re
from datetime import datetime, timezone

# Repo root for output path (worktree)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data lives only in the main repo (untracked, not in worktrees)
DATA_ROOT = '/Users/mattalldian/Personal/projects/Darts-Cricket/docs/data/unequal'

OUTPUT_PATH = os.path.join(REPO_ROOT, 'docs', 'site', 'data', 'bull_lookup.json')

# Directory-name → (p1_bull_mult, p2_bull_mult) for symmetric dirs without JSON fields
SYMMETRIC_BULL_MAP = {
    'bull_0.25': (0.25, 0.25),
    'bull_0.5': (0.5, 0.5),
    'post_bull': (0.75, 0.75),
    # root-level files → standard bull (1.0)
    '__root__': (1.0, 1.0),
}

# Regex to parse asymmetric directories: bull_p1_X_p2_Y
ASYM_PATTERN = re.compile(r'^bull_p1_([0-9.]+)_p2_([0-9.]+)$')


def get_bull_mults_for_dir(dirpath: str, dirname: str) -> tuple[float, float] | None:
    """Return (p1_bull, p2_bull) for a given directory, or None to skip."""
    if dirname == 'unequal':
        # Skip nested duplicate directory
        return None
    if dirname in SYMMETRIC_BULL_MAP:
        return SYMMETRIC_BULL_MAP[dirname]
    m = ASYM_PATTERN.match(dirname)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def build_h2h(data: dict) -> dict | None:
    """Build head-to-head best-response table from the 22x22 matrix.

    Returns for each player's top-5 strategies:
      - their win rate vs the opponent's best counter
      - what that counter is
    """
    matrix = data.get('matrix')
    strats = data.get('strategies')
    if not matrix or not strats:
        return None

    n = len(strats)
    strat_idx = {s: i for i, s in enumerate(strats)}

    def player_h2h(rankings_key: str, is_p1: bool) -> list:
        rankings = data.get(rankings_key, [])[:5]
        result = []
        for r in rankings:
            name = r['name']
            i = strat_idx.get(name)
            if i is None:
                continue
            if is_p1:
                # P1 plays row i. P2's best counter = col j that minimizes matrix[i][j]
                row = matrix[i]
                worst = min(row)
                worst_j = row.index(worst)
                counter = strats[worst_j]
            else:
                # P2 plays col j=i. P1's best counter = row k that maximizes matrix[k][i]
                col_vals = [matrix[k][i] for k in range(n)]
                best_p1 = max(col_vals)
                worst = round(100 - best_p1, 1)  # P2's WR when P1 plays optimally
                best_k = col_vals.index(best_p1)
                counter = strats[best_k]
            result.append({
                'name': name,
                'avg': round(r['avg'], 1),
                'vs_counter': round(worst, 1),
                'counter': counter
            })
        return result

    return {
        'p1': player_h2h('rankings_p1', True),
        'p2': player_h2h('rankings_p2', False),
    }


def build_lookup() -> dict:
    lookup: dict[str, dict] = {}
    skipped = 0
    processed = 0

    # --- Process root-level JSON files (bull = 1.0|1.0) ---
    root_bull = SYMMETRIC_BULL_MAP['__root__']
    for fname in os.listdir(DATA_ROOT):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(DATA_ROOT, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        if 'rankings_p1' not in data:
            skipped += 1
            continue

        p1_mpr = data.get('p1_mpr_target')
        p2_mpr = data.get('p2_mpr_target')
        if p1_mpr is None or p2_mpr is None:
            skipped += 1
            continue

        p1_bull, p2_bull = root_bull
        key = f"{p1_mpr}|{p2_mpr}|{p1_bull}|{p2_bull}"
        entry = {
            'p1_best': data['rankings_p1'][:5],
            'p2_best': data['rankings_p2'][:5],
            'avg_turns': data.get('avg_turns'),
            'avg_darts': data.get('avg_darts'),
        }
        h2h = build_h2h(data)
        if h2h:
            entry['h2h'] = h2h
        lookup[key] = entry
        processed += 1

    # --- Process subdirectories ---
    for dirname in os.listdir(DATA_ROOT):
        dirpath = os.path.join(DATA_ROOT, dirname)
        if not os.path.isdir(dirpath):
            continue

        bull_mults = get_bull_mults_for_dir(dirpath, dirname)
        if bull_mults is None:
            print(f"  Skipping directory: {dirname}")
            continue

        p1_bull, p2_bull = bull_mults

        for fname in os.listdir(dirpath):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(dirpath, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                skipped += 1
                continue

            if 'rankings_p1' not in data:
                skipped += 1
                continue

            p1_mpr = data.get('p1_mpr_target')
            p2_mpr = data.get('p2_mpr_target')
            if p1_mpr is None or p2_mpr is None:
                skipped += 1
                continue

            # For asymmetric dirs, prefer JSON fields if present
            if data.get('p1_bull_mult') is not None:
                p1_bull = data['p1_bull_mult']
            if data.get('p2_bull_mult') is not None:
                p2_bull = data['p2_bull_mult']

            key = f"{p1_mpr}|{p2_mpr}|{p1_bull}|{p2_bull}"
            entry = {
                'p1_best': data['rankings_p1'][:5],
                'p2_best': data['rankings_p2'][:5],
                'avg_turns': data.get('avg_turns'),
                'avg_darts': data.get('avg_darts'),
            }
            h2h = build_h2h(data)
            if h2h:
                entry['h2h'] = h2h
            lookup[key] = entry
            processed += 1

    print(f"Processed: {processed}, Skipped: {skipped}")
    return lookup


def main():
    print(f"Reading data from: {DATA_ROOT}")
    print(f"Output path:       {OUTPUT_PATH}")
    print()

    lookup = build_lookup()

    assert len(lookup) > 1000, f"Expected >1000 entries, got {len(lookup)}"

    # Collect unique MPR and bull levels from keys
    mpr_set: set[float] = set()
    bull_set: set[float] = set()
    for key in lookup:
        parts = key.split('|')
        mpr_set.add(float(parts[0]))
        mpr_set.add(float(parts[1]))
        bull_set.add(float(parts[2]))
        bull_set.add(float(parts[3]))

    output = {
        'metadata': {
            'generated': datetime.now(timezone.utc).isoformat(),
            'total_entries': len(lookup),
        },
        'lookup': lookup,
        'mpr_levels': sorted(mpr_set),
        'bull_levels': sorted(bull_set),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nWrote {len(lookup)} entries to {OUTPUT_PATH}")
    print(f"File size: {size_kb:.1f} KB")


if __name__ == '__main__':
    main()
