"""Assemble the balanced V2 corpus manifest from the six capability families.

WHAT THIS DECIDES
-----------------
Which examples reach the optimizer, and in what proportion. That is a bigger
lever on the finished model than anything in the training loop, and it is
entirely CPU work — so it can be finished while a GPU is unavailable, and the
training run that follows is then a straight execution rather than a design
exercise.

Two rules drive the design.

**Weighted sampling, never concatenation.** Slakh contributes thousands of
segments; VocalSet contributes about ten hours. Concatenated, the vocal corpus
becomes rounding error and the model learns nothing new about singing — which
is the exact weakness V2 exists to fix. Families are therefore sampled to
target shares, and a family that cannot fill its share reports the shortfall
instead of quietly shrinking.

**Capped, stratified segments per track.** A fourth 60-second slice of one
arrangement teaches far less than the first slice of a new one, and costs the
same GPU-seconds to tensorize. Segments are spread across each track rather
than taken from the head.

    python scripts/assemble_balanced_corpus.py \\
        --families family_manifests.json \\
        --target 8000 \\
        --output processed/v2/manifests/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# Target shares from configs/datasets/v2_sprint.yaml. Kept here as the
# authoritative runtime default so a corpus built without the config still has
# the balance the design calls for.
TARGET_SHARES: dict[str, float] = {
    "fma": 0.33,                     # real production, genre diversity
    "slakh": 0.27,                   # arrangement, instrument interaction
    "vocals": 0.13,                  # singing realism — V1 had none at all
    "piano_strings": 0.13,           # acoustic realism, the weakest frozen-8 axis
    "guitar": 0.07,                  # articulation, voicing
    "drums": 0.07,                   # percussion realism
}

MAX_SEGMENTS_PER_TRACK = 3


def stratified_segments(segments: list[dict], cap: int, rng: random.Random) -> list[dict]:
    """Pick up to `cap` segments spread across a track, not clustered at the head.

    Taking the first N segments of every track biases the corpus toward
    intros — which are systematically sparser and quieter than the music that
    follows, and would teach the model that songs are mostly beginnings.
    """
    if len(segments) <= cap:
        return list(segments)
    step = len(segments) / cap
    picked = [segments[min(int(i * step), len(segments) - 1)] for i in range(cap)]
    # Deduplicate while preserving order in case rounding collided.
    seen, out = set(), []
    for seg in picked:
        key = id(seg)
        if key not in seen:
            seen.add(key)
            out.append(seg)
    while len(out) < cap and len(out) < len(segments):
        candidate = rng.choice(segments)
        if id(candidate) not in seen:
            seen.add(id(candidate))
            out.append(candidate)
    return out


def load_family(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", required=True,
                    help='JSON mapping family name -> manifest .jsonl path')
    ap.add_argument("--target", type=int, default=8000,
                    help="total segments; band 6000-10000")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--max-per-track", type=int, default=MAX_SEGMENTS_PER_TRACK)
    ap.add_argument("--share-ceiling", type=float, default=1.35,
                    help="a family may exceed its target share by at most this "
                         "factor when absorbing another family's shortfall")
    ap.add_argument("--max-repeat", type=int, default=2,
                    help="scarce families may reuse a segment up to N times to "
                         "reach their share; 1 disables upsampling")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    families: dict[str, str] = json.loads(Path(args.families).read_text())

    report: dict = {"target": args.target, "families": {}, "shortfalls": []}

    # Build every family's usable pool first, so the achievable corpus size is
    # known before anything is selected.
    pools: dict[str, list[dict]] = {}
    track_counts: dict[str, int] = {}
    for name in TARGET_SHARES:
        manifest = families.get(name)
        if not manifest or not Path(manifest).exists():
            pools[name], track_counts[name] = [], 0
            report["shortfalls"].append(f"{name}: no manifest")
            continue
        rows = load_family(Path(manifest))
        by_track: dict[str, list[dict]] = defaultdict(list)
        for i, row in enumerate(rows):
            by_track[str(row.get("track_id", f"__row{i}"))].append(row)
        pool: list[dict] = []
        for segs in by_track.values():
            pool.extend(stratified_segments(segs, args.max_per_track, rng))
        rng.shuffle(pool)
        pools[name] = pool
        track_counts[name] = len(by_track)

    # ------------------------------------------------------------- allocation
    #
    # Two failure modes bracket this decision, and both are worse than the
    # bounded middle taken here.
    #
    # Let a big family absorb every shortfall and FMA reaches 55% of a corpus
    # designed for 33% — vocals become rounding error and V2 repeats V1's
    # weakness. Enforce the ratios exactly instead, and the scarcest family
    # sets the size for everyone: 90 drum segments would cap an 8,000-segment
    # corpus at 1,285, discarding thousands of usable examples to protect a
    # proportion that was itself only an estimate.
    #
    # So: surplus families may expand, but only to `share x ceiling`, and
    # scarce families upsample their own pool up to `max_repeat` before
    # yielding any of their share. Repeats are marked, never disguised.
    quota = {n: int(round(args.target * sh)) for n, sh in TARGET_SHARES.items()}
    ceiling = {n: int(args.target * sh * args.share_ceiling)
               for n, sh in TARGET_SHARES.items()}
    unique = {n: min(len(pools[n]), quota[n]) for n in TARGET_SHARES}

    # Redistribute the shortfall across families that still hold unseen
    # segments, bounded by the ceiling so no single family can run away.
    deficit = args.target - sum(unique.values())
    granted: dict[str, int] = defaultdict(int)
    while deficit > 0:
        eligible = [n for n in TARGET_SHARES
                    if unique[n] < min(len(pools[n]), ceiling[n])]
        if not eligible:
            break
        per = max(1, deficit // len(eligible))
        for name in eligible:
            if deficit <= 0:
                break
            room = min(len(pools[name]), ceiling[name]) - unique[name]
            give = min(room, per, deficit)
            unique[name] += give
            granted[name] += give
            deficit -= give

    selected: list[dict] = []
    for name, share in TARGET_SHARES.items():
        pool = pools[name]
        take = [dict(r) for r in pool[:unique[name]]]

        # A family short of its own quota reuses its pool rather than handing
        # the slots to a larger family. Upsampling a small corpus risks
        # overfitting it; letting it vanish guarantees the model learns nothing
        # from it at all.
        repeats = 0
        want_more = quota[name] - len(take)
        if want_more > 0 and pool and args.max_repeat > 1:
            headroom = len(pool) * (args.max_repeat - 1)
            repeats = min(want_more, headroom)
            for i in range(repeats):
                dup = dict(pool[i % len(pool)])
                dup["repeat_index"] = (i // len(pool)) + 1
                take.append(dup)

        for row in take:
            row["family"] = name
        selected.extend(take)

        report["families"][name] = {
            "target": quota[name],
            "got": len(take),
            "unique": len(take) - repeats,
            "repeats": repeats,
            "granted": granted.get(name, 0),
            "tracks": track_counts[name],
            "pool": len(pool),
        }
        if len(take) < quota[name]:
            report["shortfalls"].append(
                f"{name}: wanted {quota[name]}, pool held {len(pool)} "
                f"(+{repeats} repeats) — short {quota[name] - len(take)}")

    rng.shuffle(selected)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report["total_selected"] = len(selected)
    report["total_unique"] = sum(f.get("unique", 0) for f in report["families"].values())
    report["actual_shares"] = {
        name: round(sum(1 for r in selected if r.get("family") == name) / max(len(selected), 1), 3)
        for name in TARGET_SHARES
    }

    print(f"selected {len(selected)} segments "
          f"({report['total_unique']} unique) -> {out}")
    for name, info in report["families"].items():
        got, want = info["got"], info["target"]
        share = report["actual_shares"][name]
        flag = "" if got >= want else "  << SHORT"
        extra = ""
        if info.get("repeats"):
            extra += f"  +{info['repeats']} repeat"
        if info.get("granted"):
            extra += f"  +{info['granted']} granted"
        print(f"  {name:15s} {got:5d} / {want:5d}   share {share:.1%}{extra}{flag}")
    for line in report["shortfalls"]:
        print("  !", line)

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))
        print("report:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
