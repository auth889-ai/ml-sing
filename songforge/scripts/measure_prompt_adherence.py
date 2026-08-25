"""Measure whether generated audio actually matches the prompt that asked for it.

WHY THIS IS THE CENTRAL EXPERIMENT
----------------------------------
ACE-Step 1.5 beats Suno v5 on raw audio quality (SongEval 8.12 vs 7.87) and
loses to it on the two axes that measure doing what you were asked: style
alignment 39.1 vs 46.8, lyric alignment 26.3 vs 34.2. Better sound, worse
obedience. That published gap -- not audio fidelity -- is what SongForge's
adapter and corpus exist to close.

Closing it is a claim. This turns it into a number.

CLAP embeds audio and text into one space, so the cosine similarity between a
generated track and the prompt that produced it is a direct measure of prompt
adherence. Comparing that score for the same prompt at the same seed, with and
without our adapter, isolates the adapter as the only difference.

WHAT MAKES THIS EVIDENCE RATHER THAN A DEMO
- Paired. Every prompt is scored under both conditions, so per-prompt
  difficulty cancels out instead of becoming noise.
- Bootstrapped. A mean improvement over a handful of prompts means nothing
  without an interval; four prompts can move 10% on luck alone.
- Reports losses. Per-prompt deltas are printed including the negative ones.
  An adapter that helps orchestral and wrecks electronic is not an improvement,
  it is a trade, and the corpus design needs to know which.
- Held-out only, ideally. Scoring prompts the adapter trained on measures
  memorisation. benchmarks/generalization_prompts.yaml keeps a held-out tier
  for exactly this, and CI fails if those ids leak into the codebase.

    python scripts/measure_prompt_adherence.py \\
        --pairs adherence_pairs.json --report adherence.json

pairs JSON: [{"prompt": "...", "base": "a.wav", "v1": "b.wav"}, ...]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load_clap(model_id: str, device: str):
    """CLAP maps audio and text into a shared space; that is the whole trick."""
    import torch
    from transformers import ClapModel, ClapProcessor

    model = ClapModel.from_pretrained(model_id).to(device).eval()
    processor = ClapProcessor.from_pretrained(model_id)
    return model, processor, torch


def embed_audio(model, processor, torch, path: Path, device: str, sr: int = 48000):
    import numpy as np
    import soundfile as sf

    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if file_sr != sr:
        duration = mono.shape[0] / file_sr
        target_n = int(round(duration * sr))
        mono = np.interp(
            np.linspace(0.0, duration, target_n, endpoint=False),
            np.linspace(0.0, duration, mono.shape[0], endpoint=False),
            mono).astype("float32")
    inputs = processor(audios=mono, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    return torch.nn.functional.normalize(emb, dim=-1)


def embed_text(model, processor, torch, text: str, device: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(emb, dim=-1)


def bootstrap_ci(deltas: list[float], iterations: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference.

    With a handful of prompts the mean is unstable, and quoting it alone
    invites a conclusion the data does not support. The interval is the honest
    version of the same number.
    """
    if len(deltas) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(int(0.975 * len(means)), len(means) - 1)]
    return (lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", required=True,
                    help='JSON list of {prompt, base, v1} with wav paths')
    ap.add_argument("--model", default="laion/clap-htsat-unfused")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    if not pairs:
        raise SystemExit("no pairs given")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, torch = load_clap(args.model, device)

    rows, deltas = [], []
    for item in pairs:
        prompt = item["prompt"]
        text = embed_text(model, processor, torch, prompt, device)
        row = {"prompt": prompt[:90]}
        for condition in ("base", "v1", "v2"):
            path = item.get(condition)
            if not path or not Path(path).exists():
                continue
            audio = embed_audio(model, processor, torch, Path(path), device)
            row[condition] = round(float((audio @ text.T).item()), 4)
        if "base" in row and "v1" in row:
            row["delta"] = round(row["v1"] - row["base"], 4)
            deltas.append(row["delta"])
        rows.append(row)

    report = {"model": args.model, "device": device, "pairs": len(rows),
              "rows": rows}

    print(f"CLAP prompt adherence  ({args.model})")
    print("-" * 74)
    print(f"  {'prompt':<44} {'base':>7} {'V1':>7} {'delta':>8}")
    for row in rows:
        print(f"  {row['prompt'][:44]:<44} "
              f"{row.get('base', float('nan')):>7.4f} "
              f"{row.get('v1', float('nan')):>7.4f} "
              f"{row.get('delta', float('nan')):>+8.4f}")
    print("-" * 74)

    if deltas:
        mean = sum(deltas) / len(deltas)
        lo, hi = bootstrap_ci(deltas, args.bootstrap, args.seed)
        wins = sum(1 for d in deltas if d > 0)
        report.update(mean_delta=round(mean, 4),
                      ci95=[round(lo, 4), round(hi, 4)],
                      wins=wins, losses=len(deltas) - wins,
                      n=len(deltas))
        print(f"  mean delta {mean:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"{wins} up / {len(deltas) - wins} down of {len(deltas)}")

        # The interval, not the mean, decides what may be claimed.
        if lo > 0:
            verdict = "ADAPTER IMPROVES prompt adherence (CI excludes zero)"
        elif hi < 0:
            verdict = "ADAPTER HARMS prompt adherence (CI excludes zero)"
        else:
            verdict = ("INCONCLUSIVE -- CI spans zero. More prompts needed "
                       "before claiming either direction.")
        report["verdict"] = verdict
        print(f"  {verdict}")
        if len(deltas) < 20:
            print(f"  NOTE: {len(deltas)} prompts is a small sample. Treat this "
                  "as a signal to investigate, not a published result.")
    else:
        print("  no base/V1 pairs to compare")

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))
        print(f"  report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
