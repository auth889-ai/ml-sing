"""Render audible before/after pairs through SongForge's own neural audio codec.

WHY THIS EXISTS
---------------
The generation path needs a GPU with compute capability >= 8.0, because the
ACE-Step foundation is 4,991,023,206 parameters and bf16 below Ampere produces
NaNs. That is a hard dependency, and when no such GPU is available there is
nothing to listen to.

This codec is ours and it is 5,068,481 parameters -- three orders of magnitude
smaller. It runs on a CPU in seconds. So the one thing that can always be
demonstrated is the component we actually built and trained: audio in, encoded
to discrete RVQ tokens, decoded back to a waveform.

What this shows is *reconstruction*, not generation, and the two must not be
confused. The claim it supports is narrow and true: our codec was trained from
random initialisation and learned to represent real music. The frozen
acceptance run (docs/CODEC_RESULTS_FROZEN.md) put held-out SNR at +3.64 dB,
up from -2.81 dB at initialisation -- audible, and a long way from
transparent. Expect to hear the music through obvious artefacts. Printing the
per-file SNR alongside the audio is the point: a listener can check the number
against their own ears.

    python scripts/demo_codec_reconstruction.py \\
        --checkpoint outputs/codec_m03_acceptance/<run>/checkpoint.pt \\
        --audio some.wav more.wav \\
        --output-dir demo_audio/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from songforge.models.codec.model import NeuralCodec


def model_kwargs_from_config(config: dict) -> dict:
    """Rebuild the exact architecture the checkpoint was trained with.

    The checkpoint carries its own config, so the demo never has to be told
    which variant it is loading -- and can never silently load weights into a
    differently-shaped model.
    """
    model = config.get("model", config)
    return {
        "sample_rate": int(model.get("sample_rate", 24000)),
        "channels": int(model.get("channels", 1)),
        "base_channels": int(model.get("base_channels", 32)),
        "latent_dim": int(model.get("latent_dim", 64)),
        "codebook_size": int(model.get("codebook_size", 256)),
        "num_quantizers": int(model.get("num_quantizers", 4)),
        "strides": tuple(model.get("strides", [2, 4, 5, 5])),
        "commitment_weight": float(model.get("commitment_weight", 0.25)),
    }


def load_mono(path: Path, target_sr: int, seconds: float | None) -> np.ndarray:
    """Read an audio file as mono float32 at the codec's sample rate."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)

    if sr != target_sr:
        # Linear resampling. The codec is the subject of this demo, so the
        # resampler is deliberately dependency-free; for a quality comparison
        # rather than a demonstration, resample offline with a proper filter.
        duration = audio.shape[0] / sr
        target_n = int(round(duration * target_sr))
        source_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        target_x = np.linspace(0.0, duration, num=target_n, endpoint=False)
        audio = np.interp(target_x, source_x, audio).astype("float32")

    if seconds:
        audio = audio[: int(seconds * target_sr)]
    return audio


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-noise ratio of the reconstruction against the original."""
    n = min(reference.shape[0], estimate.shape[0])
    reference, estimate = reference[:n], estimate[:n]
    noise = reference - estimate
    signal_power = float(np.sum(reference ** 2))
    noise_power = float(np.sum(noise ** 2))
    if noise_power <= 0 or signal_power <= 0:
        return float("inf") if noise_power <= 0 else float("-inf")
    return 10.0 * math.log10(signal_power / noise_power)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint.pt from a codec training run")
    ap.add_argument("--audio", nargs="+", required=True,
                    help="input audio files")
    ap.add_argument("--output-dir", default="demo_audio")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="trim each input to this length; 0 keeps the whole file")
    ap.add_argument("--device", default="cpu",
                    help="cpu is the point of this script, but cuda works")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = blob.get("config", {})
    kwargs = model_kwargs_from_config(config)

    model = NeuralCodec(**kwargs).to(device)
    # strict=True on purpose: a partially-loaded codec would still emit audio,
    # just quietly worse audio, and the SNR would be blamed on training.
    model.load_state_dict(blob["model"], strict=True)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    sr = kwargs["sample_rate"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"codec        {params:,} params, {sr} Hz, "
          f"{kwargs['num_quantizers']}x{kwargs['codebook_size']} RVQ")
    print(f"trained step {blob.get('step', '?')}   run {blob.get('run_id', '?')}")
    print(f"device       {device}")
    print("-" * 66)

    rows = []
    for path in args.audio:
        src = Path(path)
        if not src.exists():
            print(f"  {src.name:30s} MISSING")
            continue

        audio = load_mono(src, sr, args.seconds or None)
        if audio.size == 0:
            print(f"  {src.name:30s} EMPTY")
            continue

        x = torch.from_numpy(audio)[None, None, :].to(device)
        with torch.no_grad():
            out = model(x)
        recon = out["reconstruction"][0, 0].cpu().numpy()

        measured = snr_db(audio, recon)
        stats = model.compression_stats(audio.shape[0])

        stem = src.stem
        original_path = out_dir / f"{stem}__original.wav"
        recon_path = out_dir / f"{stem}__codec.wav"
        sf.write(original_path, audio, sr)
        sf.write(recon_path, np.clip(recon, -1.0, 1.0), sr)

        rows.append({
            "input": str(src),
            "original": str(original_path),
            "reconstruction": str(recon_path),
            "seconds": round(audio.shape[0] / sr, 2),
            "snr_db": round(measured, 2),
            **{k: v for k, v in stats.items()},
        })
        print(f"  {stem[:28]:30s} {audio.shape[0]/sr:5.1f}s   "
              f"SNR {measured:+6.2f} dB")

    print("-" * 66)
    if rows:
        values = [r["snr_db"] for r in rows]
        print(f"  {len(rows)} file(s)   mean SNR {sum(values)/len(values):+.2f} dB")
        print(f"  audio written to {out_dir}/")
        print("\n  Reconstruction, not generation. Our codec, trained from")
        print("  random init; the foundation model is not involved here.")
    else:
        print("  nothing processed")
        return 1

    if args.report:
        Path(args.report).write_text(json.dumps(rows, indent=1))
        print(f"  report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
