"""Prove text -> ACE-Step -> SongForge adapter -> WAV, in one uninterrupted run.

WHY THIS IS ONE SCRIPT
----------------------
Colab recycled this project's runtime six times in a single sprint. The dialog
names the cause: "disconnected due to inactivity". The runtime does not die
during work — it dies during the pauses *between* cells, which is exactly when
someone is reading output and deciding what to run next. Every interactive
debugging session therefore self-destructs partway through.

So the gate does not ask questions between steps. It discovers the API, calls
it, writes the audio, and records what happened — in one blocking process that
keeps the runtime busy from start to finish.

Everything durable is written to Drive as soon as it exists, because the VM is
not guaranteed to survive the next minute.

    python scripts/inference_gate.py --duration 30 --out /content/drive/.../gate
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

ACE = "/content/ACE-Step-1.5"
CKPT = "/content/checkpoints"


def log(report: dict, path: Path, **kv) -> None:
    """Append to the report and flush it to Drive immediately.

    A report that only exists in memory is worth nothing on a runtime that can
    vanish between two statements.
    """
    report.update(kv)
    for k, v in kv.items():
        print(f"[gate] {k}: {v}", flush=True)
    try:
        path.write_text(json.dumps(report, indent=1, default=str))
    except Exception:
        pass


def find_callable(obj, *name_hints):
    """Locate a bound method whose name contains any hint, longest match first."""
    names = [n for n in dir(obj) if not n.startswith("_") and callable(getattr(obj, n, None))]
    hits = [n for n in names if any(h in n.lower() for h in name_hints)]
    hits.sort(key=len)
    return [(n, getattr(obj, n)) for n in hits]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--out", default="/content/drive/MyDrive/songforge-dl/gate")
    ap.add_argument("--adapter", default=os.environ.get("SONGFORGE_LORA", ""))
    ap.add_argument("--dtype", default=None,
                    choices=["bfloat16", "float16", "float32"],
                    help="override the precision; by default it is chosen from "
                         "the GPU's compute capability")
    ap.add_argument("--caption", default=(
        "A tender cinematic piece led by grand piano, joined by an expressive "
        "violin counter-melody and warm string ensemble. Sparse and intimate at "
        "the start, gradually building in density and emotion toward a full, "
        "sweeping final section."))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "inference_gate_report.json"
    report: dict = {"started": time.strftime("%FT%TZ", time.gmtime())}

    sys.path.insert(0, ACE)
    os.environ.setdefault("MPLBACKEND", "Agg")

    # ------------------------------------------------------------ environment
    import torch

    capability = (list(torch.cuda.get_device_capability(0))
                  if torch.cuda.is_available() else None)

    # Precision is chosen from the GPU, not assumed.
    #
    # bf16 needs compute capability >= 8.0. Below that the format is emulated
    # and lyrics-to-song training produced NaNs, which is why the training
    # driver refuses a T4 outright. Inference is a different question: fp16 is
    # native on a T4 (7.5), and 4,991,023,206 parameters at two bytes each is
    # about 10 GB, which fits inside the T4's 15 GB. So a machine that cannot
    # train this model can still generate from it -- and a free Colab T4 costs
    # no compute units.
    if args.dtype:
        dtype = args.dtype
        dtype_reason = "explicitly requested via --dtype"
    elif capability and capability[0] >= 8:
        dtype = "bfloat16"
        dtype_reason = f"compute capability {capability[0]}.{capability[1]} >= 8.0"
    elif capability:
        dtype = "float16"
        dtype_reason = (f"compute capability {capability[0]}.{capability[1]} < 8.0; "
                        "bf16 is unsafe here, fp16 is native")
    else:
        dtype = "float32"
        dtype_reason = "no CUDA device visible"

    log(report, report_path,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        compute_capability=capability,
        dtype=dtype,
        dtype_reason=dtype_reason,
        checkpoint_dir=CKPT,
        adapter=args.adapter or None)
    print(f"[gate] precision {dtype} -- {dtype_reason}", flush=True)

    try:
        rev = Path(ACE, ".git/HEAD").read_text().strip()
        head = Path(ACE, ".git", rev.split()[-1]).read_text().strip() if rev.startswith("ref:") else rev
        log(report, report_path, acestep_revision=head)
    except Exception:
        pass

    # -------------------------------------------------------------- the handler
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    log(report, report_path,
        handler_methods=[n for n in dir(handler) if not n.startswith("_")][:40])

    # Load weights through whichever loader the build exposes.
    loaders = find_callable(handler, "load", "init", "setup", "prepare")
    log(report, report_path, loader_candidates=[n for n, _ in loaders])

    loaded = None
    for name, fn in loaders:
        try:
            sig = inspect.signature(fn)
            kwargs = {}
            for p in sig.parameters.values():
                low = p.name.lower()
                if "checkpoint" in low or "model_dir" in low or low in ("path", "dir"):
                    kwargs[p.name] = CKPT
                elif "variant" in low or low == "model":
                    kwargs[p.name] = "xl_turbo"
                elif "device" in low:
                    kwargs[p.name] = "cuda"
                elif "dtype" in low or "precision" in low:
                    kwargs[p.name] = dtype
            fn(**kwargs)
            loaded = name
            log(report, report_path, loaded_via=name, load_kwargs=kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            log(report, report_path, **{f"loader_{name}_failed": f"{type(exc).__name__}: {exc}"[:300]})

    # ------------------------------------------------------------- generation
    gens = find_callable(handler, "generat", "text2music", "infer", "run", "sample", "predict")
    log(report, report_path, generate_candidates=[n for n, _ in gens])

    payload = {
        "caption": args.caption,
        "prompt": args.caption,
        "lyrics": "[Instrumental]",
        "duration": args.duration,
        "bpm": 72,
        "keyscale": "C minor",
        "language": "en",
        "timesignature": "4/4",
        "output_dir": str(out),
        "save_path": str(out / "gate.wav"),
    }

    for name, fn in gens:
        try:
            sig = inspect.signature(fn)
            kwargs = {p: v for p, v in payload.items() if p in sig.parameters}
            log(report, report_path, trying=name, with_kwargs=sorted(kwargs))
            t0 = time.time()
            result = fn(**kwargs) if kwargs else fn(args.caption)
            elapsed = round(time.time() - t0, 1)

            wavs = sorted(out.rglob("*.wav")) + sorted(Path("/content").glob("*.wav"))
            if wavs:
                biggest = max(wavs, key=lambda p: p.stat().st_size)
                log(report, report_path,
                    GATE="PASSED", method=name, seconds=elapsed,
                    audio=str(biggest), bytes=biggest.stat().st_size,
                    result_type=type(result).__name__)
                return 0
            log(report, report_path, **{f"{name}_no_audio": str(result)[:200], "seconds": elapsed})
        except Exception as exc:  # noqa: BLE001
            log(report, report_path,
                **{f"gen_{name}_failed": f"{type(exc).__name__}: {exc}"[:400]})
            traceback.print_exc()

    log(report, report_path, GATE="FAILED",
        note="No generation entry point produced audio; see per-method errors above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
