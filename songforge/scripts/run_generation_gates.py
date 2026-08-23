"""Prove real generated audio, base first, then the adapter, then a comparison.

WHAT THIS ANSWERS, IN ORDER
---------------------------
1. base     -- does the frozen foundation generate audio at all on this machine?
2. adapter  -- does our V1 LoKr load, and how many modules actually matched?
3. matrix   -- base vs V1 on four deliberately different prompts, identical
               settings, so any difference is the adapter and nothing else.
4. strength -- LoKr scale 0.2/0.4/0.6/0.8, to find where adaptation helps
               before it starts eating the foundation's breadth.

The ordering is the point. A failure at stage 1 is an environment problem; a
failure at stage 2 with stage 1 passing is an adapter problem. Running them
together, as the previous gate did, makes those two indistinguishable.

WHAT IT REFUSES TO DO
---------------------
- Report a pass on zero matched LoKr modules. An adapter that matches nothing
  loads without error and generates audio identical to base, which reads as
  "the adapter did nothing useful" when it in fact never applied. That is a
  hard failure here, not a warning.
- Vary anything between the base and V1 renders of the same prompt. Same seed,
  duration, steps, guidance, precision. Otherwise the comparison is worthless.
- Judge the adapter by training loss. Loss fell during V1 training; that says
  the optimizer worked, not that the audio is better. Only the WAVs decide.

Every stage writes its JSON before and after the work, and stages already
holding output are skipped, so a dropped session resumes rather than restarts.

    python scripts/run_generation_gates.py --stage all \\
        --ace-dir ~/ACE-Step-1.5 --checkpoint-dir ~/checkpoints \\
        --adapter ~/adapters/v1_best --out ~/gate_out
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

# The four prompts are deliberately far apart: acoustic/orchestral, loud band
# with a violin lead, sparse and intimate, fully electronic. A model that has
# collapsed onto one sound will reveal it across this spread far faster than
# across four variations of the same idea.
PROMPT_MATRIX = [
    ("A_cinematic_piano_violin",
     "A tender cinematic instrumental. Grand piano takes the lead from the "
     "opening, joined by an expressive violin counter-melody over warm "
     "sustained strings. It begins sparse and intimate, then builds steadily "
     "into a sweeping, full-hearted climax."),
    ("B_prog_rock_violin_lead",
     "An energetic instrumental progressive rock track led by a soaring violin "
     "over distorted electric guitars, driving bass and live drums played with "
     "force. Complex, shifting rhythms and a dramatic, decisive ending."),
    ("C_sparse_acoustic",
     "A sparse acoustic piece. Finger-picked acoustic guitar carries the "
     "melody, with soft piano underneath. Intimate, warm, close-mic'd "
     "production with plenty of space between the notes."),
    ("D_electronic_cinematic",
     "A dark electronic cinematic track. Pulsing synth bass and crisp "
     "electronic drums under wide atmospheric pads, rising through the middle "
     "into a large orchestral climax."),
]

ADAPTER_STRENGTHS = (0.2, 0.4, 0.6, 0.8)


# ----------------------------------------------------------------- reporting

class Report:
    """A JSON report that is on disk before it is needed, not after."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def set(self, **kv):
        self.data.update(kv)
        for k, v in kv.items():
            print(f"[gate] {k}: {v}", flush=True)
        self.flush()

    def flush(self):
        try:
            self.path.write_text(json.dumps(self.data, indent=1, default=str))
        except Exception:  # noqa: BLE001
            pass


def find_callable(obj, *hints):
    names = [n for n in dir(obj)
             if not n.startswith("_") and callable(getattr(obj, n, None))]
    hits = [n for n in names if any(h in n.lower() for h in hints)]
    hits.sort(key=len)
    return [(n, getattr(obj, n)) for n in hits]


def adapter_key_count(path: Path) -> tuple[int, list[str]]:
    """Count LoKr/LoRA tensors in an adapter checkpoint, without a GPU."""
    import torch

    files = []
    p = Path(path)
    if p.is_dir():
        for pattern in ("*.safetensors", "*.pt", "*.ckpt", "*.bin"):
            files.extend(sorted(p.rglob(pattern)))
    elif p.exists():
        files = [p]
    if not files:
        return 0, []

    keys: list[str] = []
    for f in files:
        try:
            if f.suffix == ".safetensors":
                from safetensors.torch import load_file
                blob = load_file(str(f))
            else:
                blob = torch.load(f, map_location="cpu", weights_only=False)
                for k in ("state_dict", "lokr", "adapter", "model"):
                    if isinstance(blob, dict) and k in blob and isinstance(blob[k], dict):
                        blob = blob[k]
                        break
            if isinstance(blob, dict):
                keys.extend(k for k in blob
                            if any(m in str(k).lower()
                                   for m in ("lokr", "lycoris", "lora")))
        except Exception:  # noqa: BLE001
            continue
    return len(keys), keys[:12]


# ------------------------------------------------------------------ the model

class Foundation:
    """One loaded ACE-Step handler, plus whatever adapter is currently applied."""

    def __init__(self, ace_dir: str, ckpt_dir: str, dtype: str,
                 device: str, report: Report):
        self.ace_dir, self.ckpt_dir = ace_dir, ckpt_dir
        self.dtype, self.device = dtype, device
        self.report = report
        self.handler = None
        self.generate_fn = None
        self.generate_name = None
        self.adapter_active = False

    def load(self):
        sys.path.insert(0, self.ace_dir)
        from acestep.handler import AceStepHandler

        self.handler = AceStepHandler()
        self.report.set(handler_methods=[n for n in dir(self.handler)
                                         if not n.startswith("_")][:50])

        for name, fn in find_callable(self.handler, "load", "init", "setup", "prepare"):
            try:
                kwargs = {}
                for p in inspect.signature(fn).parameters.values():
                    low = p.name.lower()
                    if "checkpoint" in low or "model_dir" in low or low in ("path", "dir"):
                        kwargs[p.name] = self.ckpt_dir
                    elif "variant" in low or low == "model":
                        kwargs[p.name] = "xl_turbo"
                    elif "device" in low:
                        kwargs[p.name] = self.device
                    elif "dtype" in low or "precision" in low:
                        kwargs[p.name] = self.dtype
                fn(**kwargs)
                self.report.set(loaded_via=name, load_kwargs=kwargs)
                return
            except Exception as exc:  # noqa: BLE001
                self.report.set(**{f"loader_{name}_failed":
                                   f"{type(exc).__name__}: {exc}"[:300]})
        raise SystemExit("no loader on AceStepHandler accepted our arguments")

    def apply_adapter(self, adapter_path: str, strength: float) -> int:
        """Attach the LoKr adapter and return how many modules actually matched.

        Zero matched modules is a hard failure. The adapter would otherwise
        load silently, generate audio indistinguishable from base, and be
        reported as "adapter made no difference" -- a conclusion about our
        training that would in fact be a bug in our loading.
        """
        file_keys, sample = adapter_key_count(Path(adapter_path))
        self.report.set(adapter_path=adapter_path,
                        adapter_tensor_keys=file_keys,
                        adapter_key_sample=sample)
        if file_keys == 0:
            raise SystemExit(
                f"FATAL: {adapter_path} contains no LoKr/LoRA tensors. "
                "Nothing to apply.")

        loaders = find_callable(self.handler, "lora", "adapter", "lokr", "lycoris")
        self.report.set(adapter_loader_candidates=[n for n, _ in loaders])
        applied_via = None
        for name, fn in loaders:
            if "unload" in name.lower() or "remove" in name.lower():
                continue
            try:
                kwargs = {}
                for p in inspect.signature(fn).parameters.values():
                    low = p.name.lower()
                    if "path" in low or "dir" in low or "lora" in low or "adapter" in low:
                        kwargs[p.name] = adapter_path
                    elif "scale" in low or "strength" in low or "weight" in low or "alpha" in low:
                        kwargs[p.name] = strength
                fn(**kwargs)
                applied_via = name
                self.report.set(adapter_applied_via=name, adapter_kwargs=kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                self.report.set(**{f"adapter_{name}_failed":
                                   f"{type(exc).__name__}: {exc}"[:300]})

        matched = self._count_live_modules()
        self.adapter_active = matched > 0
        self.report.set(adapter_modules_matched=matched,
                        adapter_strength=strength,
                        adapter_applied=bool(applied_via))
        if matched == 0:
            raise SystemExit(
                "FATAL: adapter loaded but 0 modules matched the model. "
                "Generation would be identical to base and would be "
                "misreported as 'the adapter did not help'.")
        return matched

    def detach_adapter(self) -> None:
        """Remove the adapter and PROVE it is gone before returning.

        A "base" render with the adapter still quietly attached would make the
        base-vs-V1 comparison meaningless while looking entirely healthy, so
        this verifies by counting modules rather than trusting that an unload
        call did what its name suggests. If nothing can detach it, the handler
        is rebuilt from scratch -- slow, and still far cheaper than publishing
        a comparison that silently compared the adapter against itself.
        """
        if not self.adapter_active:
            return

        for name, fn in find_callable(self.handler, "unload", "remove",
                                      "disable", "unfuse", "unmerge"):
            if not any(h in name.lower()
                       for h in ("lora", "adapter", "lokr", "lycoris")):
                continue
            try:
                fn()
                self.report.set(adapter_detached_via=name)
                break
            except Exception as exc:  # noqa: BLE001
                self.report.set(**{f"detach_{name}_failed":
                                   f"{type(exc).__name__}: {exc}"[:200]})

        if self._count_live_modules() > 0:
            self.report.set(adapter_detach="unload failed; reloading foundation")
            self.handler = None
            self.generate_fn = self.generate_name = None
            self.load()
            if self._count_live_modules() > 0:
                raise SystemExit(
                    "FATAL: adapter could not be detached even by reloading. "
                    "A base render here would silently include the adapter.")
        self.adapter_active = False
        self.report.set(adapter_detached=True)

    def _count_live_modules(self) -> int:
        """Count adapter-bearing modules on whatever model object we can find."""
        import torch.nn as nn

        seen = 0
        for attr in dir(self.handler):
            if attr.startswith("_"):
                continue
            try:
                obj = getattr(self.handler, attr)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, nn.Module):
                for mod_name, _ in obj.named_modules():
                    if any(m in mod_name.lower()
                           for m in ("lokr", "lycoris", "lora")):
                        seen += 1
                if seen:
                    break
        return seen

    def resolve_generator(self):
        gens = find_callable(self.handler, "generat", "text2music", "infer",
                             "run", "sample", "predict")
        self.report.set(generate_candidates=[n for n, _ in gens])
        return gens

    def generate(self, caption: str, out_wav: Path, seed: int, duration: float,
                 steps: int, guidance: float) -> dict:
        """Render one take. Every knob is explicit so runs are comparable."""
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "caption": caption, "prompt": caption, "text": caption,
            "lyrics": "[Instrumental]",
            "duration": duration, "audio_duration": duration,
            "seed": seed, "manual_seeds": str(seed),
            "infer_step": steps, "steps": steps, "num_inference_steps": steps,
            "guidance_scale": guidance, "cfg": guidance,
            "language": "en", "timesignature": "4/4",
            "save_path": str(out_wav), "output_path": str(out_wav),
            "output_dir": str(out_wav.parent),
        }

        candidates = [(self.generate_name, self.generate_fn)] if self.generate_fn \
            else self.resolve_generator()

        before = {p: p.stat().st_mtime for p in out_wav.parent.rglob("*.wav")}
        for name, fn in candidates:
            if fn is None:
                continue
            try:
                sig = inspect.signature(fn)
                kwargs = {k: v for k, v in payload.items() if k in sig.parameters}
                t0 = time.time()
                result = fn(**kwargs) if kwargs else fn(caption)
                elapsed = round(time.time() - t0, 1)

                fresh = [p for p in out_wav.parent.rglob("*.wav")
                         if p not in before or p.stat().st_mtime > before[p]]
                if not fresh and out_wav.exists():
                    fresh = [out_wav]
                if fresh:
                    produced = max(fresh, key=lambda p: p.stat().st_size)
                    if produced != out_wav:
                        produced.replace(out_wav)
                    self.generate_fn, self.generate_name = fn, name
                    return {"ok": True, "method": name, "seconds": elapsed,
                            "wav": str(out_wav), "bytes": out_wav.stat().st_size,
                            "result_type": type(result).__name__}
                self.report.set(**{f"{name}_no_audio": str(result)[:200]})
            except Exception as exc:  # noqa: BLE001
                self.report.set(**{f"gen_{name}_failed":
                                   f"{type(exc).__name__}: {exc}"[:400]})
                traceback.print_exc()
        return {"ok": False}


def describe_wav(path: Path) -> dict:
    """Cheap objective facts about a render, so 'it made a file' is not the bar.

    Silence and DC offset are the two ways a generation can 'succeed' while
    producing nothing anyone would call audio.
    """
    try:
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        peak = float(np.abs(mono).max()) if mono.size else 0.0
        rms = float(np.sqrt((mono ** 2).mean())) if mono.size else 0.0
        return {
            "seconds": round(mono.shape[0] / sr, 2) if sr else 0,
            "sample_rate": sr,
            "peak": round(peak, 4),
            "rms": round(rms, 5),
            "dc_offset": round(float(mono.mean()), 5) if mono.size else 0.0,
            "silent": bool(peak < 1e-3),
        }
    except Exception as exc:  # noqa: BLE001
        return {"describe_failed": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- the stages

def stage_base(model: Foundation, out: Path, args, report: Report) -> bool:
    """Foundation alone. If this fails, nothing downstream is worth debugging."""
    wav = out / "base_gate.wav"
    own = Report(out / "base_gate.json")
    if wav.exists() and not args.force:
        own.set(SKIPPED="already present", wav=str(wav), **describe_wav(wav))
        report.set(BASE_INFERENCE="PASSED (cached)")
        return True

    own.set(stage="base", prompt=args.caption, seed=args.seed,
            duration=args.duration, steps=args.steps, guidance=args.guidance,
            dtype=model.dtype, adapter="NONE -- foundation only")
    result = model.generate(args.caption, wav, args.seed, args.duration,
                            args.steps, args.guidance)
    if not result.get("ok"):
        own.set(BASE_INFERENCE="FAILED")
        report.set(BASE_INFERENCE="FAILED")
        return False
    facts = describe_wav(wav)
    own.set(BASE_INFERENCE="PASSED", **result, **facts)
    report.set(BASE_INFERENCE="PASSED", base_wav=str(wav),
               base_audio=facts)
    if facts.get("silent"):
        own.set(WARNING="file written but audio is silent")
        report.set(BASE_INFERENCE="FAILED (silent output)")
        return False
    return True


def stage_adapter(model: Foundation, out: Path, args, report: Report) -> bool:
    """The same prompt, same seed, through our V1 LoKr."""
    wav = out / "v1_gate.wav"
    own = Report(out / "v1_gate.json")
    if not args.adapter:
        own.set(SKIPPED="no --adapter given")
        report.set(V1_INFERENCE="SKIPPED (no adapter path)")
        return False

    matched = model.apply_adapter(args.adapter, args.strength)
    own.set(stage="v1", prompt=args.caption, seed=args.seed,
            duration=args.duration, steps=args.steps, guidance=args.guidance,
            dtype=model.dtype, adapter=args.adapter,
            adapter_modules_matched=matched, adapter_strength=args.strength)
    report.set(ADAPTER_LOADED=f"modules={matched}")

    result = model.generate(args.caption, wav, args.seed, args.duration,
                            args.steps, args.guidance)
    if not result.get("ok"):
        own.set(V1_INFERENCE="FAILED")
        report.set(V1_INFERENCE="FAILED")
        return False
    facts = describe_wav(wav)
    own.set(V1_INFERENCE="PASSED", **result, **facts)
    report.set(V1_INFERENCE="PASSED", v1_wav=str(wav), v1_audio=facts)
    return True


def stage_matrix(model: Foundation, out: Path, args, report: Report) -> bool:
    """Four very different prompts, base vs V1, identical settings throughout.

    Rendered in two passes -- every base take first, then the adapter attached
    once and every V1 take -- rather than toggling the adapter per prompt. One
    attach and one detach is far less to get wrong than eight, and a stray
    adapter left applied during a base render would corrupt the comparison
    while looking perfectly healthy.

    Identical settings are what make this a comparison rather than an anecdote:
    the only thing differing between two renders of a prompt is the adapter.
    """
    matrix_dir = out / "matrix"
    own = Report(matrix_dir / "matrix.json")
    results: dict[str, dict] = {}

    model.detach_adapter()
    for key, prompt in PROMPT_MATRIX:
        wav = matrix_dir / f"{key}__base.wav"
        if not wav.exists() or args.force:
            r = model.generate(prompt, wav, args.seed, args.duration,
                               args.steps, args.guidance)
            if not r.get("ok"):
                own.set(**{f"{key}_base": "FAILED"})
                continue
        results[key] = {"prompt_key": key, "prompt": prompt,
                        "base": describe_wav(wav)}
        own.set(**{f"{key}_base": "rendered"})

    if args.adapter:
        matched = model.apply_adapter(args.adapter, args.strength)
        own.set(adapter_modules_matched=matched)
        for key, prompt in PROMPT_MATRIX:
            if key not in results:
                continue
            wav = matrix_dir / f"{key}__v1.wav"
            if not wav.exists() or args.force:
                r = model.generate(prompt, wav, args.seed, args.duration,
                                   args.steps, args.guidance)
                if not r.get("ok"):
                    own.set(**{f"{key}_v1": "FAILED"})
                    continue
            results[key]["v1"] = describe_wav(wav)
            own.set(**{f"{key}_v1": "rendered"})

    rows = list(results.values())
    own.set(rows=rows, settings={"seed": args.seed, "duration": args.duration,
                                 "steps": args.steps, "guidance": args.guidance,
                                 "strength": args.strength},
            note="Numbers here only rule out broken or silent takes. Whether "
                 "V1 sounds better than base is a listening decision.")
    report.set(MATRIX=f"{len(rows)}/{len(PROMPT_MATRIX)} prompts rendered",
               matrix_dir=str(matrix_dir))
    return len(rows) == len(PROMPT_MATRIX)


def stage_strength(model: Foundation, out: Path, args, report: Report) -> bool:
    """Sweep LoKr scale to find where adaptation helps before it narrows the model.

    A 0.04% adapter trained on one corpus can pull the foundation toward that
    corpus hard enough to lose the breadth that made it useful. The sweep is
    how that gets measured instead of assumed.
    """
    sweep_dir = out / "strength"
    own = Report(sweep_dir / "strength.json")
    if not args.adapter:
        own.set(SKIPPED="no --adapter given")
        return False

    prompt = PROMPT_MATRIX[0][1]
    rows = []
    for strength in ADAPTER_STRENGTHS:
        wav = sweep_dir / f"strength_{strength:.1f}.wav"
        if not wav.exists() or args.force:
            matched = model.apply_adapter(args.adapter, strength)
            r = model.generate(prompt, wav, args.seed, args.duration,
                               args.steps, args.guidance)
            if not r.get("ok"):
                own.set(**{f"strength_{strength}": "FAILED"})
                continue
        rows.append({"strength": strength, **describe_wav(wav)})
        own.set(**{f"strength_{strength}": "rendered"})

    own.set(rows=rows, prompt=prompt,
            note="Objective facts only. Which strength is best is a listening "
                 "decision; these numbers only rule out broken takes.")
    report.set(STRENGTH_SWEEP=f"{len(rows)}/{len(ADAPTER_STRENGTHS)} rendered",
               strength_dir=str(sweep_dir))
    return bool(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all",
                    choices=["all", "base", "adapter", "matrix", "strength"])
    ap.add_argument("--ace-dir", default=os.environ.get("ACESTEP_DIR", str(Path.home() / "ACE-Step-1.5")))
    ap.add_argument("--checkpoint-dir", default=os.environ.get("ACESTEP_CHECKPOINTS", str(Path.home() / "checkpoints")))
    ap.add_argument("--adapter", default=os.environ.get("SONGFORGE_LORA", ""))
    ap.add_argument("--out", default=str(Path.home() / "gate_out"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None,
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="LoKr scale for the base/adapter/matrix stages")
    ap.add_argument("--force", action="store_true",
                    help="re-render even where output already exists")
    ap.add_argument("--caption", default=PROMPT_MATRIX[0][1])
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report = Report(out / "run_report.json")

    import torch

    capability = (list(torch.cuda.get_device_capability(0))
                  if torch.cuda.is_available() else None)
    if args.dtype:
        dtype = args.dtype
    elif capability and capability[0] >= 8:
        dtype = "bfloat16"
    elif capability:
        dtype = "float16"
    else:
        dtype = "float32"

    report.set(torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               compute_capability=capability, dtype=dtype,
               ace_dir=args.ace_dir, checkpoint_dir=args.checkpoint_dir,
               adapter=args.adapter or None, out=str(out),
               settings={"seed": args.seed, "duration": args.duration,
                         "steps": args.steps, "guidance": args.guidance})

    model = Foundation(args.ace_dir, args.checkpoint_dir, dtype, args.device, report)
    model.load()

    stages = ["base", "adapter", "matrix", "strength"] if args.stage == "all" \
        else [args.stage]

    ok = True
    for stage in stages:
        print(f"\n{'=' * 66}\nSTAGE: {stage}\n{'=' * 66}", flush=True)
        if stage == "base":
            ok = stage_base(model, out, args, report)
            # Everything after this assumes the foundation works. Continuing
            # past a base failure only produces confusing adapter errors.
            if not ok:
                report.set(HALTED="base inference failed; later stages skipped")
                break
        elif stage == "adapter":
            ok = stage_adapter(model, out, args, report)
            if not ok and args.stage == "all":
                report.set(HALTED="adapter stage failed; later stages skipped")
                break
        elif stage == "matrix":
            stage_matrix(model, out, args, report)
        elif stage == "strength":
            stage_strength(model, out, args, report)

    print(f"\n{'=' * 66}\nSUMMARY\n{'=' * 66}")
    for key in ("BASE_INFERENCE", "ADAPTER_LOADED", "V1_INFERENCE",
                "MATRIX", "STRENGTH_SWEEP", "HALTED"):
        if key in report.data:
            print(f"  {key:18s} {report.data[key]}")
    print(f"  report            {report.path}")
    return 0 if report.data.get("BASE_INFERENCE", "").startswith("PASSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
