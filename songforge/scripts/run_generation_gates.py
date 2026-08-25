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
    """One loaded ACE-Step handler, driven through its documented public API.

    An earlier version discovered the API by introspection -- listing methods
    whose names looked like "load" or "generate" and guessing keyword names.
    It reached the model and then failed four different ways at once, because
    it was calling internals: service_generate() wants `captions` (plural) and
    returns latents rather than audio, and infer_text_embeddings() needs a
    text_encoder that only initialize_service() creates.

    ACE-Step ships a supported surface -- acestep.inference.generate_music with
    GenerationParams/GenerationConfig, which is what its own cli.py uses -- so
    this now calls that. Guessing at internals was never going to be stable
    across versions anyway.
    """

    def __init__(self, ace_dir: str, ckpt_dir: str, dtype: str,
                 device: str, report: Report):
        self.ace_dir, self.ckpt_dir = ace_dir, ckpt_dir
        self.dtype, self.device = dtype, device
        self.report = report
        self.dit = None
        self.llm = None
        self.adapter_active = False

    def load(self):
        sys.path.insert(0, self.ace_dir)
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        self.dit = AceStepHandler()
        self.llm = LLMHandler()

        models = self.dit.get_available_acestep_v15_models() or []
        self.report.set(available_models=models)
        if not models:
            raise SystemExit(
                f"no ACE-Step v1.5 models found under {self.ckpt_dir}. "
                "The weight download did not complete.")

        # Prefer the turbo variant the project trained against; fall back to
        # whatever is present rather than failing on an exact name.
        config_path = next((m for m in models if "turbo" in m.lower()), models[0])

        # flash-attn is deliberately not installed (no wheel for this pair, and
        # compiling costs about an hour), so ask the handler rather than
        # assuming -- it falls back to PyTorch SDPA on its own.
        use_flash = False
        try:
            use_flash = bool(self.dit.is_flash_attention_available(self.device))
        except Exception:  # noqa: BLE001
            pass

        self.report.set(config_path=config_path, use_flash_attention=use_flash)
        self.dit.initialize_service(
            project_root=self.ace_dir,
            config_path=config_path,
            device=self.device,
            use_flash_attention=use_flash,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )
        self.report.set(handler_initialised=True)

    # ------------------------------------------------------------- adapters
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
                f"FATAL: {adapter_path} contains no LoKr/LoRA tensors.")

        # load_lora resolves LyCORIS/LoKr layouts itself and reports with a
        # leading tick or cross rather than raising.
        message = self.dit.load_lora(adapter_path)
        self.report.set(adapter_load_message=message)
        if not str(message).startswith("\u2705"):
            raise SystemExit(f"FATAL: adapter did not load: {message}")

        try:
            self.report.set(scale_message=self.dit.set_lora_scale(strength))
        except Exception as exc:  # noqa: BLE001
            self.report.set(scale_unsupported=f"{type(exc).__name__}: {exc}"[:200])

        matched = self._count_live_modules()
        self.adapter_active = matched > 0
        self.report.set(adapter_modules_matched=matched,
                        adapter_strength=strength,
                        lora_loaded_flag=bool(getattr(self.dit, "lora_loaded", False)))
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
        this verifies by counting modules rather than trusting the unload call.
        """
        if not self.adapter_active:
            return
        self.report.set(adapter_unload_message=self.dit.unload_lora())
        remaining = self._count_live_modules()
        if remaining > 0:
            raise SystemExit(
                f"FATAL: {remaining} adapter modules still attached after "
                "unload_lora. A base render here would silently include the "
                "adapter and corrupt the comparison.")
        self.adapter_active = False
        self.report.set(adapter_detached=True)

    def _count_live_modules(self) -> int:
        model = getattr(self.dit, "model", None)
        if model is None:
            return 0
        return sum(1 for name, _ in model.named_modules()
                   if any(m in name.lower() for m in ("lokr", "lycoris", "lora")))

    # ----------------------------------------------------------- generation
    def generate(self, caption: str, out_wav: Path, seed: int, duration: float,
                 steps: int, guidance: float) -> dict:
        """Render one take. Every knob is explicit so runs are comparable."""
        from acestep.inference import (GenerationConfig, GenerationParams,
                                       generate_music)

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        workdir = out_wav.parent / f"_{out_wav.stem}"
        workdir.mkdir(parents=True, exist_ok=True)

        params = GenerationParams(
            caption=caption,
            lyrics="[Instrumental]",
            instrumental=True,
            duration=duration,
            inference_steps=steps,
            guidance_scale=guidance,
            seed=seed,
            task_type="text2music",
        )
        # use_random_seed=False with an explicit seed list is what makes the
        # base and V1 renders of a prompt actually comparable.
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[seed],
            audio_format="wav",
        )

        t0 = time.time()
        result = generate_music(self.dit, self.llm, params, config,
                                save_dir=str(workdir))
        elapsed = round(time.time() - t0, 1)

        if not getattr(result, "success", False):
            self.report.set(generate_error=str(getattr(result, "error", ""))[:400],
                            generate_status=str(getattr(result, "status_message", ""))[:300])
            return {"ok": False, "seconds": elapsed}

        produced = [Path(a["path"]) for a in (result.audios or [])
                    if isinstance(a, dict) and a.get("path")
                    and Path(a["path"]).exists()]
        if not produced:
            produced = sorted(workdir.rglob("*.wav")) + sorted(workdir.rglob("*.flac"))
        if not produced:
            self.report.set(generate_no_files=str(getattr(result, "status_message", ""))[:300])
            return {"ok": False, "seconds": elapsed}

        biggest = max(produced, key=lambda p: p.stat().st_size)
        biggest.replace(out_wav)
        return {"ok": True, "seconds": elapsed, "wav": str(out_wav),
                "bytes": out_wav.stat().st_size,
                "status": str(getattr(result, "status_message", ""))[:200]}


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
