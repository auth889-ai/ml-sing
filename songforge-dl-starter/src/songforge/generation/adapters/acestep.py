"""ACE-Step 1.5 via the diffusers `AceStepPipeline`.

Capabilities below are taken from the actual `__call__` signature, not from
marketing copy. The distinction that matters for SongForge: ACE-Step has genuine
typed inputs for **duration, BPM, keyscale, time signature, vocal language and
seed**, while **genre, mood, instruments and vocal character are only free text**
inside the caption. There is no genre enum and no instrument list parameter, so
this adapter declares those PROMPT, never NATIVE. Chord and melody conditioning
do not exist at all.

Note on the diffusers path: it wraps the DiT half of the stack and does not run
the LM planner. That sidesteps the known unseeded-planner reproducibility bug, at
the cost of the planner's rhythm stabilisation on the non-turbo checkpoints.

Licence: MIT for code and weights (ACE-Step 1.5; the legacy v1 was Apache-2.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..adapter import FoundationAdapter, LicensePosition
from ..capabilities import Capabilities, ControlSupport
from ..registry import register
from ..request import SongRequest

#: Checkpoints exposed through the diffusers port.
CHECKPOINTS = {
    "xl-turbo": "ACE-Step/acestep-v15-xl-turbo-diffusers",
    "xl-sft": "ACE-Step/acestep-v15-xl-sft-diffusers",
}
DEFAULT_CHECKPOINT = "xl-turbo"

#: Languages the model card lists explicitly. It claims 50+; these are the ones
#: named, so these are the ones we will assert support for.
NAMED_LANGUAGES = ("en", "zh", "ja", "ko", "fr", "de", "es", "it", "pt", "ru")


class AceStepAdapter(FoundationAdapter):
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        dtype: str = "bfloat16",
        steps: int | None = None,
        guidance_scale: float | None = None,
        shift: float = 3.0,
        cpu_offload: bool = False,
    ) -> None:
        self.checkpoint_key = checkpoint
        self.repo_id = CHECKPOINTS.get(checkpoint, checkpoint)
        self.device = device
        self.dtype_name = dtype
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.shift = shift
        self.cpu_offload = cpu_offload
        self._pipe: Any = None

    # --- declaration -----------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            model="ace_step_15",
            version=self.checkpoint_key,
            controls={
                # typed arguments of AceStepPipeline.__call__
                "prompt": ControlSupport.NATIVE,
                "lyrics": ControlSupport.NATIVE,
                "duration": ControlSupport.NATIVE,
                "bpm": ControlSupport.NATIVE,
                "key": ControlSupport.NATIVE,
                "seed": ControlSupport.NATIVE,
                # free text inside the caption; the model has no typed field
                "genre": ControlSupport.PROMPT,
                "mood": ControlSupport.PROMPT,
                "instruments": ControlSupport.PROMPT,
                "vocal_style": ControlSupport.PROMPT,
                # bracket tags embedded in the lyrics string, not a field.
                # The issue tracker documents sections being dropped outright,
                # so calling this NATIVE would overstate it.
                "structure": ControlSupport.PROMPT,
            },
            max_duration_seconds=600.0,
            sample_rate=48000,
            channels=2,
            produces_vocals=True,
            languages=NAMED_LANGUAGES,
            notes=(
                "Typed: duration, bpm, keyscale, timesignature, vocal_language, seed. "
                "Text only: genre, mood, instruments, vocal character. "
                "No chord or melody conditioning exists. "
                "Requires bf16 (Ampere+); lyrics-to-song fails with NaN latents on Turing/T4."
            ),
        )

    @property
    def license(self) -> LicensePosition:
        return LicensePosition(
            code_license="MIT",
            weights_license="MIT",
            commercial_use="allowed",
            redistribution="unrestricted (MIT)",
            attribution="MIT notice for the software; none required for generated audio",
            training_data_notes=(
                "27M-sample corpus. No dataset list and no provenance audit published. "
                "The 'commercially safe output' claim is an unaudited vendor assertion; "
                "a request for written output-rights terms went unanswered."
            ),
            usable_as_research_baseline=True,
            usable_for_finetuning=True,
            usable_as_product_foundation=True,
            sources=(
                "https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE",
                "https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers",
            ),
        )

    # --- execution -------------------------------------------------------

    def load(self) -> None:
        import torch
        from diffusers import AceStepPipeline

        dtype = getattr(torch, self.dtype_name)
        if self.device == "cuda" and dtype is torch.bfloat16 and torch.cuda.is_available():
            major = torch.cuda.get_device_capability(0)[0]
            if major < 8:
                raise RuntimeError(
                    f"{torch.cuda.get_device_name(0)} is compute capability {major}.x and has no bfloat16. "
                    "ACE-Step produces NaN latents for any non-empty lyrics on Turing. "
                    "Use an L4 or A100, or pass dtype=float32 for instrumental-only debugging."
                )

        self._pipe = AceStepPipeline.from_pretrained(self.repo_id, torch_dtype=dtype)
        if self.cpu_offload:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(self.device)

    def _compose_prompt(self, request: SongRequest) -> str:
        """Fold the untyped controls into the caption.

        These are genuinely soft. The capability table already says so, and the
        benchmark metadata records it, so the reader is never told that a genre
        request was honoured as a constraint.
        """
        parts = [request.prompt.strip().rstrip(".")]
        if request.genre:
            parts.append(f"genre: {', '.join(request.genre)}")
        if request.mood:
            parts.append(f"mood: {', '.join(request.mood)}")
        if request.instruments:
            parts.append(f"instruments: {', '.join(request.instruments)}")
        if request.vocal and request.vocal.present:
            descriptors = request.vocal.descriptors()
            parts.append(f"{' '.join(descriptors)} vocal" if descriptors else "with vocals")
        elif request.vocal is not None and not request.vocal.present:
            parts.append("instrumental, no vocals")
        return ". ".join(parts)

    def _compose_lyrics(self, request: SongRequest) -> str:
        """Lyrics plus bracket structure tags, which live in the same string."""
        if request.lyrics:
            return request.lyrics.strip()
        if request.structure:
            # Structure without lyrics: tags alone still steer the arrangement.
            return "\n".join(f"[{section.kind.replace('_', ' ')}]" for section in request.structure)
        return ""

    def _generate(self, request: SongRequest, output_path: Path) -> dict[str, Any]:
        import soundfile as sf
        import torch

        if self._pipe is None:
            raise RuntimeError("load() must be called before generate()")

        generator = torch.Generator(device=self.device).manual_seed(request.seed)
        language = (request.vocal.language if request.vocal and request.vocal.language else "en")

        kwargs: dict[str, Any] = {
            "prompt": self._compose_prompt(request),
            "lyrics": self._compose_lyrics(request),
            "audio_duration": float(request.duration_seconds),
            "vocal_language": language,
            "shift": self.shift,
            "generator": generator,
        }
        if self.steps is not None:
            kwargs["num_inference_steps"] = self.steps
        if self.guidance_scale is not None:
            kwargs["guidance_scale"] = self.guidance_scale
        if request.bpm is not None:
            kwargs["bpm"] = int(request.bpm)
        if request.key is not None:
            kwargs["keyscale"] = request.key

        audio = self._pipe(**kwargs).audios
        # [batch, channels, samples] -> [samples, channels] for soundfile
        waveform = audio[0].T.cpu().float().numpy()
        sf.write(str(output_path), waveform, self._pipe.sample_rate)

        settings = {k: v for k, v in kwargs.items() if k != "generator"}
        settings.update({
            "checkpoint": self.repo_id,
            "dtype": self.dtype_name,
            "seed": request.seed,
            "cpu_offload": self.cpu_offload,
            "sample_rate": self._pipe.sample_rate,
        })
        return settings


@register("acestep")
def _build(**kwargs: Any) -> AceStepAdapter:
    return AceStepAdapter(**kwargs)
