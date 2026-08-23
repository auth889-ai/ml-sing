"""Generate the identical benchmark prompt set with one pretrained foundation.

Writes exactly the agreed layout so candidates are directly comparable:

    foundation_benchmarks/<model>/
        piano.wav violin.wav guitar.wav rock.wav
        edm.wav cinematic.wav vocal.wav rich_mix.wav
        metadata.json

The decision is made on the audio, not on README claims, so this script's job is
to produce real audio under recorded conditions and to be honest in the metadata
about which requested controls the model actually honoured.

    python scripts/foundation_benchmark.py --adapter acestep \
        --output-root foundation_benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from songforge.generation import available, build, load_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one pretrained music foundation.")
    parser.add_argument("--adapter", required=True, help=f"one of: {', '.join(available()) or 'none registered'}")
    parser.add_argument("--prompts", default=str(ROOT / "benchmarks" / "prompts.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "foundation_benchmarks"))
    parser.add_argument("--only", nargs="+", default=None, help="Run a subset of prompt ids.")
    parser.add_argument("--duration", type=float, default=None, help="Override duration for every prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Report capabilities and licence, generate nothing.")
    args, extra = parser.parse_known_args()

    options = _parse_extra(extra)
    adapter = build(args.adapter, **options)
    capabilities = adapter.capabilities
    license_position = adapter.license

    print(f"=== {capabilities.model} {capabilities.version} ===")
    print(f"sings: {capabilities.produces_vocals} | max {capabilities.max_duration_seconds:.0f}s "
          f"| {capabilities.sample_rate} Hz {capabilities.channels}ch")
    print(f"native controls: {', '.join(capabilities.native_controls()) or 'none'}")
    print(f"code licence   : {license_position.code_license}")
    print(f"weights licence: {license_position.weights_license}")
    print(f"commercial use : {license_position.commercial_use}")
    print(f"product foundation: {license_position.usable_as_product_foundation} | "
          f"finetune: {license_position.usable_for_finetuning} | "
          f"baseline: {license_position.usable_as_research_baseline}")

    requests = load_prompts(args.prompts)
    if args.only:
        requests = [r for r in requests if r.extra["id"] in set(args.only)]
        if not requests:
            raise SystemExit(f"no prompts matched {args.only}")

    if args.dry_run:
        print("\ndry run: control resolution per prompt")
        from songforge.generation import resolve_controls

        for request in requests:
            resolution = resolve_controls(request, capabilities)
            flag = "ok" if resolution.honest else f"{len(resolution.warnings)} unsupported"
            print(f"  {request.extra['id']:<10} {flag}")
            for warning in resolution.warnings:
                print(f"      - {warning}")
        return

    out_dir = Path(args.output_root) / capabilities.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nloading weights...")
    adapter.load()

    entries: list[dict[str, Any]] = []
    for index, request in enumerate(requests, start=1):
        prompt_id = request.extra["id"]
        if args.duration is not None:
            request = _with_duration(request, args.duration)
        target = out_dir / f"{prompt_id}.wav"
        print(f"[{index}/{len(requests)}] {prompt_id:<10} {request.duration_seconds:.0f}s ... ", end="", flush=True)

        result = adapter.generate(request, target)
        if result.error:
            print(f"FAILED  {result.error}")
        else:
            print(f"{result.generation_seconds:.1f}s  RTF {result.real_time_factor:.2f}  "
                  f"VRAM {result.peak_vram_gb:.2f} GB")
        for warning in result.resolution.warnings:
            print(f"      unsupported: {warning}")

        entry = result.to_dict()
        entry["prompt_id"] = prompt_id
        entry["title"] = request.extra.get("title")
        entry["expects_instruments"] = request.extra.get("expects_instruments", [])
        entries.append(entry)

    metadata = {
        "model": capabilities.model,
        "version": capabilities.version,
        "capabilities": capabilities.to_dict(),
        "license": license_position.to_dict(),
        "prompt_set": str(Path(args.prompts).name),
        "results": entries,
        "generated": sum(1 for e in entries if not e["error"]),
        "failed": sum(1 for e in entries if e["error"]),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n{metadata['generated']}/{len(entries)} generated -> {out_dir}")
    print(f"metadata: {metadata_path}")
    if metadata["failed"]:
        print(f"{metadata['failed']} prompt(s) failed; see metadata.json for the errors")


def _with_duration(request, seconds: float):
    from dataclasses import replace

    return replace(request, duration_seconds=seconds)


def _parse_extra(extra: list[str]) -> dict[str, Any]:
    """Pass adapter-specific options through as --key value pairs."""
    options: dict[str, Any] = {}
    key: str | None = None
    for token in extra:
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            options[key] = True
        elif key is not None:
            options[key] = _coerce(token)
            key = None
    return options


def _coerce(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


if __name__ == "__main__":
    main()
