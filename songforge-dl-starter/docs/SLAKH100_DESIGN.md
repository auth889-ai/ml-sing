# Slakh-100 — sizing before download

Research date 2026-08-18. Machine-readable spec: `configs/datasets/slakh100.yaml`.
Nothing here has been downloaded; per project rule, sizes are recorded while the
storage cost is still a decision.

## Slakh2100, authoritative facts

| fact | value | source |
| --- | --- | --- |
| licence | **CC-BY-4.0** — redistribution, derivatives, commercial use allowed with attribution | slakh.com, Zenodo 4599666 & 7708270 |
| original tracks | 2100 (1500/375/225 train/val/test) | slakh-utils, paper (arXiv:1909.08494) |
| redux tracks | **1710** (1289/270/151) — duplicate MIDI removed to `omitted/` | slakh-utils, Zenodo 4599666 |
| mixture audio | 145 h across 2100 tracks → avg track ≈ 4.14 min; redux ≈ 118 h (derived) | paper abstract |
| stems | 4–48 per mix, ~10.5 average, ~21.5–22.5k total (±5%, read from official charts) | slakh.com |
| format | mono 44.1 kHz 16-bit FLAC; `mix.flac` + `stems/SXX.flac` + `MIDI/` + `metadata.yaml` | slakh-utils |
| canonical download | Zenodo 4599666, **one 104.3 GB tar.gz** (byte count verified by HTTP HEAD); ~500 GB as WAV | Zenodo |
| 16 kHz variant | **Zenodo 7708270**: `slakh2100_redux_16k.tar.gz`, **48.7 GB**, CC-BY-4.0, full redux resampled to 16 kHz | Zenodo |
| synthesis | rendered from Lakh MIDI with NI Kontakt (Komplete 12, 187 patches / 34 classes) via RenderMan; per-stem −13 LUFS, no live recordings | slakh_generation_scripts |
| guarantee | every track contains piano, bass, guitar, drums with ≥50 notes each | slakh.com |

Instrument availability across the 2100 mixtures (official charts, ±30):
piano/bass/guitar/drums 100%; strings ~84%; synth pad ~34%; reed ~31%;
brass ~31%; organ ~28%; pipe ~26%; synth lead ~22%; chromatic percussion ~19%.

**Per-track download does not exist.** Zenodo serves monolithic gzip (Range
requests work — verified 206 — but gzip is not seekable). HF mirrors are
mixture-only, test-only, or blind chunks; one relicenses NC-SA in conflict with
upstream CC-BY and is not relied on.

## Why the 16 kHz variant wins

The whole pipeline (BabySlakh M02/M04 builds) runs at 16 kHz. Zenodo 7708270 is
the same redux corpus already at that rate: less than half the download, and no
resampling pass. One check before committing: `tar -tzf | head` must confirm
stems + `metadata.yaml` survived the resample (inferred from the 48.7 GB size,
not yet listed — 145 h of mixtures alone would be ~10 GB).

## Slakh-100 composition

- **100 tracks = 80 train / 10 val / 10 test**, selected only from the
  corresponding official redux split directories; `omitted/` is never touched.
- Eligibility: 3–6 min duration, ≥6 stems, every stem `audio_rendered: true`,
  quota-family stems louder than −30 LUFS integrated.
- Balance: piano/guitar/bass/drums are free (dataset guarantee). Rare families
  filled by greedy weighted max-coverage with the train quotas in
  `configs/datasets/slakh100.yaml` (strings ≥40 of 80, brass/reed/pad ≥25, …).
- No genre or instrument hardcoding: selection reads only corpus
  `metadata.yaml` `inst_class` values, and the labels condition data, not code.

| quantity | estimate |
| --- | ---: |
| mixture audio | ~6.9 h |
| stem audio (silences included) | ~72 h |
| kept raw 16 kHz FLAC | ~3 GB (out of the one-time 48.7 GB download) |
| decoded 16 kHz WAV | ~9 GB |
| ACE-Step processed output (scaled from BabySlakh: ×5) | ~49 h, ~88k segments, ~6–7 GB |
| **peak Drive requirement** | **~15 GB** |
| peak Colab local disk during acquisition | ~55 GB |

## Acquisition plan (when the download is approved)

1. Colab: `wget -c` the 48.7 GB archive to local runtime disk (resumable —
   matches the checkpoint-resumable rule); verify MD5 `66a2301e…`.
2. Extract only `*/metadata.yaml` (~1710 tiny files, minutes).
3. Run selection → 100 `TrackXXXXX` ids.
4. Selectively extract those 100 directories, copy ~3 GB to Drive, delete the
   archive.

Fallbacks, in order: (1) streamed `curl | tar -xz --wildcards` of the 44.1 kHz
redux archive if native rate proves necessary (~104 GB through the pipe, ~5 GB
stored, not resumable); (2) mixture-only subset from the per-track HF mirror,
only if stems are unneeded and its NC-SA relicensing risk is accepted;
(3) BabySlakh-only — already processed, zero new download.

## Flagged as not verified

- internal structure of the 16 kHz archive (stems/metadata presence);
- per-family counts are chart readings, ±30 mixtures;
- redux mixture-hours and extracted-FLAC size are derivations;
- slakh.com was read despite a broken TLS cert (GitHub Pages misconfig);
  every overlapping fact cross-checked against Zenodo and slakh-utils.
