# Permissive vocal & real-music data — deployable vs research-only

Research date 2026-08-18. Every licence below was verified on the official page
that date unless marked UNVERIFIED. Rule enforced by
`scripts/build_acestep_lora_dataset.py`: nothing outside the DEPLOYABLE list
enters the public adapter line; research-only corpora require
`--allow-nonpermissive` and produce adapters that are never merged.

## DEPLOYABLE (permissive), ranked by SongForge usefulness

| # | dataset | licence | size | languages | content | role |
| - | --- | --- | --- | --- | --- | --- |
| 1 | [VocalSet](https://zenodo.org/records/1193957) | CC-BY-4.0 ✅ | 10.1 h / 2.1 GB | vowels only | 20 pro singers, 17 techniques (vibrato, belt, breathy…) | **vocal-realism LoRA core** — teaches timbre/technique, not word articulation |
| 2 | [PJS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus) | CC-BY-SA-4.0 (official page quote UNVERIFIED) | 100 songs + 100 readings | Japanese | phoneme-labelled solo singing, MIDI + MusicXML | multilingual lyric-aligned singing; SA scope for weights needs one legal check |
| 3 | [vocadito](https://zenodo.org/records/5578807) | CC-BY-4.0 ✅ | 40 excerpts / 58.5 MB | **7 languages** | solo singing + lyrics + f0 + notes | permissive sung-lyrics **evaluation** set; too small to train alone |
| 4 | [Dagstuhl ChoirSet](https://zenodo.org/records/4608395) + [Choral Singing Dataset](https://zenodo.org/records/1319597) | CC-BY-4.0 ✅ both | 5.1 + 0.9 GB | Latin/Spanish/Catalan (CSD-UPF) | per-singer choir multitracks | choral/backing-vocal texture LoRA |
| 5 | [GuitarSet](https://zenodo.org/records/3371780) | CC-BY-4.0 ✅ | ~3 h / 8.2 GB | — | **real recorded** acoustic guitar, pitch/chord/beat annotated | acoustic-guitar-realism LoRA — the first real-recorded counterweight to synthesized Slakh |
| 6 | [E-GMD](https://magenta.tensorflow.org/datasets/e-gmd) (+ [GMD](https://magenta.tensorflow.org/datasets/groove)) | CC-BY-4.0 ✅ | 444 h / 90 GB (13.6 h / 4.8 GB) | — | human-performed drumming, 43 kits; audio is kit-module rendered | drum groove/feel LoRA; big — subset before download |
| 7 | [FMA](https://github.com/mdeff/fma) CC-BY/CC0 subset | per-track; filter `tracks.csv` (counts UNVERIFIED; ~14% of CC0 labels known-misfiled — re-check) | up to 879 GB before filtering | many | full songs incl. vocals | **arrangement-richness + vocals-with-lyrics at scale** (with in-project transcription); needs attribution manifest |
| 8 | [ccMixter](http://dig.ccmixter.org/) "Free for Commercial Use" | per-track CC ✅ (filter exists) | thousands of tracks/stems | many | remixes with separated vocal stems | vocals-in-mix, commercial-safe subset |
| 9 | [Musopen](https://musopen.org) PD/CC0 picks | per-track (UNVERIFIED per-track) | curated | — | real classical recordings | orchestral/classical realism pool |

Deployment obligation: an attribution page on the public site listing every
CC-BY corpus used by the shipped adapter (extends `docs/ATTRIBUTION.md`).

**The named gap:** no large permissive corpus of lyric-aligned popular-style
solo singing exists. The deployable vocal stack is timbre-rich (VocalSet) but
lyric-poor (vocadito/PJS are small). The scale route for lyric-carrying vocals
is FMA/ccMixter CC-BY vocal tracks + in-project Whisper transcription and
alignment — which we already run for lyric-recall evaluation.

## RESEARCH-ONLY / BLOCKED

| dataset | why |
| --- | --- |
| MAESTRO | CC-BY-NC-SA (verified) — best real piano (198.7 h), NC |
| CSD Children's Song | CC-BY-NC-SA (verified) — ideal Korean/English lyric alignment, NC |
| MUSDB18 / -HQ | "educational purposes only", NC constituents (verified) |
| MoisesDB | CC-BY-NC-SA (verified) |
| MedleyDB | non-commercial research terms (primary-page text UNVERIFIED) |
| MTG-Jamendo | "solely for non-commercial research"; commercial needs Jamendo written authorization (verified) |
| MusicNet | Zenodo says CC-BY but Gardner Museum source tracks are CC-BY-NC-ND — **conflicted**; usable only after per-track provenance filtering |
| URMP | **no licence published at all** (verified absent) — unusable publicly |
| DAMP / DAMP-VSEP | Smule research licence, no transfer/commercial use (verified) |
| GTSinger, OpenSinger, Opencpop, M4Singer, ACE-Opencpop | known CC-BY-NC family, already rejected |
| NUS-48E, Kiritan, JVS-MuSiC, MAPS, SMD, Bach10, ENST-Drums, Cambridge-MT, jaCappella | research agreements or unclear terms (mostly UNVERIFIED individually) |

No NC corpus is ever silently mixed into the deployable adapter. If an NC
corpus is ever used for a research-only comparison adapter, it is labelled as
such in its config (`licence_class: research-only`) and in every report.

## Gap-closure research round 2 (2026-08-18, verified on official pages)

New DEPLOYABLE finds:

| dataset | licence | size | role |
| --- | --- | --- | --- |
| [SingStyle111](https://zenodo.org/records/10265401) | CC-BY-4.0 ✅ | 12.8 h, 111 songs, 8 pro singers | **the lyric-aligned singing find**: pop/jazz/bel canto, EN/ZH/IT, lyrics + MIDI + phoneme alignment; per-song cover-composition check before public audio release |
| [Open Goldberg](https://opengoldbergvariations.org/) + [Open WTC](https://welltemperedclavier.org/) | CC0 ✅ | ~3.3 h studio piano | real-piano fidelity anchor |
| [Musopen Complete Chopin](https://archive.org/details/musopen-chopin) | CC0 ✅ | ~8–12 h, 104 tracks | largest permissive real-piano block |
| [Musopen Kickstarter DVD](https://archive.org/details/musopen-lossless-dvd) | PD ✅ | several hours | real full-orchestra (Brahms 1–4, Beethoven 3, Tchaikovsky 6…) + string quartets |
| [Cantoría](https://zenodo.org/records/5878677) | CC-BY-4.0 ✅ | 11 songs, SATB | Spanish-language ensemble singing |
| [SingVERSE](https://huggingface.co/datasets/amphion/SingVERSE) | CC-BY-4.0 ✅ | 18.1 h EN/ZH vocal pairs | clean vocal audio / augmentation, no alignment |
| [Bach Violin Dataset](https://zenodo.org/records/6050245) | "Other (Open)", per-file — audit after download | 6.5 h solo violin | only substantial solo-violin corpus; conditional |
| [cc0-music-captioned](https://huggingface.co/datasets/mrfakename/cc0-music-captioned) | CC0 (HF card; aggregator provenance — gate per-track) | 8,680 tracks / 294 GB, captioned | large captioned instrumental pool |
| [VSCO 2 CE](https://versilian-studios.com/vsco-community/) | CC0 ✅ | ~3 GB samples | orchestral timbre augmentation only (notes, not songs) |

Confirmed blocked in round 2: PiJAMA + PIAST (NC — jazz/pop real piano stays an open gap),
DALI (NC-SA annotations, YouTube audio), JamendoLyrics (NC per-track), Saraga (CC-BY-NC — Hindi
gap stays open), JVS-MuSiC (research-only — Japanese beyond PJS stays open),
Aalto anechoic (contact-only). **Bangla singing: nothing exists under any open licence** —
the honest options are collection/commissioning, not downloading.
