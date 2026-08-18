# SongForge data program — multi-corpus, no Slakh cap

Established 2026-08-18. SongForge's corpus is **multi-corpus by design**:
Slakh2100 is one source among several, never the total. The goal is a broad,
rich, professional-style free-form generator — not a 2,100-song Slakh
specialist. Nothing here changes the V1 smoke test already carded in
`benchmarks/EXPERIMENT_CARD.md`; this is the ladder above it.

## The corpora and their jobs

| | corpus | job | licence | status |
| - | --- | --- | --- | --- |
| A | Slakh2100-redux (1,710 unique tracks, 44.1 kHz) | arrangement, multi-instrument structure, stem-level captions | CC-BY-4.0 ✅ | V1 uses the Slakh-100 subset; V2 scales toward all 1,710 |
| B | FMA CC0/CC-BY subset | real full songs, arrangement richness, vocals at scale | per-track, censused below | **metadata censused 2026-08-18; audio not yet downloaded** |
| C | VocalSet + Dagstuhl ChoirSet + UPF Choral (+ vocadito eval) | singing realism, vocal techniques, choral texture | CC-BY-4.0 ✅ | verified, not downloaded |
| D | E-GMD | realistic drum performance | CC-BY-4.0 ✅ | start with a diverse subset (kits × tempi × styles, ~10–20 GB), never blindly all 132 GB |
| E | GuitarSet | real acoustic-guitar timbre | CC-BY-4.0 ✅ (gate must re-verify at download time) | verified, not downloaded |
| F | additional verified CC0/CC-BY corpora | strings/violin, piano, orchestral, multilingual singing | per-source | open search track; Musopen PD picks are the current best lead; the permissive-piano and permissive-multilingual-singing gaps are real (see DATA_LICENSING_SHORTLIST.md) |

Every corpus passes the seven-gate dataset admission check and the licence
gate before a byte of audio is trained on. Research-only data never enters
the deployable line.

## FMA census — exact numbers (metadata only, computed before any audio)

`scripts/fma_license_report.py` over the official `tracks.csv`
(106,574 tracks), full report in `benchmarks/fma_license_report.json`:

| bucket | tracks | hours | ~GB |
| --- | ---: | ---: | ---: |
| CC0 / public domain | 1,820 | 97.2 | 10.7 |
| CC BY | 7,019 | 508.9 | 59.4 |
| CC BY-SA (flagged, decision needed) | 2,802 | 180.6 | 22.5 |
| CC BY-NC (never deployable) | 93,713 | 7,353.4 | 872.9 |
| CC BY-ND (never deployable) | 903 | 58.9 | 6.2 |
| other/unknown | 317 | 26.4 | 2.5 |

**Deployable default (CC0 + CC BY): 8,839 tracks, 606 hours, ~70 GB.**
Adding BY-SA would bring 11,641 tracks / 787 h / ~93 GB — a separate decision
because share-alike scope for model weights needs one legal read.

Accepted-subset genre spread (genre_top): Electronic 635, Rock 534, Old-Time/
Historic 454, Pop 379, Experimental 346, Hip-Hop 330, Classical 328,
Instrumental 200, Folk 178 — plus 5,310 untagged (fine-grained `genres` ids
exist for most and are resolved at adapter-build time). Vocal/language proxy:
1,109 tagged `en`, small fr/es/it/pt/ar tails, 7,608 untagged — tracks.csv has
**no instrument tags**, so true vocal/instrument coverage is computed by
audio-side tagging after download, not guessed from metadata.

Known caveat, carried into the gate: ~14% of FMA's CC0 labels were historically
misfiled; high-stakes tracks are re-verified against freemusicarchive.org
before deployable training.

## One canonical manifest

Corpora are never concatenated blindly. Every dataset adapter emits
`songforge.data.corpus_manifest.CorpusRecord`:

```
dataset · track_id · audio_path · caption · lyrics · instrument_tags · genre
bpm · key · licence · licence_class · source_url · artist · language · split
quality_score · duplicate_hash
```

Validation refuses a `permissive` claim for any licence not on the explicit
allowlist; `assert_deployable` refuses a deployable training mix containing a
research-only record; `assert_no_cross_corpus_duplicates` refuses the same
audio arriving twice through different corpora. Unknown stays None — a
missing BPM is never invented.

## 100-hour scaling strategy

1. **V1 (now): Slakh-100 smoke test.** 100–200 songs exist to prove the
   recipe produces positive learning — nothing else. Carded, gated on the
   frozen-8 + generalization ablation.
2. **V2 (immediately after V1 passes): broad run.** ~1,000–1,700 unique
   Slakh songs + the FMA deployable subset (start with the best-tagged few
   thousand) + VocalSet stack + E-GMD diverse subset + GuitarSet, all through
   the canonical manifest with cross-corpus dedup and per-corpus mixing
   weights. The final corpus may contain many thousands of examples.
3. **V3+: fill measured gaps** — whatever the V2 ablation says is still weak,
   sourced through track F.

## Storage rule

Current: ~175 GB free on Drive. The 2 TB upgrade is recommended **the moment
the approved working set — raw + processed + checkpoints + evaluation renders
+ 25% headroom — exceeds ~140 GB**, which happens at V2 approval, not before:

| stage | raw kept | processed | ckpts + eval | working set |
| --- | ---: | ---: | ---: | ---: |
| V1 (Slakh-100) | ~5 GB | ~16 GB | ~3 GB | **~30 GB — fits now** |
| V2 (full Slakh + FMA-deployable + C/D/E) | ~200 GB | ~120 GB | ~15 GB | **~420 GB — buy the 2 TB at V2 go-decision** |

Final quality is never constrained to 175 GB; the constraint is only that
storage is bought when a carded, approved run actually needs it.
