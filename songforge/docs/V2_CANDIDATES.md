# V2 corpus candidates — prepared while V1 trains

Drafted 2026-08-18 during the V1 run. Nothing here downloads or trains until
V1's ablation passes and the V2 experiment card is written.

## Slakh V2 manifest candidates

V2 scales Slakh from the 100-track subset toward the full deduplicated redux:

- **Pool:** all 1,710 redux tracks (1,289/270/151 official splits), 44.1 kHz.
  The archive is already on the Colab path for V1 — V2 extraction reuses the
  same one-time download if run in the same session, else re-downloads.
- **Selection:** `scripts/select_slakh100.py` generalizes — the same eligibility
  rules with `splits: {train: 1289, val: 270, test: 151}` degenerate to
  "every eligible track", so the selector doubles as the V2 eligibility filter
  (unrendered stems and <6-stem tracks still excluded; expect ~1,500–1,650
  survivors — exact count reported by the selector at build time).
- **Sizing (scaled from Slakh-100 estimates):** ~50 GB kept raw FLAC,
  ~160 GB processed 60 s segments at 44.1 kHz — the working set that triggers
  the 2 TB Drive purchase.
- **Manifest:** through `CorpusRecord` with `dataset: slakh2100_redux`,
  cross-corpus dedup against every other V2 source.

## V2 mixing candidates (all licence-verified, priority order)

1. **FMA deployable subset** — 8,839 tracks / 606 h / ~70 GB (CC0+CC-BY,
   censused). Fine-grained genres now resolved: only 479 tracks truly
   untagged; top genres Pop 1322, Rock 1225, Experimental 1222, Folk 803,
   Electronic 775, Classical 519, Ambient 464, Jazz 189. Start with the
   best-tagged ~3–5k tracks, re-verify CC0-labelled tracks per the known
   ~14% misfiling.
2. **SingStyle111** — 12.8 h lyric-aligned pro singing (CC-BY) — the vocal
   centrepiece V1 lacked.
3. **VocalSet + Dagstuhl + UPF Choral** — vocal timbre + choral texture.
4. **E-GMD diverse subset** — sample across the 43 kits × tempo × style grid,
   ~10–20 GB, never all 132 GB.
5. **GuitarSet** — 3 h real acoustic guitar.
6. **CC0 piano stack** — Open Goldberg + Open WTC + Musopen Chopin (~12–16 h
   real studio piano).
7. **Musopen orchestral PD** — real full-orchestra hours.
8. **Bach Violin Dataset** — after its per-file licence audit passes the gate.

Per-corpus mixing weights are a V2-card decision, made against V1's measured
weaknesses — not fixed here.
