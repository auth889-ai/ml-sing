# Canonical Training Manifest

`songforge.data.manifest.AudioRecord` is the **single** manifest schema for this project.
There is no separate "processed", "segment", or "singing" record type, and adding one would
be a regression. Every milestone that consumes audio reads this format.

Schema id: `songforge.audio.v1` (`MANIFEST_SCHEMA`).

## Format

JSON Lines. One record per line, keys sorted, UTF-8, no trailing commas:

```text
manifests/
  all.jsonl              every segment
  train.jsonl            one file per split
  val.jsonl
  test.jsonl
  manifest_summary.json  counts and durations
```

Read and write it with the library, not by hand:

```python
from songforge.data.manifest import read_jsonl, write_jsonl, write_split_manifests

records = read_jsonl("manifests/train.jsonl")
write_split_manifests(records, "manifests")
```

## Fields

### Identity and licensing (required, from M01)

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | str | Stable segment id. Derived from content coordinates, not iteration order. |
| `path` | str | Path to the processed segment audio. |
| `split` | str | `train`, `val`, `test`, or `unassigned` before splitting. |
| `source` | str | Dataset id, matching a key in `configs/data/datasets.yaml`. |
| `license` | str | Licence name, propagated from the registry. |
| `track_id` | str | Source-song id. The unit of song-disjoint splitting. |
| `singer_id` | str \| null | Performer id where the corpus provides one. |
| `tags` | list[str] | Normalized: lowercased, underscored, de-duplicated, sorted. |

`REQUIRED_FIELDS` enumerates the six that must be non-empty, mirroring
`global_split_policy.manifest_required_fields` in the dataset registry.

### Preprocessing description (M02)

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | str | Schema id, for forward compatibility. |
| `source_path` | str | The raw file this segment came from. |
| `segment_index` | int | 0-based index within the source file. |
| `start_sample` | int | Offset into the resampled source. |
| `num_samples` | int | Segment length in samples. |
| `sample_rate` | int | Sample rate of the stored segment. |
| `channels` | int | Channel count of the stored segment. |
| `duration_seconds` | float | `num_samples / sample_rate`. Validated for consistency. |

### Amplitude (M02)

`peak`, `rms`, `peak_dbfs`, `rms_dbfs`, `clipping_ratio`, `silent`.

These describe the **stored** segment, after normalization. Defects of the original master
live in `preprocessing.source_clipping_ratio` and `preprocessing.source_peak_dbfs`, because
peak normalization pulls a clipped master down below full scale and would otherwise hide it.

### Integrity and duplicate detection (M02)

| Field | Meaning |
| --- | --- |
| `audio_sha256` | Content hash of the decoded segment, quantized to int16. Matches across containers. |
| `source_sha256` | SHA-256 of the raw source file bytes. |

`songforge.data.dedup` uses these for `find_duplicate_groups`,
`assert_no_cross_split_duplicates`, and `duplicate_report`. Near-duplicate detection
(fingerprinting, chroma matching) is a later milestone and extends `duplicate_report`.

### Provenance and reproducibility (M02)

`provenance` carries the dataset identity and licence forward from the registry:

```json
{
  "dataset_id": "babyslakh",
  "dataset_name": "BabySlakh",
  "role": "debug_multitrack_audio_midi",
  "source_url": "https://zenodo.org/records/4603844",
  "citation_url": "https://www.slakh.com/",
  "license_name": "CC-BY-4.0",
  "license_url": "https://creativecommons.org/licenses/by/4.0/",
  "commercial_allowed": true,
  "requires_user_acceptance": false,
  "access_method": "zenodo"
}
```

`assert_provenance_complete` requires `dataset_id`, `source_url`, and `license_name` on every
record, so a corpus cannot lose its licence on the way into training.

`preprocessing` records every setting that changed the audio, plus what the source looked
like, so a manifest is reproducible without the original config file:

```json
{
  "version": "m02.v1",
  "sample_rate": 24000,
  "channels": 1,
  "segment_seconds": 2.0,
  "normalize": "peak",
  "target_dbfs": -1.0,
  "source_sample_rate": 16000,
  "source_codec": "pcm_s16le",
  "applied_gain_db": 4.21,
  "source_clipping_ratio": 0.0
}
```

`extra` holds fields from newer writers that this version does not know about, so an older
reader round-trips a newer manifest without dropping data.

## Compatibility

Every M02 field is optional and defaulted. A minimal M01-era record still loads:

```json
{"id": "1", "path": "a.wav", "split": "train", "source": "babyslakh",
 "license": "CC-BY-4.0", "track_id": "Track00001"}
```

## Invariants

Enforced by `scripts/preprocess_dataset.py` and `scripts/run_preprocessing_acceptance.py`:

- `validate_records` — required fields present, ids unique per path, duration consistent.
- `assert_provenance_complete` — licence and source survived preprocessing.
- `assert_no_track_leakage` / `assert_group_disjoint` — a song lives in exactly one split.
- `assert_singer_disjoint` — no performer appears in both train and an eval split.
- `assert_no_cross_split_duplicates` — identical audio does not straddle splits.

## Paths are environment-specific

`path` and `source_path` are absolute for the machine that ran preprocessing. Manifests are
regenerated per environment rather than copied between machines; the reproducible parts are
`id`, `audio_sha256`, `segment_index`, and `start_sample`, which are identical everywhere for
the same input and config.
