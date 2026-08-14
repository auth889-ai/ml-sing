# Dataset Plan

M01 keeps dataset access explicit, license-aware, and reproducible. The canonical machine-readable registry is `configs/data/datasets.yaml`; this document explains how to use it.

## Registered Datasets

| Dataset | Use | Access | License posture |
| --- | --- | --- | --- |
| BabySlakh | M02/M03 debug audio+MIDI | Zenodo, 883 MB | CC-BY-4.0 |
| Slakh2100-redux | multitrack audio+MIDI | Zenodo, 104.3 GB compressed | CC-BY-4.0 |
| Lakh MIDI | symbolic planner pretraining | official project page | CC-BY-4.0 distribution; preserve MIDI metadata |
| GTSinger | singing voice synthesis | Hugging Face dataset | CC-BY-NC-SA-4.0, non-commercial |
| MTG-Jamendo | tagged full-track audio | official scripts | per-track Creative Commons; preserve track licenses |
| NSynth | note/timbre audio pretraining | TensorFlow Datasets or Magenta archives | CC-BY-4.0 |

## Usage Order

1. BabySlakh: debug registry, audio decode, segmentation, MIDI parsing, and tiny overfit runs.
2. Slakh2100-redux: train codec and accompaniment pipelines after M02 passes.
3. Lakh MIDI: train symbolic tokenizer/planner after M05 passes.
4. GTSinger: train singer preprocessing/model after M07 passes; non-commercial only.
5. MTG-Jamendo: style/tag conditioning with track-level license retention.
6. NSynth: optional note-level timbre pretraining; not a full-song source.

## Split Rules

- Track IDs are split-level units. Segments from one track never cross train/val/test.
- Singing experiments must support singer-disjoint evaluation. A singer ID in validation/test must not appear in train for that run.
- Audio and MIDI manifests must store `source`, `license`, `track_id`, and local SHA-256 values after files are downloaded.
- For Slakh, respect redux/omitted duplicate-MIDI guidance before transcription-style training.
- MTG-Jamendo manifests must retain the per-track license from `audio_licenses.txt`.

## Remote Download Policy

Do not download large or gated datasets on the local workstation. Use the Colab workflow in `docs/COLAB_REMOTE_TRAINING.md` and keep `SONGFORGE_DATA` on Google Drive or Colab storage.

Gated or non-commercial datasets require explicit user acceptance in the Colab notebook before commands run. This currently includes GTSinger and MTG-Jamendo.

Validate metadata without downloading data:

```bash
python scripts/validate_dataset_registry.py
```

## Sources Checked

- Slakh/BabySlakh: https://www.slakh.com/ and Zenodo records `4599666`, `4603844`
- Lakh MIDI: https://colinraffel.com/projects/lmd/
- GTSinger: https://huggingface.co/datasets/YGGYY/GTSinger
- MTG-Jamendo: https://github.com/MTG/mtg-jamendo-dataset
- NSynth: https://magenta.tensorflow.org/datasets/nsynth
