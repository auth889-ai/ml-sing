# Colab Remote Training

This project should not fetch huge datasets or train flagship models on the local workstation. Use `notebooks/songforge_colab_remote.ipynb` from the Google account that owns the target Drive storage.

## What Codex Can and Cannot Do

Codex cannot sign into `anastasiasaat81@gmail.com` or operate that Google Colab account from this environment. The notebook is prepared so the account owner can run the cells in Colab after authorizing Drive and any dataset provider credentials.

## Colab Steps

1. Open Colab from the target Google account.
2. Upload or open `notebooks/songforge_colab_remote.ipynb`.
3. Set `REPO_URL`, `BRANCH`, `SONGFORGE_DATA`, and the dataset choices in the first code cell.
4. Run the setup cells. They clone/pull the repo into Colab, mount Google Drive, install dependencies, and validate the dataset registry.
5. Accept dataset terms in the notebook before running any GTSinger or MTG-Jamendo download cell.
6. Run only milestone-appropriate jobs:
   - M00: `pytest -q`
   - M01: `python scripts/validate_dataset_registry.py`
   - M02/M03 later: preprocessing and codec trainers after they are implemented

## Storage Layout

```text
/content/drive/MyDrive/songforge-dl/
  data/
    raw/
    processed/
    manifests/
  checkpoints/
  logs/
  repo/
```

Keep raw datasets and checkpoints out of git. The repository `.gitignore` already blocks `data/raw`, `data/processed`, checkpoints, logs, audio files, and PyTorch weight files.

## Push Results

Only push source code, configs, docs, small manifests, and experiment metadata. Do not push raw datasets, generated WAV/MP3/FLAC files, or large checkpoints.
