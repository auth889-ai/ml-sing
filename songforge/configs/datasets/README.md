# Dataset build recipes

One YAML per dataset build, so a corpus can be rebuilt byte-identically and its
provenance traced from a checkpoint back to a download.

Required keys: `dataset_id`, `variant`, `source_url`, `licence`, `download_gb`,
`processed_gb`, `tracks`, `split_strategy`, `instrument_stratification`, `goal`.

Sizes are recorded BEFORE download, not after, so the storage cost of a decision
is known while it is still a decision.
