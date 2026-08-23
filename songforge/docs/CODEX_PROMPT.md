# Master prompt for Codex

Read `AGENTS.md`, `docs/ARCHITECTURE.md`, `docs/DATASETS.md`, and all current tests before changing code.

Build SongForge-DL milestone-by-milestone. Do not call hosted music-generation APIs and do not silently substitute pretrained MusicGen/ACE-Step/Suno-like weights for missing components. The flagship path must train and load local weights produced by this repository.

For every milestone:
1. state the acceptance criteria;
2. implement the smallest correct version;
3. add/extend tests;
4. run tests;
5. run a CPU smoke experiment when possible;
6. document exact command, config, output, and known limitations;
7. only then mark the milestone PASS.

Start with M01 dataset registry/licensing and M02 preprocessing after confirming the existing M00 smoke tests pass.
