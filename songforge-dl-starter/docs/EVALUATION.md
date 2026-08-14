# Evaluation plan

## Codec
- waveform L1 / SI-SDR (diagnostic)
- multi-resolution STFT distance
- codebook utilization/perplexity
- bitrate / tokens per second
- blind A/B listening export

## Planner
- token NLL / perplexity
- invalid-event rate
- pitch-class distribution
- note-duration distribution
- repetition / n-gram statistics
- structure validity (bars/sections)

## Singer
- diffusion validation loss
- F0 RMSE / voiced-unvoiced accuracy where annotation supports it
- phoneme-duration/alignment diagnostics
- mel/STFT distance
- MOS-style listening-study export

## Full song
- prompt/tag consistency using a separately trained evaluator only as an evaluation tool
- beat/tempo adherence
- clipping/loudness checks
- diversity across seeds
- human ratings: musicality, vocal intelligibility, prompt adherence, audio quality

Never report a single loss value as "Suno quality".
