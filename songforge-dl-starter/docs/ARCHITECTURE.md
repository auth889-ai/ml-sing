# Architecture

## V1 system

```text
prompt + lyrics
      |
      v
Symbolic Song Planner Transformer
      |
      +--> melody/chords/sections/tempo
                |                 |
                v                 v
        Singing Acoustic      Accompaniment
        Diffusion Model       Transformer
                |                 |
                v                 v
           vocal audio      instrumental audio
                \                 /
                 \               /
                    local mixer
                        |
                    final.wav
```

## Audio codec path

```text
waveform -> Conv1d encoder -> residual vector quantizer -> ConvTranspose1d decoder -> waveform
```

Train codec independently first. When stable, accompaniment may generate codec tokens rather than waveform samples.

## Planner

Decoder-only Transformer on MIDI/event tokens. Add conditioning tokens for genre, mood, BPM, key and section. Start symbolic because it is cheaper and easier to debug than direct raw-audio generation.

## Singer

Condition diffusion noise prediction on phoneme, note, duration, pitch and singer/style embeddings. Start with mel-spectrogram output; add local neural vocoder only after mel generation works.

## Scaling strategy

- V1: 5-10 s, 24 kHz mono, small models.
- V2: 10-30 s, better codec/vocoder, multi-section planning.
- V3: 30-60 s hierarchical plan + chunked audio generation.
- V4: long-form song structure and higher fidelity.
