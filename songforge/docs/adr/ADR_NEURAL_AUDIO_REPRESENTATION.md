# ADR: Neural Audio Representation

## Status

Accepted for M03. Revisit in M04 after benchmark results.

## Context

SongForge needs a learned audio representation before any long-form song generator is built. The representation must reconstruct music, preserve pitch/transients/vocals, shorten sequence length enough for 30-60 second and later multi-minute generation, and remain compatible with later autoregressive, Diffusion Transformer, or flow-matching models.

M03 constraints:

- Google Colab Pro class compute, not a research cluster.
- Small model from our own random initialization.
- No pretrained flagship weights and no hosted inference APIs.
- 24 kHz mono first; stereo is a later extension.
- Demonstrate a real segment -> encoded representation -> decoded reconstruction.

## Options Compared

### 1. Continuous VAE Latent Representation

Continuous latents are natural for diffusion, DiT, and flow matching. Stable Audio-style systems show that latent audio modeling can support long-form generation when the waveform is first compressed by an autoencoder. Continuous latents also avoid codebook collapse and can preserve fine detail if the bottleneck is not too aggressive.

Tradeoffs:

- Good match for diffusion/flow training.
- Harder to use directly with discrete autoregressive token models.
- Latent entropy/rate is less explicit than a token bitrate.
- A KL-heavy VAE may blur transients and pitch unless trained carefully.
- Strong perceptual reconstruction typically needs more compute than the M03 budget.

### 2. Vector-Quantized Codec

A single-codebook VQ codec gives discrete symbols and an explicit compression interface. This matches the original VQ-VAE lineage and makes future token language modeling straightforward.

Tradeoffs:

- Simple to implement and test.
- Directly supports token streams.
- Single codebook often has insufficient bandwidth for music, vocals, and transients at useful frame rates.
- Codebook collapse/dead codes must be monitored.

### 3. Residual Vector Quantization

Residual vector quantization stacks several codebooks, quantizing the remaining residual at each stage. SoundStream/EnCodec/DAC-style codecs use this family because it provides scalable bitrate: fewer codebooks for low bitrate, more codebooks for quality. MusicGen-style systems then model EnCodec token streams as discrete sequences.

Tradeoffs:

- Explicit bitrate and frame rate.
- Token streams are compatible with autoregressive models.
- Quantized embeddings can also be treated as latents for diffusion/DiT conditioning.
- Scales naturally from tiny Colab experiments to larger M04 codecs.
- Needs commitment/codebook losses and codebook utilization monitoring.

### 4. Very-Low-Frame-Rate Representation

Modern song-generation systems often need very low temporal rates so generation over minutes is tractable. The attraction is clear: a 60-second clip at 25-50 Hz is only 1500-3000 latent frames before considering multiple streams.

Tradeoffs:

- Best long-form sequence length.
- Risky for M03 because low frame rate can discard transients, consonants, and fast pitch changes.
- Usually requires a stronger decoder and better perceptual/adversarial losses than M03 should take on.
- Better handled as an M04/M04+ ablation after a normal codec is working.

### 5. Discrete Codec Token Streams

Discrete token streams are the downstream interface produced by a VQ/RVQ codec. They can feed MusicGen-style autoregressive models, masked modeling, diffusion over embeddings, or hierarchical planners.

Tradeoffs:

- Natural interface for long-context language-model tooling.
- Multi-codebook streams increase sequence count; flattening streams can be expensive.
- Needs careful design later: interleaving, codebook delay patterns, hierarchical coding, or semantic/acoustic separation.

## Decision

For M03, implement a small residual-vector-quantized neural codec:

```text
waveform -> convolutional residual encoder -> RVQ code streams -> convolutional residual decoder -> waveform
```

Initial target:

- 24 kHz mono.
- Downsample factor 200 by default, giving 120 latent frames/sec.
- Tiny Colab config: 2 codebooks, 128 entries each.
- M03 loss: waveform L1 + multi-resolution STFT + RVQ commitment/codebook losses.
- M03 metrics: L1, multi-resolution STFT, spectral convergence, SNR, frame rate, bitrate, compression ratio, codebook usage.

Rationale:

- RVQ is the most practical bridge between codec reconstruction and future token/latent generation.
- It gives an explicit compression interface now without forcing the final generator design.
- Colab can train a tiny overfit/debug model from scratch, while M04 can scale model size, codebooks, frame rates, and losses.
- Later lyrics/vocal conditioning can target token streams, quantized embeddings, or continuous latents before quantization.

## Consequences

- M03 does not attempt Suno-class fidelity.
- M03 does not add a discriminator. Adversarial/perceptual quality work belongs in M04.
- M03 must prove shape safety, finite losses, gradient flow, checkpoint resume, deterministic eval, WAV export, and compression statistics.
- M04 must benchmark at least three codec configurations before freezing the representation interface for later SongForge models.

## Research References

- SoundStream: https://arxiv.org/abs/2107.03312
- EnCodec: https://arxiv.org/abs/2210.13438 and https://github.com/facebookresearch/encodec
- MusicGen: https://arxiv.org/abs/2306.05284 and https://github.com/facebookresearch/audiocraft
- DAC: https://arxiv.org/abs/2306.06546 and https://github.com/descriptinc/descript-audio-codec
- Stable Audio Open: https://arxiv.org/abs/2407.14358 and https://github.com/Stability-AI/stable-audio-tools
- Jukebox VQ-VAE music tokens: https://arxiv.org/abs/2005.00341
