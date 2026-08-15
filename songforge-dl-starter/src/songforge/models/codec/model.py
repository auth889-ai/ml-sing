from __future__ import annotations

import math

import torch
from torch import nn

from .quantizer import ResidualVectorQuantizer


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation * 3
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _deconv_padding(stride: int) -> tuple[int, int]:
    padding = math.ceil(stride / 2)
    output_padding = 2 * padding - stride
    return padding, output_padding


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        padding = math.ceil(stride / 2)
        self.net = nn.Sequential(
            ResidualBlock(in_channels, dilation=1),
            ResidualBlock(in_channels, dilation=3),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        padding, output_padding = _deconv_padding(stride)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            ResidualBlock(out_channels, dilation=1),
            ResidualBlock(out_channels, dilation=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CodecEncoder(nn.Module):
    def __init__(self, channels: int, base_channels: int, latent_dim: int, strides: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(channels, base_channels, kernel_size=7, padding=3)]
        in_channels = base_channels
        for i, stride in enumerate(strides):
            out_channels = base_channels * min(2 ** (i + 1), 8)
            layers.append(EncoderBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        layers.extend([nn.GELU(), nn.Conv1d(in_channels, latent_dim, kernel_size=3, padding=1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CodecDecoder(nn.Module):
    def __init__(self, channels: int, base_channels: int, latent_dim: int, strides: tuple[int, ...]):
        super().__init__()
        encoder_channels = [base_channels * min(2 ** (i + 1), 8) for i in range(len(strides))]
        current = encoder_channels[-1] if encoder_channels else base_channels
        layers: list[nn.Module] = [nn.Conv1d(latent_dim, current, kernel_size=3, padding=1)]
        for stride, out_channels in zip(reversed(strides), reversed([base_channels] + encoder_channels[:-1])):
            layers.append(DecoderBlock(current, out_channels, stride))
            current = out_channels
        layers.extend([nn.GELU(), nn.Conv1d(current, channels, kernel_size=7, padding=3), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class NeuralCodec(nn.Module):
    """M03 neural audio codec: Conv encoder, RVQ bottleneck, Conv decoder."""

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 64,
        codebook_size: int = 256,
        num_quantizers: int = 4,
        strides: tuple[int, ...] | list[int] = (2, 4, 5, 5),
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.strides = tuple(strides)
        self.downsample_factor = math.prod(self.strides)

        self.encoder = CodecEncoder(channels, base_channels, latent_dim, self.strides)
        self.quantizer = ResidualVectorQuantizer(latent_dim, codebook_size, num_quantizers, commitment_weight)
        self.decoder = CodecDecoder(channels, base_channels, latent_dim, self.strides)

    def _pad_to_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        length = x.shape[-1]
        padded_length = math.ceil(length / self.downsample_factor) * self.downsample_factor
        pad = padded_length - length
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        return x, pad

    def encode(self, x: torch.Tensor):
        if x.ndim != 3 or x.size(1) != self.channels:
            raise ValueError(f"Expected [B,{self.channels},T], got {tuple(x.shape)}")
        padded, pad = self._pad_to_multiple(x)
        z = self.encoder(padded)
        q = self.quantizer(z)
        q["latents"] = z
        q["pad"] = torch.tensor(pad, device=x.device)
        q["input_length"] = torch.tensor(x.shape[-1], device=x.device)
        return q

    def decode(self, zq: torch.Tensor) -> torch.Tensor:
        return self.decoder(zq)

    def decode_codes(self, codes: torch.Tensor, length: int | None = None) -> torch.Tensor:
        zq = self.quantizer.decode_codes(codes)
        reconstruction = self.decode(zq)
        if length is not None:
            reconstruction = reconstruction[..., :length]
        return reconstruction

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        encoded = self.encode(x)
        zq = encoded["quantized"]
        reconstruction = self.decode(zq)
        reconstruction = reconstruction[..., : x.shape[-1]]
        codes = encoded["codes"]
        return {
            "reconstruction": reconstruction,
            "codes": codes,
            "indices": [codes[:, i] for i in range(codes.size(1))],
            "latents": encoded["latents"],
            "quantized": zq,
            "vq_loss": encoded["loss"],
            "codebook_loss": encoded["codebook_loss"],
            "commitment_loss": encoded["commitment_loss"],
        }

    def compression_stats(self, num_samples: int) -> dict[str, float]:
        frames = math.ceil(num_samples / self.downsample_factor)
        frame_rate = self.sample_rate / self.downsample_factor
        bits_per_code = math.ceil(math.log2(self.codebook_size))
        bitrate = frame_rate * self.num_quantizers * bits_per_code
        pcm16_bitrate = self.sample_rate * self.channels * 16
        return {
            "downsample_factor": float(self.downsample_factor),
            "latent_frames": float(frames),
            "latent_frame_rate_hz": float(frame_rate),
            "code_streams": float(self.num_quantizers),
            "bits_per_code": float(bits_per_code),
            "discrete_bitrate_bps": float(bitrate),
            "pcm16_compression_ratio": float(pcm16_bitrate / max(bitrate, 1.0)),
        }
