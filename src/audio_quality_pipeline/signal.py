from __future__ import annotations

import array
import hashlib
import math
import sys
import wave
from pathlib import Path

from .models import SignalFeatures


class AudioDecodeError(RuntimeError):
    pass


def extract_wav_features(path: Path) -> SignalFeatures:
    """Bounded-memory, standard-library DSP checks for PCM WAV inputs.

    Keep this stage dependency-free and cheap. Transcode other containers to PCM WAV
    in ingestion, or add an ffmpeg decoder adapter without changing policy semantics.
    """
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            rate = wav.getframerate()
            frame_count = wav.getnframes()
            compression = wav.getcomptype()
            if compression != "NONE":
                raise AudioDecodeError(f"Unsupported compressed WAV codec: {compression}")
            if sample_width not in (1, 2, 3, 4):
                raise AudioDecodeError(f"Unsupported PCM sample width: {sample_width}")
            if not rate or not channels or not frame_count:
                raise AudioDecodeError("Empty or malformed WAV")
            stats = _StreamingStats(rate, sample_width)
            while chunk := wav.readframes(32_768):
                stats.update(chunk, channels)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioDecodeError(str(exc)) from exc
    if not stats.sample_count:
        raise AudioDecodeError("No decoded samples")
    return SignalFeatures(
        sample_rate_hz=rate,
        channels=channels,
        frame_count=frame_count,
        duration_s=frame_count / rate,
        peak_dbfs=_dbfs(stats.peak),
        rms_dbfs=_dbfs(math.sqrt(stats.sum_square / stats.sample_count)),
        dc_offset=stats.sum_value / stats.sample_count,
        clipping_ratio=stats.clipped / stats.sample_count,
        silence_ratio=stats.silence_ratio(),
        sha256=stats.digest.hexdigest(),
    )


class _StreamingStats:
    """Accumulates all L1 metrics while holding only one decoder chunk in memory."""

    def __init__(self, sample_rate: int, sample_width: int) -> None:
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.full_scale = 1.0 - 1.0 / (2 ** (8 * sample_width - 1))
        self.digest = hashlib.sha256()
        self.sample_count = 0
        self.sum_value = 0.0
        self.sum_square = 0.0
        self.peak = 0.0
        self.clipped = 0
        self.silence_window = max(1, int(sample_rate * 0.02))
        self.silence_frames = 0
        self.silent_frames = 0
        self.window_count = 0
        self.window_sum_square = 0.0

    def update(self, raw: bytes, channels: int) -> None:
        self.digest.update(raw)
        for value in _mixdown(_pcm_to_float(raw, self.sample_width), channels):
            self._add(value)

    def _add(self, value: float) -> None:
        self.sample_count += 1
        self.sum_value += value
        squared = value * value
        self.sum_square += squared
        self.peak = max(self.peak, abs(value))
        self.clipped += abs(value) >= self.full_scale
        self.window_count += 1
        self.window_sum_square += squared
        if self.window_count == self.silence_window:
            self._close_silence_window()

    def _close_silence_window(self) -> None:
        rms = math.sqrt(self.window_sum_square / self.window_count)
        self.silent_frames += _dbfs(rms) < -50.0
        self.silence_frames += 1
        self.window_count = 0
        self.window_sum_square = 0.0

    def silence_ratio(self) -> float:
        if self.window_count:
            self._close_silence_window()
        return self.silent_frames / self.silence_frames if self.silence_frames else 1.0


def _pcm_to_float(raw: bytes, width: int) -> list[float]:
    if width == 1:
        return [(byte - 128) / 128.0 for byte in raw]
    if width == 2:
        values = array.array("h")
        values.frombytes(raw)
        if values.itemsize != 2:
            raise AudioDecodeError("Unexpected int16 width")
        if sys.byteorder != "little":
            values.byteswap()
        return [value / 32768.0 for value in values]
    if width == 3:
        return [
            int.from_bytes(
                raw[index : index + 3] + (b"\xff" if raw[index + 2] & 0x80 else b"\x00"),
                "little",
                signed=True,
            )
            / 8388608.0
            for index in range(0, len(raw), 3)
        ]
    values = array.array("i")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return [value / 2147483648.0 for value in values]


def _mixdown(samples: list[float], channels: int) -> list[float]:
    return [
        math.fsum(samples[i : i + channels]) / channels for i in range(0, len(samples), channels)
    ]


def _dbfs(value: float) -> float:
    return 20 * math.log10(max(value, 1e-12))
