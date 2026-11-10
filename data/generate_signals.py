#!/usr/bin/env python3
"""Synthesizes a batch of mono 16-bit PCM WAV files for the spectral
analyzer to process, so the project needs no network access or external
audio archive to run. Each file's injected frequency content is recorded
in its filename, so a reviewer can sanity-check the program's output CSV
against a known answer (e.g. "tone_0440hz.wav" should show its spectral
peak at 440 Hz).

Usage: python3 generate_signals.py [num_tone_files]
"""
import struct
import sys
import wave

import numpy as np

SAMPLE_RATE = 16000
DURATION_S = 2.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_S)
RNG = np.random.default_rng(seed=42)


def write_wav(path, samples):
    """Writes float32 samples in [-1, 1] as a mono 16-bit PCM WAV file."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(struct.pack("<%dh" % len(pcm16), *pcm16))


def tone(freq_hz, amplitude=0.6):
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def chirp(start_hz, end_hz, amplitude=0.6):
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    k = (end_hz - start_hz) / DURATION_S
    phase = 2 * np.pi * (start_hz * t + 0.5 * k * t * t)
    return amplitude * np.sin(phase)


def two_tone(freq_a, freq_b, amplitude=0.4):
    return tone(freq_a, amplitude) + tone(freq_b, amplitude)


def noisy_tone(freq_hz, noise_level=0.15, amplitude=0.6):
    return tone(freq_hz, amplitude) + noise_level * RNG.standard_normal(NUM_SAMPLES)


def main():
    out_dir = "input"
    tone_freqs = [220, 440, 880, 1760, 2500, 3200, 4400, 5500]
    if len(sys.argv) > 1:
        tone_freqs = tone_freqs[: int(sys.argv[1])]

    for freq in tone_freqs:
        write_wav(f"{out_dir}/tone_{freq:04d}hz.wav", tone(freq))

    for start, end in [(200, 2000), (500, 5000), (3000, 500)]:
        write_wav(f"{out_dir}/chirp_{start:04d}to{end:04d}hz.wav",
                   chirp(start, end))

    for freq_a, freq_b in [(440, 660), (1000, 1500), (220, 880)]:
        write_wav(f"{out_dir}/two_tone_{freq_a:04d}_{freq_b:04d}hz.wav",
                   two_tone(freq_a, freq_b))

    for freq in [440, 1000, 2000]:
        write_wav(f"{out_dir}/noisy_{freq:04d}hz.wav", noisy_tone(freq))

    write_wav(f"{out_dir}/white_noise.wav", 0.4 * RNG.standard_normal(NUM_SAMPLES))

    total = len(tone_freqs) + 3 + 3 + 3 + 1
    print(f"Wrote {total} synthetic WAV files to {out_dir}/ "
          f"({SAMPLE_RATE} Hz, {DURATION_S}s, mono 16-bit PCM)")


if __name__ == "__main__":
    main()
