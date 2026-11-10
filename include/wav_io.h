// Copyright 2026 Celerium-Ce
#ifndef INCLUDE_WAV_IO_H_
#define INCLUDE_WAV_IO_H_

#include <string>
#include <vector>

// In-memory representation of a mono audio signal.
struct Signal {
  int sample_rate = 0;
  std::vector<float> samples;  // normalized to [-1, 1]
};

// Reads a mono, 16-bit PCM WAV file from `path` into `*signal`. Returns
// false and logs a message to stderr on any I/O or format error (including
// a stereo or non-PCM16 file, which this project does not need to handle).
bool ReadWav(const std::string& path, Signal* signal);

#endif  // INCLUDE_WAV_IO_H_
