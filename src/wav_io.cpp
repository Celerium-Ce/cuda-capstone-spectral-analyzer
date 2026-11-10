// Copyright 2026 Celerium-Ce
#include "include/wav_io.h"

#include <cstdint>
#include <cstring>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool TagEquals(const char* tag, const char* expected) {
  return std::memcmp(tag, expected, 4) == 0;
}

}  // namespace

bool ReadWav(const std::string& path, Signal* signal) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Cannot open file: " << path << "\n";
    return false;
  }

  char riff_tag[4];
  char wave_tag[4];
  uint32_t riff_size = 0;
  in.read(riff_tag, 4);
  in.read(reinterpret_cast<char*>(&riff_size), 4);
  in.read(wave_tag, 4);
  if (!in || !TagEquals(riff_tag, "RIFF") || !TagEquals(wave_tag, "WAVE")) {
    std::cerr << "Not a RIFF/WAVE file: " << path << "\n";
    return false;
  }

  bool have_fmt = false;
  bool have_data = false;
  uint16_t audio_format = 0;
  uint16_t num_channels = 0;
  uint16_t bits_per_sample = 0;
  uint32_t sample_rate = 0;
  std::vector<char> data_bytes;

  while (in && !(have_fmt && have_data)) {
    char chunk_id[4];
    uint32_t chunk_size = 0;
    in.read(chunk_id, 4);
    in.read(reinterpret_cast<char*>(&chunk_size), 4);
    if (!in) {
      break;
    }

    if (TagEquals(chunk_id, "fmt ")) {
      uint32_t byte_rate = 0;
      uint16_t block_align = 0;
      in.read(reinterpret_cast<char*>(&audio_format), 2);
      in.read(reinterpret_cast<char*>(&num_channels), 2);
      in.read(reinterpret_cast<char*>(&sample_rate), 4);
      in.read(reinterpret_cast<char*>(&byte_rate), 4);
      in.read(reinterpret_cast<char*>(&block_align), 2);
      in.read(reinterpret_cast<char*>(&bits_per_sample), 2);
      constexpr uint32_t kPcmFmtBytes = 16;
      if (chunk_size > kPcmFmtBytes) {
        in.seekg(chunk_size - kPcmFmtBytes, std::ios::cur);
      }
      have_fmt = true;
    } else if (TagEquals(chunk_id, "data")) {
      data_bytes.resize(chunk_size);
      in.read(data_bytes.data(), chunk_size);
      have_data = true;
    } else {
      in.seekg(chunk_size, std::ios::cur);
    }
    if (chunk_size % 2 == 1) {
      in.seekg(1, std::ios::cur);  // Chunks are padded to even byte counts.
    }
  }

  if (!have_fmt || !have_data) {
    std::cerr << "Missing fmt or data chunk: " << path << "\n";
    return false;
  }
  if (audio_format != 1 || num_channels != 1 || bits_per_sample != 16) {
    std::cerr << "Only mono 16-bit PCM WAV is supported: " << path << "\n";
    return false;
  }

  signal->sample_rate = static_cast<int>(sample_rate);
  const size_t num_samples = data_bytes.size() / sizeof(int16_t);
  signal->samples.resize(num_samples);
  const int16_t* pcm = reinterpret_cast<const int16_t*>(data_bytes.data());
  for (size_t i = 0; i < num_samples; ++i) {
    signal->samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
  }
  return true;
}
