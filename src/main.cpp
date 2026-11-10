// Copyright 2026 Celerium-Ce
//
// GPU Specialization Capstone Project.
//
// Batch audio spectral analyzer: for every mono 16-bit PCM WAV file in an
// input directory, computes a short-time Fourier transform (STFT) on the
// GPU and writes a frequency-over-time spectrogram CSV. Framing and
// Hann-windowing is a hand-written CUDA kernel, the FFT itself is cuFFT,
// and converting the complex spectrum to magnitude-in-decibels is a second
// hand-written kernel. Files are distributed round-robin across several
// CUDA streams with pinned host buffers so transfers and compute for
// different files can overlap, the same "at scale" pattern used in the
// CUDA at Scale independent project this course built on.
//
// Directory handling is done with POSIX dirent/sys-stat calls rather than
// std::filesystem, so the build has no dependency on a GCC version new
// enough to ship <filesystem> without a separate -lstdc++fs link.

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "include/cuda_utils.h"
#include "include/magnitude_kernel.cuh"
#include "include/wav_io.h"
#include "include/window_kernel.cuh"

namespace {

struct Args {
  std::string input_dir = "data/input";
  std::string output_dir = "output";
  std::string log_file = "logs/run.log";
  int frame_size = 1024;
  int hop_size = 512;
  int num_streams = 4;
  int64_t limit = -1;  // -1 means "process every file found".
};

void PrintUsage(const char* prog_name) {
  std::cout
      << "Usage: " << prog_name << " [options]\n"
      << "  --input DIR     Directory of input .wav files (default: data/input)\n"
      << "  --output DIR    Directory to write spectrogram .csv files (default: output)\n"
      << "  --log FILE      Path to write the run log (default: logs/run.log)\n"
      << "  --frame-size N  STFT window length, must be <= 1024 (default: 1024)\n"
      << "  --hop-size N    STFT hop length in samples (default: 512)\n"
      << "  --streams N     Number of concurrent CUDA streams (default: 4)\n"
      << "  --limit N       Process at most N files (default: all)\n"
      << "  -h, --help      Show this message\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto NextValue = [&](const char* flag_name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag_name << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--input") {
      args->input_dir = NextValue("--input");
    } else if (arg == "--output") {
      args->output_dir = NextValue("--output");
    } else if (arg == "--log") {
      args->log_file = NextValue("--log");
    } else if (arg == "--frame-size") {
      args->frame_size = std::stoi(NextValue("--frame-size"));
    } else if (arg == "--hop-size") {
      args->hop_size = std::stoi(NextValue("--hop-size"));
    } else if (arg == "--streams") {
      args->num_streams = std::stoi(NextValue("--streams"));
    } else if (arg == "--limit") {
      args->limit = static_cast<int64_t>(std::stoll(NextValue("--limit")));
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (args->num_streams < 1) {
    std::cerr << "--streams must be >= 1\n";
    return false;
  }
  if (args->frame_size < 2 || args->frame_size > 1024) {
    std::cerr << "--frame-size must be between 2 and 1024\n";
    return false;
  }
  if (args->hop_size < 1) {
    std::cerr << "--hop-size must be >= 1\n";
    return false;
  }
  return true;
}

bool DirExists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Creates `path` and any missing parent directories (like `mkdir -p`).
// Silently succeeds if a component already exists.
void MakeDirsRecursive(const std::string& path) {
  std::string partial;
  size_t pos = 0;
  while (pos < path.size()) {
    const size_t next = path.find('/', pos);
    const size_t end = (next == std::string::npos) ? path.size() : next;
    partial = path.substr(0, end);
    if (!partial.empty() && !DirExists(partial)) {
      mkdir(partial.c_str(), 0755);
    }
    if (next == std::string::npos) break;
    pos = next + 1;
  }
}

std::string ParentDir(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return (slash == std::string::npos) ? std::string() : path.substr(0, slash);
}

std::string BaseNameNoExt(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const std::string name =
      (slash == std::string::npos) ? path : path.substr(slash + 1);
  const size_t dot = name.find_last_of('.');
  return (dot == std::string::npos) ? name : name.substr(0, dot);
}

bool HasSuffix(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> CollectInputFiles(const std::string& input_dir,
                                            int64_t limit) {
  std::vector<std::string> files;
  DIR* dir = opendir(input_dir.c_str());
  if (dir == nullptr) {
    return files;
  }
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    const std::string name = entry->d_name;
    if (HasSuffix(name, ".wav")) {
      files.push_back(input_dir + "/" + name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  if (limit >= 0 && static_cast<int64_t>(files.size()) > limit) {
    files.resize(limit);
  }
  return files;
}

// Writes a frame-time-by-frequency-bin magnitude-in-decibels spectrogram:
// a header row of bin center frequencies (Hz), then one row per STFT
// frame labeled with that frame's start time (s). Returns false and logs
// a message to stderr if the file cannot be written.
bool WriteSpectrogramCsv(const std::string& path, const float* magnitude_db,
                          int num_frames, int num_bins, int sample_rate,
                          int frame_size, int hop_size) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Cannot open file for writing: " << path << "\n";
    return false;
  }

  out << "frame_time_s";
  for (int b = 0; b < num_bins; ++b) {
    const double freq_hz =
        static_cast<double>(b) * sample_rate / frame_size;
    out << "," << freq_hz;
  }
  out << "\n";

  for (int f = 0; f < num_frames; ++f) {
    const double frame_time = static_cast<double>(f) * hop_size / sample_rate;
    out << frame_time;
    for (int b = 0; b < num_bins; ++b) {
      out << "," << magnitude_db[f * num_bins + b];
    }
    out << "\n";
  }
  return out.good();
}

// Processes one WAV file end-to-end on `stream`: pinned H2D copy,
// windowing kernel, batched cuFFT R2C, magnitude kernel, pinned D2H copy,
// then writes the spectrogram CSV. Returns the elapsed GPU time in
// milliseconds, or a negative value if the file could not be read or
// written. Writes the frame/bin counts actually used to `*out_num_frames`
// / `*out_num_bins` for the caller's log line.
double ProcessSignal(const std::string& input_path,
                      const std::string& output_path, int frame_size,
                      int hop_size, cudaStream_t stream, int* out_num_frames,
                      int* out_num_bins) {
  Signal signal;
  if (!ReadWav(input_path, &signal)) {
    return -1.0;
  }

  const int signal_len = static_cast<int>(signal.samples.size());
  const int num_frames = (signal_len >= frame_size)
                              ? (signal_len - frame_size) / hop_size + 1
                              : 1;
  const int num_bins = frame_size / 2 + 1;
  const size_t frames_bytes =
      static_cast<size_t>(num_frames) * frame_size * sizeof(float);
  const size_t spectrum_bytes =
      static_cast<size_t>(num_frames) * num_bins * sizeof(cufftComplex);
  const size_t magnitude_bytes =
      static_cast<size_t>(num_frames) * num_bins * sizeof(float);

  float* h_pinned_signal = nullptr;
  float* h_pinned_magnitude = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_pinned_signal, signal_len * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_pinned_magnitude, magnitude_bytes));
  std::memcpy(h_pinned_signal, signal.samples.data(),
              signal_len * sizeof(float));

  float* d_signal = nullptr;
  float* d_frames = nullptr;
  cufftComplex* d_spectrum = nullptr;
  float* d_magnitude = nullptr;
  CUDA_CHECK(cudaMalloc(&d_signal, signal_len * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_frames, frames_bytes));
  CUDA_CHECK(cudaMalloc(&d_spectrum, spectrum_bytes));
  CUDA_CHECK(cudaMalloc(&d_magnitude, magnitude_bytes));

  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, frame_size, CUFFT_R2C, num_frames));
  CUFFT_CHECK(cufftSetStream(plan, stream));

  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));
  CUDA_CHECK(cudaEventRecord(start_event, stream));

  CUDA_CHECK(cudaMemcpyAsync(d_signal, h_pinned_signal,
                              signal_len * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
  LaunchWindowKernel(d_signal, signal_len, frame_size, hop_size, num_frames,
                     d_frames, stream);
  CUFFT_CHECK(cufftExecR2C(plan, d_frames, d_spectrum));
  LaunchMagnitudeKernel(d_spectrum, num_frames, num_bins, d_magnitude, stream);
  CUDA_CHECK(cudaMemcpyAsync(h_pinned_magnitude, d_magnitude, magnitude_bytes,
                              cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaEventRecord(stop_event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

  const bool wrote_ok =
      WriteSpectrogramCsv(output_path, h_pinned_magnitude, num_frames,
                           num_bins, signal.sample_rate, frame_size, hop_size);

  cufftDestroy(plan);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  cudaFreeHost(h_pinned_signal);
  cudaFreeHost(h_pinned_magnitude);
  cudaFree(d_signal);
  cudaFree(d_frames);
  cudaFree(d_spectrum);
  cudaFree(d_magnitude);

  if (!wrote_ok) {
    return -1.0;
  }
  *out_num_frames = num_frames;
  *out_num_bins = num_bins;
  return static_cast<double>(elapsed_ms);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    return EXIT_FAILURE;
  }

  if (!DirExists(args.input_dir)) {
    std::cerr << "Input directory does not exist: " << args.input_dir << "\n";
    return EXIT_FAILURE;
  }
  MakeDirsRecursive(args.output_dir);
  const std::string log_parent = ParentDir(args.log_file);
  if (!log_parent.empty()) {
    MakeDirsRecursive(log_parent);
  }

  const std::vector<std::string> files =
      CollectInputFiles(args.input_dir, args.limit);
  if (files.empty()) {
    std::cerr << "No .wav files found in " << args.input_dir << "\n";
    return EXIT_FAILURE;
  }

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA-capable device found.\n";
    return EXIT_FAILURE;
  }
  cudaDeviceProp device_prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));

  std::ofstream log(args.log_file);
  auto LogLine = [&](const std::string& line) {
    std::cout << line << "\n";
    log << line << "\n";
  };

  LogLine("GPU Specialization Capstone - Batch Audio Spectral Analyzer");
  LogLine(std::string("Device: ") + device_prop.name);
  LogLine("Files to process: " + std::to_string(files.size()));
  LogLine("Frame size: " + std::to_string(args.frame_size) +
          ", hop size: " + std::to_string(args.hop_size) +
          ", streams: " + std::to_string(args.num_streams));
  LogLine("---");

  std::vector<cudaStream_t> streams(args.num_streams);
  for (int i = 0; i < args.num_streams; ++i) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  const auto batch_start = std::chrono::high_resolution_clock::now();
  int64_t ok_count = 0;
  int64_t total_frames = 0;
  double total_gpu_ms = 0.0;

  for (size_t idx = 0; idx < files.size(); ++idx) {
    const int stream_idx = static_cast<int>(idx % args.num_streams);
    const std::string& input_path = files[idx];
    const std::string output_path =
        args.output_dir + "/" + BaseNameNoExt(input_path) + ".csv";

    int num_frames = 0;
    int num_bins = 0;
    const double elapsed_ms =
        ProcessSignal(input_path, output_path, args.frame_size,
                      args.hop_size, streams[stream_idx], &num_frames,
                      &num_bins);
    if (elapsed_ms < 0.0) {
      LogLine("  [SKIP] Failed to process " + input_path);
      continue;
    }

    total_gpu_ms += elapsed_ms;
    total_frames += num_frames;
    ++ok_count;
    LogLine("  [OK] " + input_path + " -> " + output_path + " (" +
            std::to_string(num_frames) + " frames x " +
            std::to_string(num_bins) + " bins, " +
            std::to_string(elapsed_ms) + " ms) [stream " +
            std::to_string(stream_idx) + "]");
  }

  for (int i = 0; i < args.num_streams; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    cudaStreamDestroy(streams[i]);
  }

  const auto batch_end = std::chrono::high_resolution_clock::now();
  const double wall_ms =
      std::chrono::duration<double, std::milli>(batch_end - batch_start)
          .count();

  LogLine("---");
  LogLine("Processed " + std::to_string(ok_count) + "/" +
           std::to_string(files.size()) + " files successfully.");
  LogLine("Total STFT frames computed: " + std::to_string(total_frames));
  LogLine("Total wall-clock time: " + std::to_string(wall_ms) + " ms");
  LogLine("Sum of per-file GPU times: " + std::to_string(total_gpu_ms) + " ms");
  if (wall_ms > 0.0) {
    LogLine("Throughput: " +
             std::to_string(total_frames / (wall_ms / 1000.0)) +
             " frames/sec");
  }

  return (ok_count == static_cast<int64_t>(files.size())) ? EXIT_SUCCESS
                                                          : EXIT_FAILURE;
}
