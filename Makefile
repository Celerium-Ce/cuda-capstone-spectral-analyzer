# GPU Specialization Capstone Project
# Builds bin/spectral_analyzer: a batch audio STFT pipeline combining a
# hand-written CUDA windowing kernel, cuFFT, and a hand-written
# magnitude-in-decibels kernel.
#
# Requirements: CUDA Toolkit (nvcc on PATH) with cuFFT and a CUDA-capable
# GPU.
#
# Usage:
#   make                 # build bin/spectral_analyzer
#   make dataset         # (re)generate the synthetic data/input signals
#   make run             # build, then process data/input -> output
#   make clean           # remove build artifacts
#
# Override the target GPU architecture(s) if the default list below does
# not cover your device, e.g.:
#   make ARCHES="-gencode arch=compute_86,code=sm_86"

NVCC      ?= nvcc

TARGET    := bin/spectral_analyzer
SRC_DIR   := src
OBJ_DIR   := bin/obj

CPP_SRCS  := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS   := $(wildcard $(SRC_DIR)/*.cu)
OBJS      := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS)) \
             $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

# -I. lets "include/foo.h" in src/*.cpp|cu resolve from the project root
# (Google C++ Style Guide: include paths mirror the directory layout).
NVCCFLAGS := -std=c++17 -O2 -I.
LDFLAGS   := -lcufft -lcudart

# Covers Maxwell (sm_52) through Ampere (sm_86); adjust for newer/older GPUs.
ARCHES ?= -gencode arch=compute_52,code=sm_52 \
          -gencode arch=compute_60,code=sm_60 \
          -gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_75,code=sm_75 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_86,code=sm_86

.PHONY: all dirs dataset run clean

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR) bin output logs

$(TARGET): $(OBJS)
	$(NVCC) $(ARCHES) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) $(ARCHES) -c $< -o $@

dataset:
	cd data && python3 generate_signals.py

run: all
	./$(TARGET) --input data/input --output output --log logs/run.log

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
