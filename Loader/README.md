# GGUF Tensor Loader and Dequantizer

## Technical Overview

1. File Mapping: Memory-maps (mmap) the GGUF binary file directly into virtual address space for fast zero-copy read access.
2. File Structure: Parses GGUF layout consisting of magic header bytes, version number, metadata key-value pairs, tensor headers, and raw data buffers.
3. Header Metadata: Reads global model attributes including tensor counts, shape dimensions, data types, and metadata blocks.
4. Offset Calculation: Computes exact binary file offsets for each tensor based on header sizes and data alignment.
5. Alignment and Padding: Handles default 32-byte alignment padding required between metadata headers and raw binary payloads.
6. Weight Fetching: Directs pointers to tensor byte ranges using computed offsets without allocating redundant memory.
7. Q8_0 Dequantization: Converts quantized Q8_0 data blocks into standard float32 values.
8. FP16 Scale Conversion: Reads the 16-bit float scale factor (d) per block and converts it to standard 32-bit float (FP32) via bitcasting.
9. Weight Calculation: Multiplies the converted FP32 scale factor (d) by each of the next 32 int8_t quantized integers (qs) in the block.
10. Weight Output: Produces dequantized floating-point array using the formula: weight[i] = qs[i] * fp16_to_fp32(d).

---

## Makefile Configuration

Place the Makefile in your directory with the following contents:

Compiler: gcc
Flags: -Wall -Wextra -O3 -std=c11
Includes: -Iinc/loader
Target Executable: loader
Sources: src/loader/load_safetensors.c

---

## Build and Run Instructions

### 1. Compilation

To compile the C loader executable:
make

To clean compiled object files and binaries:
make clean

### 2. Individual Tensor Testing

To run the compiled C loader directly on a target tensor:
./loader tl.gguf token_embd.weight

To run the Python oracle test verification on an individual tensor:
python3 UnitTests/LOT.py tl.gguf token_embd.weight

### 3. Full Test Suite Execution

To execute the automated test script across all model tensors:
chmod +x UnitTests/test_all.sh
./UnitTests/test_all.sh tl.gguf