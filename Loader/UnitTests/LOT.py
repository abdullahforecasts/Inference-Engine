#!/usr/bin/env python3
import sys
import subprocess
import re
import numpy as np

try:
    import gguf
except ImportError:
    print("Error: 'gguf' package is not installed. Install via: pip install gguf")
    sys.exit(1)


def dequantize_q8_0(block_data, count=20):
    """
    Dequantizes the first 'count' values from a Q8_0 byte buffer.
    Q8_0 layout per 32-element block:
      - scale (d): 2 bytes (FP16)
      - qs: 32 bytes (int8)
      Total block size: 34 bytes
    """
    weights = []
    blocks_needed = (count + 31) // 32

    for b in range(blocks_needed):
        offset = b * 34
        # 1. Read FP16 scale factor d
        scale_bytes = block_data[offset : offset + 2]
        d = np.frombuffer(scale_bytes, dtype=np.float16)[0].astype(np.float32)

        # 2. Read 32 int8 quantized weights
        qs_bytes = block_data[offset + 2 : offset + 34]
        qs = np.frombuffer(qs_bytes, dtype=np.int8)

        # 3. Dequantize: w = qs * d
        for val in qs:
            weights.append(float(val * d))
            if len(weights) == count:
                break

    return weights


def extract_python_weights(gguf_path, tensor_name, count=20):
    reader = gguf.GGUFReader(gguf_path)
    
    # Locate tensor
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break

    if tensor is None:
        print(f"[Python] Error: Tensor '{tensor_name}' not found in {gguf_path}")
        sys.exit(1)

    print(f"[Python] Target Tensor: {tensor.name}")
    print(f"[Python] Tensor Type: {tensor.tensor_type.name}")
    print(f"[Python] Tensor Shape: {tensor.shape}")

    raw_bytes = tensor.data.tobytes()

    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        weights = np.frombuffer(raw_bytes[: count * 4], dtype=np.float32).tolist()
    elif tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
        weights = dequantize_q8_0(raw_bytes, count=count)
    else:
        print(f"[Python] Warning: Type {tensor.tensor_type.name} dequantization not yet implemented.")
        sys.exit(1)

    return weights, tensor.tensor_type.name


def run_c_loader(loader_path, gguf_path, tensor_name):
    cmd = [loader_path, gguf_path, tensor_name]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[C Loader] Error executing command: {' '.join(cmd)}")
        print(result.stderr)
        sys.exit(1)

    c_weights = []
    # Match patterns like: Weight[0] (Dequantized) = -0.00195312 or Weight[0] = -0.001953
    pattern = re.compile(r"Weight\[\d+\]\s*(?:\(Dequantized\))?\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    for line in result.stdout.splitlines():
        match = pattern.search(line)
        if match:
            c_weights.append(float(match.group(1)))

    return c_weights


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 UnitTests/LOT.py <model.gguf> <tensor_name>")
        print("Example: python3 UnitTests/LOT.py tl.gguf blk.0.attn_q.weight")
        sys.exit(1)

    gguf_path = sys.argv[1]
    tensor_name = sys.argv[2]
    loader_path = "./loader"

    print("==================================================")
    print("               ORACLE TEST INITIALIZED            ")
    print("==================================================")

    # 1. Extract Python weights
    py_weights, tensor_type = extract_python_weights(gguf_path, tensor_name, count=20)

    # 2. Extract C Loader weights
    c_weights = run_c_loader(loader_path, gguf_path, tensor_name)

    if len(c_weights) < 20:
        print(f"\n[Error] C Loader returned {len(c_weights)} weights, expected 20.")
        sys.exit(1)

    c_weights = c_weights[:20]

    # 3. Print Comparison Table
    print("\n--------------------------------------------------")
    print(f"{'Idx':<5} | {'Python (Ground Truth)':<22} | {'C Loader Output':<22} | {'Match'}")
    print("--------------------------------------------------")

    all_matched = True
    tolerance = 1e-5  # Float precision tolerance

    for i in range(20):
        py_val = py_weights[i]
        c_val = c_weights[i]
        diff = abs(py_val - c_val)
        is_match = diff < tolerance

        if not is_match:
            all_matched = False

        status = "✓" if is_match else f"✗ (Diff: {diff:.2e})"
        print(f"{i:<5} | {py_val:<22.8f} | {c_val:<22.8f} | {status}")

    print("--------------------------------------------------")

    if all_matched:
        print(" SUCCESS: All 20 weights match perfectly within tolerance!")
    else:
        print(" FAILURE: Weight mismatch detected between Python Oracle and C implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()