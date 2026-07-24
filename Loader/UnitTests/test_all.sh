#!/usr/bin/env bash

# File paths
GGUF_FILE="${1:-tl.gguf}"
LOT_SCRIPT="UnitTests/LOT.py"

if [ ! -f "$GGUF_FILE" ]; then
    echo "Error: File '$GGUF_FILE' not found!"
    exit 1
fi

echo "=================================================="
echo "      RUNNING FULL TENSOR SUITE TEST ON $GGUF_FILE "
echo "=================================================="

# Extract all tensor names using Python
TENSORS=$(python3 -c "
import gguf
reader = gguf.GGUFReader('$GGUF_FILE')
for t in reader.tensors:
    print(t.name)
")

TOTAL_TENSORS=$(echo "$TENSORS" | wc -l)
PASSED=0
FAILED=0
FAILED_LIST=()

echo "Found $TOTAL_TENSORS tensors. Initializing verification..."
echo "--------------------------------------------------"

# Loop through each tensor and test
for TENSOR in $TENSORS; do
    printf "Testing: %-35s ... " "$TENSOR"
    
    # Run LOT.py quietly
    OUTPUT=$(python3 "$LOT_SCRIPT" "$GGUF_FILE" "$TENSOR" 2>&1)
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[ PASSED ] ✓"
        ((PASSED++))
    else
        echo "[ FAILED ] ✗"
        ((FAILED++))
        FAILED_LIST+=("$TENSOR")
    fi
done

echo "--------------------------------------------------"
echo "                  FINAL SUMMARY                   "
echo "--------------------------------------------------"
echo "Total Tensors Tested: $TOTAL_TENSORS"
echo "Passed:               $PASSED"
echo "Failed:               $FAILED"

if [ $FAILED -gt 0 ]; then
    echo -e "\nFailed Tensors:"
    for F_TENSOR in "${FAILED_LIST[@]}"; do
        echo "  - $F_TENSOR"
    done
    exit 1
else
    echo -e "\n PERFECT RUN! ALL $TOTAL_TENSORS TENSORS PASSED ORACLE TEST! "
    exit 0
fi