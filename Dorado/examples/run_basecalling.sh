#!/bin/bash
#
# Dorado GPU Basecalling Example Script
# Author: Soumith Paritala
# Version: Updated for Dorado v1.3.1
# Purpose: Demonstrate Dorado basecalling with GPU acceleration
#

# Exit on any error
set -e

echo "============================================"
echo "  DORADO v1.3.1 GPU BASECALLING DEMO"
echo "============================================"
echo ""

# ============================================
# STEP 1: CHECK GPU AVAILABILITY
# ============================================
echo "Step 1: Checking GPU availability..."
echo "-------------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "⚠ WARNING: nvidia-smi not found"
    echo "GPU may not be available or NVIDIA drivers not installed"
    echo "This script requires an NVIDIA GPU with CUDA support"
    exit 1
fi

# ============================================
# STEP 2: CONFIGURATION
# ============================================
echo "Step 2: Configuration"
echo "-------------------------------------------"

# Model selection (R10.4.1 chemistry, 5kHz sampling, fast model)
MODEL="dna_r10.4.1_e8.2_400bps_fast@v5.0.0"
echo "Model: $MODEL"
echo "  Chemistry: R10.4.1"
echo "  Sampling: 5kHz (400 bases/second)"
echo "  Accuracy: Fast (~99.0%)"

# Input file
INPUT="dna_r10.4.1_e8.2_400bps_5khz-FLO_PRO114M-SQK_LSK114_XL-5000.pod5"
echo "Input: $INPUT"

# Output directory
OUTPUT_DIR="results"
OUTPUT_FILE="${OUTPUT_DIR}/basecalls.fastq"
echo "Output: $OUTPUT_FILE"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# ============================================
# STEP 3: DOWNLOAD TEST DATA (if needed)
# ============================================
echo "Step 3: Ensuring test data is available..."
echo "-------------------------------------------"

if [ ! -f "$INPUT" ]; then
    echo "Downloading test POD5 file..."
    wget -q https://raw.githubusercontent.com/nanoporetech/dorado/release-v1.3/tests/data/pod5/dna_r10.4.1_e8.2_400bps_5khz/dna_r10.4.1_e8.2_400bps_5khz-FLO_PRO114M-SQK_LSK114_XL-5000.pod5
    echo "✓ Test data downloaded"
else
    echo "✓ Test data already present"
fi
echo ""

# ============================================
# STEP 4: MODEL DOWNLOAD
# ============================================
echo "Step 4: Ensuring model is available..."
echo "-------------------------------------------"

MODEL_CACHE="$HOME/.cache/dorado/models/$MODEL"

if [ -d "$MODEL_CACHE" ]; then
    echo "✓ Model already downloaded: $MODEL"
else
    echo "Downloading model: $MODEL"
    echo "(This may take 2-5 minutes depending on connection speed...)"
    dorado download --model "$MODEL"
    echo "✓ Model download complete"
fi
echo ""

# ============================================
# STEP 5: RUN BASECALLING
# ============================================
echo "Step 5: Running basecalling..."
echo "-------------------------------------------"
echo "Converting raw electrical signals → DNA sequences"
echo "Using GPU acceleration (cuda:0)"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run basecalling with GPU
# Note: Using current directory as input since POD5 file is here
dorado basecaller \
    "$MODEL" \
    . \
    --device cuda:0 \
    --emit-fastq \
    > "$OUTPUT_FILE"

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "✓ Basecalling complete!"
echo "Processing time: ${ELAPSED} seconds"
echo ""

# ============================================
# STEP 6: RESULTS SUMMARY
# ============================================
echo "Step 6: Results Summary"
echo "-------------------------------------------"

if [ -f "$OUTPUT_FILE" ]; then
    # Count total reads
    READ_COUNT=$(grep -c '^@' "$OUTPUT_FILE")
    echo "Total reads generated: $READ_COUNT"
    
    # File size
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "Output file size: $FILE_SIZE"
    
    # Calculate average read length
    AVG_LENGTH=$(awk 'NR%4==2{sum+=length($0); count++} END{if(count>0) print int(sum/count); else print 0}' "$OUTPUT_FILE")
    echo "Average read length: ${AVG_LENGTH} bases"
    
    # Calculate total bases
    TOTAL_BASES=$((READ_COUNT * AVG_LENGTH))
    echo "Total bases sequenced: ${TOTAL_BASES}"
    
    # Throughput
    if [ $ELAPSED -gt 0 ]; then
        THROUGHPUT=$((TOTAL_BASES / ELAPSED))
        echo "Throughput: ${THROUGHPUT} bases/second"
    fi
    
    echo ""
    echo "Output location: $OUTPUT_FILE"
    echo ""
    
    # Show first read as example
    echo "First DNA sequence (example):"
    echo "-------------------------------------------"
    head -n 4 "$OUTPUT_FILE"
else
    echo "⚠ ERROR: Output file not created"
    exit 1
fi

echo ""
echo "============================================"
echo "  BASECALLING SUCCESSFUL"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. View full results: cat $OUTPUT_FILE"
echo "  2. Get detailed stats: seqkit stats $OUTPUT_FILE"
echo "  3. Filter by quality: seqkit seq -Q 10 $OUTPUT_FILE > filtered.fastq"
echo ""