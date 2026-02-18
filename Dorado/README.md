# Dorado GPU Basecaller v1.3.1

Oxford Nanopore Technologies' production basecaller with GPU acceleration for converting raw electrical signals to DNA sequences.

## Overview

**Dorado v1.3.1** is the latest stable release supporting R10.4.1 chemistry (5kHz sampling) and updated models with improved accuracy.

### What It Does
```
Input:  POD5 files (raw electrical signals at 5kHz)
        ↓
Process: Deep learning inference on GPU with v5.0.0 models
        ↓
Output: FASTQ files (DNA sequences with quality scores)
```

### Performance (v1.3.1)

- **Speed:** 1-3 million bases per minute (with GPU)
- **Accuracy:** 
  - Fast: ~99.0% (dna_r10.4.1_e8.2_400bps_fast@v5.0.0)
  - HAC: ~99.5% (dna_r10.4.1_e8.2_400bps_hac@v5.0.0)
  - SUP: ~99.9% (dna_r10.4.1_e8.2_400bps_sup@v5.0.0)
- **Chemistry:** R10.4.1 (5kHz sampling rate)

### What's New in v1.3.1

- Support for R10.4.1 5kHz chemistry
- Updated v5.0.0 model series
- Improved basecalling accuracy
- Better GPU memory management
- Enhanced POD5 file handling

---

## System Requirements

### Hardware

- **GPU:** NVIDIA with CUDA compute capability ≥ 7.0
  - Tested on: Tesla T4 (16GB), V100 (32GB), A100 (40GB/80GB)
  - Consumer GPUs: RTX 3060 Ti (8GB) or better
- **VRAM:** 
  - Minimum: 8 GB (fast models)
  - Recommended: 16 GB (HAC models)
  - High-end: 24+ GB (SUP models)
- **RAM:** 16+ GB system memory
- **Storage:** 
  - ~5-8 GB for model files (v5.0.0 models are larger)
  - Variable for data (1-100 GB typical)

### Software

- **OS:** Linux (Ubuntu 20.04+, tested on Ubuntu 22.04)
- **CUDA:** Version 12.0+ (v1.3.1 uses CUDA 12.x)
- **Driver:** NVIDIA driver 525.60.13+
- **Python:** 3.10-3.11 (for auxiliary tools)

---

## Installation Methods

### Option 1: Conda Environment + Binary Download
```bash
# Create environment
conda env create -f conda_env/dorado_env.yml

# Activate environment
conda activate dorado_gpu

# Download Dorado v1.3.1 binary
wget https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.3.1-linux-x64.tar.gz
tar -xzf dorado-1.3.1-linux-x64.tar.gz

# Add to PATH
export PATH=$(pwd)/dorado-1.3.1-linux-x64/bin:$PATH

# Verify installation
dorado --version
# Expected output: dorado 1.3.1+...

# Test GPU access
nvidia-smi
```

### Option 2: Singularity Container (Production Deployment)
```bash
# Build container (requires Linux with sudo or --fakeroot)
sudo singularity build dorado.sif containers/dorado.def

# Or using fakeroot (no sudo required)
singularity build --fakeroot dorado.sif containers/dorado.def

# Test container
singularity exec --nv dorado.sif dorado --version

# Verify GPU access
singularity exec --nv dorado.sif nvidia-smi
```

**Note:** Container building requires Linux. The `.def` file is provided for deployment on HPC systems.

---

## Quick Start

### 1. Download Basecalling Model

Dorado v1.3.1 uses the v5.0.0 model series for R10.4.1 chemistry:
```bash
# Fast model (99.0% accuracy, fastest)
dorado download --model dna_r10.4.1_e8.2_400bps_fast@v5.0.0

# High accuracy (99.5%, recommended for production)
dorado download --model dna_r10.4.1_e8.2_400bps_hac@v5.0.0

# Super accurate (99.9%, best accuracy)
dorado download --model dna_r10.4.1_e8.2_400bps_sup@v5.0.0
```

**Model naming explained:**
- `dna` - DNA sequencing (not RNA)
- `r10.4.1` - Pore chemistry version
- `e8.2` - Electronics version
- `400bps` - Target bases per second
- `fast/hac/sup` - Accuracy level
- `v5.0.0` - Model version

Models are cached in: `~/.cache/dorado/models/`

### 2. Download Test Data
```bash
# R10.4.1 5kHz test file
wget https://raw.githubusercontent.com/nanoporetech/dorado/release-v1.3/tests/data/pod5/dna_r10.4.1_e8.2_400bps_5khz/dna_r10.4.1_e8.2_400bps_5khz-FLO_PRO114M-SQK_LSK114_XL-5000.pod5
```

### 3. Run Basecalling

**Basic usage:**
```bash
dorado basecaller \
    dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
    input.pod5 \
    --device cuda:0 \
    --emit-fastq \
    > output.fastq
```

**With directory of POD5 files:**
```bash
dorado basecaller \
    dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
    pod5_directory/ \
    --device cuda:0 \
    > basecalls.fastq
```

**Using container:**
```bash
singularity exec --nv dorado.sif \
    dorado basecaller \
    dna_r10.4.1_e8.2_400bps_fast@v5.0.0 \
    input.pod5 \
    --device cuda:0 \
    > output.fastq
```

### 4. Check Results
```bash
# Count reads
grep -c '^@' output.fastq

# Get statistics (requires seqkit)
seqkit stats output.fastq

# View first read
head -n 4 output.fastq
```

---

## Example Workflow

An automated example script is provided:
```bash
cd examples/
bash run_basecalling.sh
```

This script:
1. Checks GPU availability
2. Downloads test POD5 file (R10.4.1)
3. Downloads v5.0.0 fast model
4. Runs basecalling on GPU
5. Generates summary statistics
6. Shows example output

**Expected runtime:** ~30-60 seconds for test file

---

## Resource Requirements by Model (v5.0.0 series)

| Model Type | VRAM Required | Throughput | Accuracy | Use Case |
|------------|---------------|------------|----------|----------|
| **Fast** | 8-10 GB | ~3M bases/min | 99.0% | Quick QC, screening |
| **HAC** (High Accuracy) | 14-18 GB | ~1.5M bases/min | 99.5% | Production sequencing |
| **SUP** (Super Accurate) | 24-32 GB | ~500K bases/min | 99.9% | Final assembly, clinical |

*Benchmarked on NVIDIA A100 40GB GPU with R10.4.1 5kHz chemistry*

### Model Size Comparison

- v5.0.0 models are ~20% larger than v3.4 models
- Improved accuracy from enhanced training
- Better handling of difficult sequences

---

## Sample Data & Results

### Test Dataset Specifications

- **Source:** Oxford Nanopore official test data
- **File:** `dna_r10.4.1_e8.2_400bps_5khz-FLO_PRO114M-SQK_LSK114_XL-5000.pod5`
- **Chemistry:** R10.4.1 (PromethION)
- **Sampling:** 5kHz (5000 samples/second)
- **Flow cell:** FLO-PRO114M
- **Kit:** SQK-LSK114-XL

### Example Output

See `results/sample_output/basecalls.fastq` for real output from Google Colab demo.

**Generated on:**
- **Platform:** Google Colab
- **GPU:** Tesla T4 (16GB VRAM)
- **Model:** dna_r10.4.1_e8.2_400bps_fast@v5.0.0
- **Processing time:** ~10-15 seconds for test file
- **CUDA Version:** 12.2
- **Date:** February 2026

---

## Model Availability (v1.3.1)

### R10.4.1 Chemistry (5kHz) - Recommended
```bash
# Fast models
dorado download --model dna_r10.4.1_e8.2_400bps_fast@v5.0.0

# High accuracy models
dorado download --model dna_r10.4.1_e8.2_400bps_hac@v5.0.0

# Super accuracy models
dorado download --model dna_r10.4.1_e8.2_400bps_sup@v5.0.0
```

### R9.4.1 Chemistry (Legacy)

Still supported for older data:
```bash
dorado download --model dna_r9.4.1_e8_fast@v3.4
dorado download --model dna_r9.4.1_e8_hac@v3.4
dorado download --model dna_r9.4.1_e8_sup@v3.4
```

---