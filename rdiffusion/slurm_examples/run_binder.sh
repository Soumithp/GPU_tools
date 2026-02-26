#!/bin/bash
#SBATCH --job-name=rfdiff_binder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=rfdiff_binder_%j.log

# ============================================================
# RFdiffusion Binder Design - SLURM Submission Script
# ============================================================
# Usage: sbatch run_binder.sh <target.pdb> <chain> <hotspots>
# Example: sbatch run_binder.sh my_target.pdb A "A30,A33,A34"
# ============================================================

# Parse arguments
TARGET_PDB=${1:-"target.pdb"}
TARGET_CHAIN=${2:-"A"}
HOTSPOTS=${3:-"A30,A33,A34"}
NUM_DESIGNS=${4:-100}
BINDER_MIN=${5:-70}
BINDER_MAX=${6:-120}

# Path to container
SIF="rfdiffusion.sif"

echo "============================================================"
echo "RFdiffusion Binder Design"
echo "============================================================"
echo "Target PDB:    ${TARGET_PDB}"
echo "Target Chain:  ${TARGET_CHAIN}"
echo "Hotspots:      ${HOTSPOTS}"
echo "Num Designs:   ${NUM_DESIGNS}"
echo "Binder Length:  ${BINDER_MIN}-${BINDER_MAX}"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

# Create output directory
OUTPUT_DIR="binder_designs_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}

# Get target chain residue range (assumes sequential numbering)
# For production, inspect the PDB to get exact range
CHAIN_START=1
CHAIN_END=$(grep "^ATOM" ${TARGET_PDB} | grep " ${TARGET_CHAIN} " | tail -1 | awk '{print $6}')

echo "Chain ${TARGET_CHAIN} range: ${CHAIN_START}-${CHAIN_END}"

# Run RFdiffusion binder design
singularity exec --nv ${SIF} python \
    /opt/RFdiffusion/scripts/run_inference.py \
    inference.output_prefix=${OUTPUT_DIR}/binder \
    inference.input_pdb=${TARGET_PDB} \
    inference.num_designs=${NUM_DESIGNS} \
    "contigmap.contigs=[${TARGET_CHAIN}${CHAIN_START}-${CHAIN_END}/0 ${BINDER_MIN}-${BINDER_MAX}]" \
    "ppi.hotspot_res=[${HOTSPOTS}]" \
    inference.ckpt_override_path=/opt/RFdiffusion/models/Complex_base_ckpt.pt

echo ""
echo "============================================================"
echo "COMPLETE: $(ls ${OUTPUT_DIR}/*.pdb 2>/dev/null | wc -l) designs generated"
echo "Output: ${OUTPUT_DIR}/"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run ProteinMPNN on generated backbones"
echo "  2. Validate with AlphaFold2/ColabFold"
echo "  3. Run GROMACS MD for stability check"