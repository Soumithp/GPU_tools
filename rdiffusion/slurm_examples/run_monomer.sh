#!/bin/bash
#SBATCH --job-name=rfdiff_monomer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=rfdiff_monomer_%j.log

# ============================================================
# RFdiffusion Unconditional Monomer Generation
# ============================================================
# Usage: sbatch run_monomer.sh [num_designs] [length_min] [length_max]
# Example: sbatch run_monomer.sh 100 100 200
# ============================================================

NUM_DESIGNS=${1:-100}
LENGTH_MIN=${2:-100}
LENGTH_MAX=${3:-200}

SIF="rfdiffusion.sif"

OUTPUT_DIR="monomer_designs_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}

echo "============================================================"
echo "RFdiffusion Unconditional Monomer Generation"
echo "  Designs: ${NUM_DESIGNS}"
echo "  Length:  ${LENGTH_MIN}-${LENGTH_MAX} residues"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================================"

singularity exec --nv ${SIF} python \
    /opt/RFdiffusion/scripts/run_inference.py \
    inference.output_prefix=${OUTPUT_DIR}/monomer \
    inference.num_designs=${NUM_DESIGNS} \
    "contigmap.contigs=[${LENGTH_MIN}-${LENGTH_MAX}]"

echo ""
echo "COMPLETE: $(ls ${OUTPUT_DIR}/*.pdb 2>/dev/null | wc -l) designs generated"
echo "Output: ${OUTPUT_DIR}/"