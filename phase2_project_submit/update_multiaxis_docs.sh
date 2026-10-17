#!/bin/bash

echo "Updating Phase 2 Documentation and Results..."

# ==========================================
# 1. UPDATE README.md
# ==========================================
cat > README.md << 'EOF'
# Phase 2 - PINN System Identification

Identifying the stiffness and damping of a steel table from a single node's
displacement response across multiple axes (X, Y, Z), using a physics-informed neural network.

## Quick start

    source venv/bin/activate

    # Select the target axis (X, Y, or Z) by uncommenting its block in config.yaml
    # Submit Slurm jobs for training
    sbatch --job-name=phase2_X slurm/gpu.sh scripts/run_train.py --tag X_run
    sbatch --job-name=phase2_Y slurm/gpu.sh scripts/run_train.py --tag Y_run
    sbatch --job-name=phase2_Z slurm/gpu.sh scripts/run_train.py --tag Z_run

    # Evaluate and plot
    python scripts/run_evaluate.py --tag X_run
    python scripts/run_plot.py --tag X_run

## Multi-Axis Flexibility

The model is direction-agnostic. Output JSONs, CSVs, and figures are automatically routed to specific subdirectories (X, Y, Z) to prevent data overwriting between cluster jobs.
EOF

# ==========================================
# 2. UPDATE AGENTS.md
# ==========================================
[ -f AGENTS.md ] && cp AGENTS.md AGENTS.md.v2

cat > AGENTS.md << 'EOF'
# Phase 2 - PINN System Identification (Multi-Axis)

**Status: DIRECTION-AGNOSTIC ARCHITECTURE ACTIVE.** 

## What this project does
A steel table is shaken by a horizontal harmonic force in ANSYS. The displacement of one corner node is recorded across three spatial directions (X, Y, Z). A physics-informed neural network is trained to reproduce that displacement while simultaneously identifying the stiffness and damping of an equivalent single-degree-of-freedom system for the selected axis.

## Execution Rules
- **Slurm Execution:** Training is handled via `sbatch`. Do not run heavy training directly on the login node.
- **Evaluation:** Run `update_multiaxis_docs.sh` only AFTER verifying via `squeue` that all Slurm GPU jobs (X, Y, Z) have completed.
EOF

echo "Architectural Memory Updated."
echo "Scanning for Phase 2 numerical results to update RESULTS.md..."

# ==========================================
# 3. DYNAMICALLY SCAN AND UPDATE RESULTS.md
# ==========================================

touch RESULTS.md
sed -i.bak '/## Phase 2 - Multi-Axis Evaluation Results/,$d' RESULTS.md
echo -e "\n## Phase 2 - Multi-Axis Evaluation Results\n" >> RESULTS.md

RESULTS_FOUND=0

# Loop specifically through your Phase 2 run directories X, Y, Z
for tag in X Y Z; do
    axis_dir="outputs/phase2/$tag"
    
    if [ -d "$axis_dir" ]; then
        for comp_file in "$axis_dir"/comparison_*.txt; do
            if [ -f "$comp_file" ]; then
                echo "### Axis: $tag" >> RESULTS.md
                cat "$comp_file" >> RESULTS.md
                echo -e "\n---\n" >> RESULTS.md
                RESULTS_FOUND=1
            fi
        done
    else
        echo "Directory $axis_dir not found yet. Skipping..."
    fi
done

if [ $RESULTS_FOUND -eq 1 ]; then
    echo "Phase 2 results successfully extracted and appended to RESULTS.md."
else
    echo "WARNING: No Phase 2 output results found. Ensure your sbatch jobs finished and run_evaluate.py was executed."
fi