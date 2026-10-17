# Phase 2 - PINN System Identification (Multi-Axis)

**Status: DIRECTION-AGNOSTIC ARCHITECTURE ACTIVE.** 

## What this project does
A steel table is shaken by a horizontal harmonic force in ANSYS. The displacement of one corner node is recorded across three spatial directions (X, Y, Z). A physics-informed neural network is trained to reproduce that displacement while simultaneously identifying the stiffness and damping of an equivalent single-degree-of-freedom system for the selected axis.

## Execution Rules
- **Slurm Execution:** Training is handled via `sbatch`. Do not run heavy training directly on the login node.
- **Evaluation:** Run `update_multiaxis_docs.sh` only AFTER verifying via `squeue` that all Slurm GPU jobs (X, Y, Z) have completed.
