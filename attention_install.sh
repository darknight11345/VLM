#!/bin/bash
#!/usr/bin/env bash
#SBATCH --partition=dev_gpu_h100        # GPU-Dev-Queue (H100)
#SBATCH --gres=gpu:1                    # 1 × H100
#SBATCH --cpus-per-task=8               # 8 CPU-Kerne
#SBATCH --mem=64G                       # 64 GB RAM
#SBATCH -t 00:20:00                     # 20 Min Testlauf
#SBATCH -J pixtral_gpu_dev_test_29072025              # Job-Name
#SBATCH --output=/pfs/work9/workspace/scratch/ul_swv79-pixtral/Pixtral-Finetune/output/slurm_log/pixtral_gpu.%j.out

#MODEL_NAME="mistral-community/pixtral-12b"
#export WS_MODEL=$(ws_find pixtral) for testing
export WS_MODEL="/pfs/work9/workspace/scratch/ul_swv79-pixtral"
export RESULTS_DIR="$WS_MODEL/Pixtral-Finetune/output"
mkdir -p "$RESULTS_DIR/slurm_log"
echo "SLURM GPU devices: $CUDA_VISIBLE_DEVICES"


## 1) load module

module load devel/miniforge/24.11.0-python-3.12
export PATH="$WS_MODEL/conda/pixtral/bin:$PATH"

## 2) CUDA-Modul -------------------------------------------------------
module load devel/cuda/12.8              # laut `module avail cuda`

## 3) Kurz-Check (ASCII-nur, ohne Umlaut) ------------------------------
python - <<'PY'
import importlib, vllm, torch
print("vllm im Pfad:", bool(importlib.util.find_spec("vllm")))
print("CUDA available:", torch.cuda.is_available())
PY
 
 pip install flash-attn==2.5.8 --no-build-isolation