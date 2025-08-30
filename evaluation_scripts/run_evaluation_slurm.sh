#!/bin/bash
#!/usr/bin/env bash
#SBATCH --partition=dev_gpu_h100        ##gpu_a100_il    ###dev_gpu_h100       ##gpu_h100_il        # GPU-Dev-Queue (H100)
#SBATCH --gres=gpu:1                    # 1 × H100
#SBATCH --cpus-per-task=8               # 8 CPU-Kerne
#SBATCH --mem=64G                       # 64 GB RAM
#SBATCH -t 00:20:00                     ###00:20:00                     ##02:50:00                     # 20 Min Testlauf
#SBATCH -J pixtral_gpu_dev_test_02082025              # Job-Name
#SBATCH --output=/pfs/work9/workspace/scratch/ul_swv79-pixtral/Pixtral-Finetune/evaluation_scripts/slurm_log/pixtral_gpu.%j.out


#MODEL_NAME="mistral-community/pixtral-12b"
#export WS_MODEL=$(ws_find pixtral) for testing
export WS_MODEL="/pfs/work9/workspace/scratch/ul_swv79-pixtral"
echo "SLURM GPU devices: $CUDA_VISIBLE_DEVICES"


## 1) load module

module load devel/miniforge/24.11.0-python-3.12
export PATH="$WS_MODEL/conda/pixtral_inference/bin:$PATH"

## 2) CUDA-Modul -------------------------------------------------------
module load devel/cuda/12.8              # laut `module avail cuda`



export PYTHONPATH=$WS_MODEL/Pixtral-Finetune/src${PYTHONPATH:+:$PYTHONPATH}


## 4) Inferenz ---------------------------------------------------------
python "$WS_MODEL/Pixtral-Finetune/evaluation_scripts/1_calculate_results_image.py"