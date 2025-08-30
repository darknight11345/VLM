#!/bin/bash
#!/usr/bin/env bash
#SBATCH --partition=gpu_a100_il    ###dev_gpu_h100       ##gpu_h100_il        # GPU-Dev-Queue (H100)
#SBATCH --gres=gpu:1                    # 1 × H100
#SBATCH --cpus-per-task=8               # 8 CPU-Kerne
#SBATCH --mem=64G                       # 64 GB RAM
#SBATCH -t 02:50:00                     ###00:20:00                     ##02:50:00                     # 20 Min Testlauf
#SBATCH -J pixtral_gpu_dev_test_02082025              # Job-Name
#SBATCH --output=/pfs/work9/workspace/scratch/ul_swv79-pixtral/Pixtral-Finetune/inference_evaluation_scripts_original_model/slurm_log/pixtral_gpu.%j.out


#MODEL_NAME="mistral-community/pixtral-12b"
#export WS_MODEL=$(ws_find pixtral) for testing
export WS_MODEL="/pfs/work9/workspace/scratch/ul_swv79-pixtral"
export RESULTS_DIR="$WS_MODEL/Pixtral-Finetune/output/inference_evaluation_results_original_model"
mkdir -p "$RESULTS_DIR/slurm_log"
echo "SLURM GPU devices: $CUDA_VISIBLE_DEVICES"


## 1) load module

#module load devel/miniforge/24.11.0-python-3.12
#export PATH="$WS_MODEL/conda/pixtral_orginal_model)/bin:$PATH"

## 2) CUDA-Modul -------------------------------------------------------
module load devel/cuda/12.8              # laut `module avail cuda`



#export PYTHONPATH=$WS_MODEL/Pixtral-Finetune/src${PYTHONPATH:+:$PYTHONPATH}


## 4) Inferenz ---------------------------------------------------------
python "$WS_MODEL/Pixtral-Finetune/inference_evaluation_scripts_original_model/all_experiments_mistral.py" \
       --model_path "$WS_MODEL/pixtral-12b_original_model" \
       --data_path  "$WS_MODEL/Dataset/Validation_dataset/image_dots" \
       --output_dir "$RESULTS_DIR" \
       --batch_size 4