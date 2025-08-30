#!/bin/bash
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -J pixtral_infer
#SBATCH --output=/pfs/work9/workspace/scratch/ul_swv79-pixtral/Pixtral-Finetune/inference_scripts/slurm_log/pixtral_infer.%j.out

echo "SLURM GPU devices: $CUDA_VISIBLE_DEVICES"


## 1) load module

module load devel/miniforge/24.11.0-python-3.12
export PATH="$WS_MODEL/conda/pixtral/bin:$PATH"

## 2) CUDA-Modul -------------------------------------------------------
module load devel/cuda/12.8              # laut `module avail cuda`



export PYTHONPATH=$WS_MODEL/Pixtral-Finetune/src${PYTHONPATH:+:$PYTHONPATH}
# Run inference script
python /pfs/work9/workspace/scratch/ul_swv79-pixtral/Pixtral-Finetune/inference_scripts/test_tokenizer.py