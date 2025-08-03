#!/bin/bash
#!/usr/bin/env bash
#SBATCH --partition=gpu_h100_il        # GPU-Dev-Queue (H100)
#SBATCH --gres=gpu:1                    # 1 × H100
#SBATCH --cpus-per-task=8               # 8 CPU-Kerne
#SBATCH --mem=64G                       # 64 GB RAM
#SBATCH -t 02:50:00                     # 20 Min Testlauf
#SBATCH -J pixtral_gpu_dev_test_02082025              # Job-Name
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
#import importlib, vllm, torch
import importlib, torch
#print("vllm im Pfad:", bool(importlib.util.find_spec("vllm")))
print("CUDA available:", torch.cuda.is_available())
PY
 

#IMAGE_FOLDER = "/pfs/work9/workspace/scratch/ul_swv79-pixtral/Dataset/Training_dataset/image_dots/"
#QA = "/pfs/work9/workspace/scratch/ul_swv79-pixtral/Dataset/Training_dataset/qa_dots.json/"
#Train.py = "/pfs/work9/workspace/scratch/ul_swv79-pixtral/Pixtral-Finetune/src/training/train.py"

# Pixtral does not support flash-attnetion2 yet.
# The multi-modal projector isn't included in the lora module, you should set tune_img_projector to True.
# Also it could be better for setting the lr for img_procjector.

export PYTHONPATH=$WS_MODEL/Pixtral-Finetune/src${PYTHONPATH:+:$PYTHONPATH}

#### tharani changed per_device_train_batch_size value to 1 from 2 

deepspeed $WS_MODEL/Pixtral-Finetune/src/training/train.py \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head','embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed "$WS_MODEL/Pixtral-Finetune/scripts/zero3.json" \
    --model_id "$WS_MODEL/pixtral-12b" \
    --data_path "$WS_MODEL/Dataset/Training_dataset" \
    --image_folder "$WS_MODEL/Dataset/Training_dataset/image_dots" \
    --qa_json_path "$WS_MODEL/Dataset/Training_dataset/qa_dots.json" \
    --disable_flash_attn2 True \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --bf16 True \
    --output_dir "$WS_MODEL/output" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \