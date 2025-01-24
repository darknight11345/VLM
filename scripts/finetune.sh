#!/bin/bash

MODEL_NAME="mistral-community/pixtral-12b"

# Pixtral does not support flash-attnetion2 yet.
# It only supports batch size 1 for now. If you want to use batch size > 1, you need to modify the model code. The model dose not support various image sizes
# in the same batch. If you want to use various image sizes, you need to modify the model code.

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm True \
    --bf16 True \
    --output_dir output/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 4