# #!/bin/bash

MODEL_NAME="mistral-community/pixtral-12b"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /home/workspace/VLM/Pixtral-finetune/output/test_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /home/workspace/VLM/Pixtral-finetune/output/test_merge \
    --safe-serialization