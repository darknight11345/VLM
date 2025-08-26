# Fine-tuning Pixtral

This repository contains a script for training Trnasformers compatible [Pixtral-12b](https://huggingface.co/mistral-community/pixtral-12b).<br>

However the model only supports **batch size=1**. So it could take a long time to fine tune.

## Supported Features

- Deepspeed
- LoRA/QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image and video training
- Training optimized with liger kernel

## Installation

Install the required packages using `environment.yaml`.

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
module load devel/miniforge/24.11.0-python-3.12
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pixtral
pip install flash-attn==2.5.8 --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.

## Model Setup
  - We are now ready to download the Pixtral-12B model from Hugging Face.<br>
  Login to Hugging Face:
  ```bash
  huggingface-cli login
  ```
  Create an access token at:<br>
  Hugging Face Profile → Access Tokens → Create New Token → [Read Permission] → Create <br>
  Paste this token when prompted.<br>
  
  Download the modeL: <br>
  ```bash
  python - <<'PY'
  from huggingface_hub import snapshot_download
  import os
  
  os.environ["HF_HOME"] = os.getenv("WS_MODEL") + "/.hf_cache"
  
  snapshot_download(
      repo_id="mistralai/Pixtral-12B-2409",
      allow_patterns=["*.safetensors","*.json","*.model"],
      local_dir=os.path.join(os.getenv("WS_MODEL"), "pixtral-12b"),
      local_dir_use_symlinks=False,
  )
  PY
  ```

## Code Modifications

  - Set the path of your base folder (WS_MODEL) where you store the model, scripts, and data.<br>
  
  Update the following files accordingly:<br>
  
  Params.py
  ```python
      qa_json_path = "/path/to/qa.json"
      image_folder = "/path/to/images"
  ```
  Finetune_lora_vision.sh
    - Update training arguments and artifact paths:
      WS_MODEL → your model base path
      RESULTS_DIR → where results/logs will be saved
      deepspeed, model_id, data_path, qa_json_path, image_folder, output_dir
    - Adjust SLURM parameters (device, partition, timelimit) as needed.

<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Pixtral model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--tune_merger` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--merger_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--max_num_frames` (int): Maxmimum frames for video dataset (default: 10)
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--min_pixels` (int): Option for minimum input tokens.
- `--max_pixles` (int): OPtion for maximum maxmimum tokens.
- `--lora_enable` (bool): Option for enabling LoRA (default: False)
- `--vision_lora` (bool): Option for including vision_tower to the LoRA module. The `lora_enable` should be `True` to use this option. (default: False)
- `--use_dora` (bool): Option for using DoRA instead of LoRA. The `lora_enable` should be `True` to use this option. (default: False)
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

## Running Training

  - To run the training script, use the following command
  
  ```bash
  sbatch Finetune_lora_vision.sh
  ```
  Monitor jobs:
  ```python
  squeue -u $(whoami) 
  ```
  After training:
  
    - Check checkpoints in output_dir<br>
    
    Logs available at: RESULTS_DIR/slurm_log<br>
    
    - Before inference, merge checkpoint shards into a single .bin file:<br>
    ```bash
    python zero_to_fp32.py WS_MODEL/checkpoint-3902 output_dir/
    ```
## Inference
- Inference requires a separate environment (to avoid version conflicts).
  1. Create the Environment
     ```bash
     conda create -p $WS_MODEL/conda/pixtral_inference
     conda activate $WS_MODEL/conda/pixtral_inference
     ```
  2. Install Dependencies
     ```bash
     "vllm>=0.6.2" "mistral_common>=1.4.4" pillow tqdm
     ```
     Important: Compatible versions are:
        transformers==4.55.0
        vllm==0.10.0
  
  3. Run Inference pip install 
     - Update paths (model_path, data_path, output_dir, WS_MODEL, RESULTS_DIR) in inference_dots.sh
     - Run the inference:
      ```bash
      sbatch inference_dots.sh
      ```
  
    - Results will be saved in RESULTS_DIR (recommended: create a subfolder like inference_results).
    - Output files:
      - qa_dots_all_images_add_run_0.json
      - qa_dots_all_images_add_run_1.json
      - qa_dots_all_images_add_run_2.json

## Evaluation
  - Use a separate environment for evaluation:
  ```bash
  conda create -p $WS_MODEL/conda/pixtral_evaluation python=3.10
  conda activate $WS_MODEL/conda/pixtral_evaluation
  ```
  Run the evaluation:
  ```bash
  python 1_calculate_results.py
  ```
  - Update paths in 1_calculate_results.py:

    base_path → where evaluation results are stored
    
    output_path → where to save metrics


With this setup, you can train, infer, and evaluate Pixtral-12B smoothly on any cluster.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Pixtral-Finetuning,
  author = {Yuwon Lee},
  title = {Pixtral-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Pixtral-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Pixtral-12B](https://huggingface.co/mistral-community/pixtral-12b): Transformer compatible version of pixtral-12b
