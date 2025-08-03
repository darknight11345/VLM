from dataclasses import dataclass, field
from typing import Dict, List, Optional


from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="mistral-community/pixtral-12b")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_img_projector: bool = field(default=True)
    disable_flash_attn2: bool = field(default=True)

    max_seq_length: int = field(
        default=131072,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    projector_lr: Optional[float] = None
    vision_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    qa_json_path: str =  field(default=None)
    max_num_frames: int = 10
    
    def __post_init__(self):
        if self.qa_json_path is None:
            ws_model = os.environ.get("WS_MODEL", "")
            self.qa_json_path = os.path.join(ws_model, "Dataset", "Training_dataset", "qa_dots.json")
            
        if self.image_folder is None:
            ws_model = os.environ.get("WS_MODEL", "")
            self.image_folder = os.path.join(ws_model, "Dataset", "Training_dataset", "image_dots")
    
