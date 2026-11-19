"""Offline configuration and lazy entry points for CUDA training."""
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import Field, model_validator
import yaml

from .data import image_path, read_rows
from .schema import DPORow, SFTRow, StrictModel, Text


class TrainConfig(StrictModel):
    model: Text
    revision: Text | None = None
    dataset: Text
    output_dir: Text
    stage: Literal["sft", "dpo"] = "sft"
    modality: Literal["text", "vision"] = "text"
    sft_adapter: Text | None = None
    resume_from_checkpoint: Text | None = None
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    lora_rank: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[Text] = Field(default_factory=lambda: ["q_proj", "v_proj"], min_length=1)
    learning_rate: float = Field(default=5e-5, gt=0, allow_inf_nan=False)
    epochs: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    max_length: int | None = Field(default=None, gt=0)
    beta: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def check_stage(self):
        if self.stage == "sft" and self.sft_adapter is not None:
            raise ValueError("sft_adapter applies only to DPO")
        if self.modality == "vision" and self.max_length is not None:
            raise ValueError("vision training requires max_length: null to preserve image tokens")
        return self


def load_train_config(path: Path) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def training_examples(config: TrainConfig) -> list[dict]:
    path = Path(config.dataset)
    schema = SFTRow if config.stage == "sft" else DPORow
    task = "appsec" if config.modality == "text" else "ui"
    examples = []
    for row in read_rows(path, schema):
        if row.task != task:
            continue
        if isinstance(row, SFTRow):
            messages = [message.model_dump() for message in row.messages]
            example = {"prompt": messages[:-1], "completion": messages[-1:]}
        else:
            example = row.model_dump(include={"prompt", "chosen", "rejected"})
        if row.image is not None:
            candidate = image_path(path, row.image)
            with Image.open(candidate) as image:
                image.verify()
            # Arrow's image feature decodes the file lazily for the TRL collator.
            example["images"] = [str(candidate)]
        examples.append(example)
    if not examples:
        raise ValueError(f"{path} contains no {task} examples")
    return examples


def train(config: TrainConfig) -> None:
    examples = training_examples(config)
    try:
        import torch
        from datasets import Dataset, Image as DatasetImage
        from datasets import Sequence
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor,
            AutoTokenizer, BitsAndBytesConfig, set_seed,
        )
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install qwen-seceval[train] for CUDA training") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("NF4 training requires CUDA")
    if config.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this CUDA device requires compute_dtype: float16")
    set_seed(config.seed)
    dtype = getattr(torch, config.compute_dtype)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    dataset = Dataset.from_list(examples)
    if config.modality == "vision":
        dataset = dataset.cast_column("images", Sequence(DatasetImage()))
        model_class, processing_class = AutoModelForImageTextToText, AutoProcessor
    else:
        model_class, processing_class = AutoModelForCausalLM, AutoTokenizer
    processor = processing_class.from_pretrained(config.model, revision=config.revision)
    model = model_class.from_pretrained(
        config.model, revision=config.revision, quantization_config=quantization,
        dtype=dtype, device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = False
    checkpointing_options = {"use_reentrant": False}
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs=checkpointing_options,
    )
    if config.sft_adapter:
        model = PeftModel.from_pretrained(model, config.sft_adapter, is_trainable=True)
        adapter = None
    else:
        adapter = LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=config.target_modules, task_type="CAUSAL_LM",
        )
    common = dict(
        output_dir=config.output_dir, num_train_epochs=config.epochs, seed=config.seed,
        learning_rate=config.learning_rate, per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs=checkpointing_options,
        bf16=config.compute_dtype == "bfloat16", fp16=config.compute_dtype == "float16",
        max_length=config.max_length,
    )
    if config.stage == "sft":
        trainer_class, args = SFTTrainer, SFTConfig(**common, completion_only_loss=True)
    else:
        trainer_class, args = DPOTrainer, DPOConfig(**common, beta=config.beta)
    trainer = trainer_class(
        model=model, args=args, train_dataset=dataset, processing_class=processor,
        peft_config=adapter,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(config.output_dir)


def merge_adapter(config: TrainConfig, adapter: Path, output: Path) -> None:
    """Merge a trained adapter into a full-precision base for serving and conversion."""
    try:
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError("install qwen-seceval[train] to merge adapters") from exc

    if config.modality == "vision":
        model_class, processing_class = AutoModelForImageTextToText, AutoProcessor
    else:
        model_class, processing_class = AutoModelForCausalLM, AutoTokenizer
    base = model_class.from_pretrained(config.model, revision=config.revision, dtype="float32")
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(output)
    processing_class.from_pretrained(config.model, revision=config.revision).save_pretrained(output)
