"""QLoRA training entry point (Step 2 §C) — runs on the A100 box, NOT on CPU.

4-bit quantized base model + PEFT/LoRA adapter, trained on the KG-grounded SFT
pairs from `dataset.py`. Every heavy import (torch, transformers, peft,
bitsandbytes, datasets) is lazy and lives inside a function, so importing this
module (and the CPU dataset tests) needs none of them. bitsandbytes 4-bit
requires CUDA — this is the first genuinely GPU-gated piece of the portfolio.

Typical use on the box:
    from agent_graph.finetune import build_sft_examples
    from agent_graph.finetune.train import run_qlora
    from agent_graph.kg import get_knowledge_graph
    kg = ...  # populated KG
    run_qlora("Qwen/Qwen2.5-1.5B-Instruct", build_sft_examples(kg), "out/adapter")
"""
from __future__ import annotations

from typing import List, Optional


def resolve_device(device: str = "auto"):
    import torch
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_quant_config():
    """4-bit NF4 quantization (QLoRA) with bf16 compute — Tensor Cores on A100."""
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def format_example(ex, eos: str = "") -> str:
    """Instruction-format one SFT example into a single training string."""
    instr = ex["instruction"] if isinstance(ex, dict) else ex.instruction
    out = ex["output"] if isinstance(ex, dict) else ex.output
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}{eos}"


def run_qlora(
    base_model: str,
    examples: List,
    output_dir: str,
    *,
    device: str = "auto",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_len: int = 1024,
):
    """Fine-tune `base_model` with QLoRA on `examples`; save the adapter to
    `output_dir`. Requires the `[finetune]` extra + a CUDA GPU (bitsandbytes)."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments)
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    dev = resolve_device(device)
    if dev.type != "cuda":
        raise RuntimeError(
            "run_qlora needs a CUDA GPU (bitsandbytes 4-bit). Run on the A100 box; "
            "the dataset builder is the CPU-testable half."
        )

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=build_quant_config(), device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config())

    rows = [{"text": format_example(e, eos=tok.eos_token)} for e in examples]
    ds = Dataset.from_list(rows)

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        bf16=True, gradient_checkpointing=True, logging_steps=10,
        save_strategy="epoch", report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        dataset_text_field="text", max_seq_length=max_len, tokenizer=tok,
    )
    trainer.train()
    model.save_pretrained(output_dir)      # adapter only (MBs)
    tok.save_pretrained(output_dir)
    return output_dir
