"""Graph-aware QLoRA fine-tuning (Step 2 §C).

Two layers, split by what needs a GPU:
  - `dataset.py` — pure-Python builder of supervised fine-tuning (SFT) pairs from
    the knowledge graph (KG-grounded, confidence-annotated). Runs and is tested
    on CPU with no ML dependencies.
  - `train.py` — the QLoRA training entry point (4-bit + PEFT/LoRA). Lazy-imports
    torch/transformers/peft/bitsandbytes; runs on the A100 box, not here.
"""
from agent_graph.finetune.dataset import SFTExample, build_sft_examples

__all__ = ["SFTExample", "build_sft_examples"]
