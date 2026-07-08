"""Training-scaffold tests. The QLoRA deps (peft/transformers/bitsandbytes) are
CUDA-only and live in the [finetune] extra, so these skip unless it's installed
(i.e. they run on the A100 box, not on CPU). The import-safety of train.py
without those deps is covered by tests importing the module elsewhere."""
import pytest


def test_format_example_instruction_response_shape():
    # format_example uses no ML deps — always runnable
    from agent_graph.finetune.train import format_example
    s = format_example({"instruction": "Summarise X", "output": "X treats Y"}, eos="</s>")
    assert "### Instruction:" in s and "### Response:" in s
    assert s.endswith("</s>")


def test_lora_and_quant_configs_build_when_deps_present():
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from agent_graph.finetune.train import build_lora_config, build_quant_config
    lora = build_lora_config(r=8)
    assert lora.r == 8 and lora.task_type == "CAUSAL_LM"
    # build_quant_config also needs bitsandbytes-aware transformers
    qc = build_quant_config()
    assert getattr(qc, "load_in_4bit", False) is True
