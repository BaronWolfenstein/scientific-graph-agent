"""Tests for the QLoRA dataset builder (CPU half of §C — no ML deps)."""
from agent_graph.app.analysis import seed_example_store
from agent_graph.finetune.dataset import build_sft_examples, SFTExample


def test_builds_examples_from_seeded_kg():
    kg = seed_example_store()
    examples = build_sft_examples(kg)
    assert examples and all(isinstance(e, SFTExample) for e in examples)
    # instruction references an entity; output is the KG's structured context
    for e in examples:
        assert e.instruction and "literature" in e.instruction.lower()
        assert "confidence" in e.output.lower()          # confidence-annotated target
        assert e.input == ""


def test_targets_are_deduplicated():
    kg = seed_example_store()
    examples = build_sft_examples(kg)
    outputs = [e.output for e in examples]
    assert len(outputs) == len(set(outputs))             # no duplicate targets


def test_min_edges_filters_isolated_entities():
    kg = seed_example_store()
    # a very high min_edges threshold should drop everything
    assert build_sft_examples(kg, min_edges=999) == []


def test_focused_entity_subset():
    kg = seed_example_store()
    examples = build_sft_examples(kg, entities=["Imatinib"])
    assert len(examples) == 1
    assert "imatinib" in examples[0].instruction.lower()


def test_as_dict_shape():
    kg = seed_example_store()
    d = build_sft_examples(kg)[0].as_dict()
    assert set(d) == {"instruction", "input", "output"}
