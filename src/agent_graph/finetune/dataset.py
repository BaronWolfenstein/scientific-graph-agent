"""Build supervised fine-tuning (SFT) pairs from the knowledge graph.

The "graph-aware" part: instead of fine-tuning on raw abstracts, we teach the
model to produce **KG-grounded, confidence-annotated** summaries — the exact
structured, contested-flagged output the KG's `to_context` renders. The model
internalises the ontology structure and the confidence framing rather than free
text. Pure Python, no ML dependencies — this is the CPU-testable half of §C.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SFTExample:
    instruction: str
    input: str
    output: str

    def as_dict(self) -> dict:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


_INSTRUCTION = (
    "Summarise what the peer-reviewed literature establishes about {entity}. "
    "State each relationship, its confidence, the number of supporting papers, "
    "and flag any contested claims."
)


def _entity_local_names(kg) -> List[str]:
    """Human-readable local names of every entity node in the store."""
    names = set()
    for s, p, o, _ev in kg.iter_claims():
        for node in (s, o):
            val = getattr(node, "value", str(node))
            names.add(val.rstrip("/").split("/")[-1])
    return sorted(names)


def build_sft_examples(
    kg,
    entities: Optional[List[str]] = None,
    *,
    min_confidence: float = 0.0,
    max_depth: int = 2,
    min_edges: int = 1,
) -> List[SFTExample]:
    """One SFT example per entity: instruction to summarise it, target = the KG's
    confidence-annotated context for its neighbourhood. Entities whose query
    returns fewer than `min_edges` claims are skipped (nothing to teach)."""
    if entities is None:
        entities = _entity_local_names(kg)

    examples: List[SFTExample] = []
    seen_outputs = set()
    for ent in entities:
        edges = kg.query([ent], min_confidence=min_confidence, max_depth=max_depth)
        if len(edges) < min_edges:
            continue
        output = kg.to_context(edges)
        if not output or output in seen_outputs:
            continue
        seen_outputs.add(output)
        examples.append(SFTExample(
            instruction=_INSTRUCTION.format(entity=ent),
            input="",
            output=output,
        ))
    return examples
