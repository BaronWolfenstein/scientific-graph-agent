"""Ontology-constrained triplet extraction schema + prompt.

The Literals are imported from ontology.py so the schema and the gate cannot
drift apart. Pydantic enforces them at the structured-output boundary.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field

from agent_graph.kg.ontology import EntityType, RelationType


class ScientificTriplet(BaseModel):
    subject: str
    subject_type: EntityType
    relation: RelationType
    object: str
    object_type: EntityType
    polarity: Literal["supports", "refutes"] = "supports"


class TripletExtraction(BaseModel):
    triplets: list[ScientificTriplet] = Field(default_factory=list)


class Evidence(BaseModel):
    """One paper's assertion of a claim — the parse-don't-validate boundary type.

    add_relation() coerces incoming dicts into this once, so the constrained
    pmid / bounded relevance / known polarity are guaranteed for everything
    downstream (confidence, serialization). A raw dict never flows deeper.
    """
    paper_uri: str
    pmid: Optional[str] = Field(default=None, pattern=r"^\d+$")
    paper_id: Optional[str] = None
    pub_year: Optional[int] = Field(default=None, ge=1800, le=2100)
    relevance: int = Field(ge=0, le=100)
    polarity: Literal["supports", "refutes"] = "supports"
    snippet: str = ""


EXTRACTION_PROMPT = """You extract structured scientific claims from a paper abstract.

Rules:
- Extract ONLY relationships expressible with the allowed entity types and relations.
  Anything that does not fit is dropped — do not invent relations.
- Allowed entity types: drug, disease, gene, biomarker, method, population, paper, author.
- Allowed relations: treats, causes, associated_with, inhibits, increases_risk_of,
  decreases_risk_of, measured_by, subtype_of, studied_in, wrote.
- Set polarity="refutes" when the paper reports a NULL or NEGATIVE finding for a claim
  (e.g. "no significant effect"); otherwise polarity="supports".
- Extract claims the paper makes; do not extract the question or background it cites.

Return the triplets for the abstract below."""
