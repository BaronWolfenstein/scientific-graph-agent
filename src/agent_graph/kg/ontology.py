"""Frozen ontology for the scientific-literature knowledge graph.

Declared ONCE here. The extractor (extract.py) is constrained to these via
Pydantic Literals, so the LLM cannot mint off-ontology predicates. This is the
single anti-sprawl gate.
"""
import re
from typing import Literal, get_args

KG = "https://kg.local/"
RDF_REIFIES = "http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies"

EntityType = Literal[
    "drug", "disease", "gene", "biomarker", "method", "population", "paper", "author"
]
ENTITY_TYPES = get_args(EntityType)

# Claim predicates + bibliographic `wrote`. `cites` / `affiliated_with` are RESERVED.
RelationType = Literal[
    "treats", "causes", "associated_with", "inhibits", "increases_risk_of",
    "decreases_risk_of", "measured_by", "subtype_of", "studied_in", "wrote",
]
RELATION_TYPES = get_args(RelationType)

# FUNCTIONAL is intentionally empty (nothing is single-valued in a literature graph)
# but the mechanism is retained for the clinical extension. MUTEX flags genuine
# logical contradiction (flag-not-reject).
FUNCTIONAL: set = set()
MUTEX: set = {("increases_risk_of", "decreases_risk_of")}


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)      # drop punctuation
    s = re.sub(r"[\s_]+", "-", s)        # spaces/underscores -> hyphen
    return s.strip("-")


def entity_uri(etype: str, name: str) -> str:
    return f"{KG}{etype}/{slugify(name)}"


def paper_uri(paper_id: str = None, pmid: str = None, arxiv_id: str = None) -> str:
    if pmid:
        return f"pmid:{pmid}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return f"paper:{slugify(paper_id or 'unknown')}"


def relation_uri(rel: str) -> str:
    return f"{KG}{rel}"


def mutex_partners(rel: str) -> list:
    out = []
    for a, b in MUTEX:
        if rel == a:
            out.append(b)
        elif rel == b:
            out.append(a)
    return out
