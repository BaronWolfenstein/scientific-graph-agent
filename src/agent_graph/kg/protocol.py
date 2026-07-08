"""Backend-agnostic KnowledgeGraph interface. Nodes import only this."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class KnowledgeGraph(Protocol):
    def add_relation(self, subject: str, subject_type: str, relation: str,
                     obj: str, object_type: str, evidence):
        """Upsert a claim, fold in one evidence item, recompute confidence,
        flag MUTEX conflicts. Returns a conflict note string or None.

        `evidence` is a dict or extract.Evidence; the implementation parses it
        into a validated Evidence at the boundary (raises ValidationError on a
        malformed blob)."""
        ...

    def query(self, entities, relation_hints=None, max_depth: int = 2,
              min_confidence: float = 0.0, as_of_year: int = None) -> list:
        ...

    def to_context(self, edges: list, limit: int = 25) -> str:
        ...

    def to_dict(self) -> dict:
        ...
