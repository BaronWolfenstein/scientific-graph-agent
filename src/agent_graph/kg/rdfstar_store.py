"""pyoxigraph implementation of KnowledgeGraph using the RDF 1.2 reifier model.

Each claim = a base triple (s, p, o) plus a reifier blank node r with
(r, rdf:reifies, <<s p o>>). Aggregate stats and the evidence JSON hang on r.
"""
import io
import json
import pyoxigraph as ox

from agent_graph.kg import ontology as ont
from agent_graph.kg import confidence as conf
from agent_graph.kg.extract import Evidence

_REIFIES = ox.NamedNode(ont.RDF_REIFIES)
_XSD_DOUBLE = ox.NamedNode("http://www.w3.org/2001/XMLSchema#double")
_XSD_INT = ox.NamedNode("http://www.w3.org/2001/XMLSchema#integer")
_XSD_BOOL = ox.NamedNode("http://www.w3.org/2001/XMLSchema#boolean")


def _pred(local):
    return ox.NamedNode(ont.KG + local)


# annotation predicates
_EVIDENCE = _pred("evidence")
_CONF = _pred("confidence")
_CONF_LB = _pred("confidence_lb")
_ALPHA = _pred("alpha")
_BETA = _pred("beta")
_SUPPORT = _pred("support")
_REFUTE = _pred("refute")
_FIRST_YEAR = _pred("first_year")
_LAST_YEAR = _pred("last_year")
_CONTESTED = _pred("contested")
_ASSERTED_BY = _pred("asserted_by")

# predicates that are real graph edges (vs reifier metadata)
_EDGE_PREDICATES = set(ont.RELATION_TYPES)


class OxigraphKG:
    def __init__(self, store=None):
        self.store = store if store is not None else ox.Store()

    # ---- node construction ----
    def _node(self, etype: str, name: str) -> "ox.NamedNode":
        if etype == "paper":
            if name.startswith(("pmid:", "arxiv:", "paper:")):
                return ox.NamedNode(name)
            return ox.NamedNode(ont.paper_uri(paper_id=name))
        return ox.NamedNode(ont.entity_uri(etype, name))

    def _claim_triple(self, s_type, s_name, relation, o_type, o_name) -> "ox.Triple":
        return ox.Triple(self._node(s_type, s_name),
                         ox.NamedNode(ont.relation_uri(relation)),
                         self._node(o_type, o_name))

    def _reifier_for(self, claim: "ox.Triple"):
        for q in self.store.quads_for_pattern(None, _REIFIES, claim, None):
            return q.subject
        return None

    # ---- annotation read/write ----
    def _set(self, r, pred, node):
        for q in list(self.store.quads_for_pattern(r, pred, None, None)):
            self.store.remove(q)
        self.store.add(ox.Quad(r, pred, node))

    def _evidence(self, r) -> list:
        for q in self.store.quads_for_pattern(r, _EVIDENCE, None, None):
            return json.loads(q.object.value)
        return []

    @staticmethod
    def _ev_key(e: dict) -> str:
        return e.get("paper_uri") or e.get("pmid") or e.get("paper_id") or ""

    def _dedupe(self, evidence: list) -> list:
        seen, out = set(), []
        for e in evidence:
            k = self._ev_key(e)
            if k in seen:
                continue
            seen.add(k)
            out.append(e)
        return out

    def _write_annotations(self, r, evidence: list):
        evidence = self._dedupe(evidence)
        a, b = conf.beta_params(evidence)
        self._set(r, _EVIDENCE, ox.Literal(json.dumps(evidence)))
        self._set(r, _CONF, ox.Literal(repr(conf.confidence(a, b)), datatype=_XSD_DOUBLE))
        self._set(r, _CONF_LB, ox.Literal(repr(conf.confidence_lb(a, b)), datatype=_XSD_DOUBLE))
        self._set(r, _ALPHA, ox.Literal(repr(a), datatype=_XSD_DOUBLE))
        self._set(r, _BETA, ox.Literal(repr(b), datatype=_XSD_DOUBLE))
        sup = sum(1 for e in evidence if e.get("polarity", "supports") == "supports")
        ref = sum(1 for e in evidence if e.get("polarity") == "refutes")
        self._set(r, _SUPPORT, ox.Literal(str(sup), datatype=_XSD_INT))
        self._set(r, _REFUTE, ox.Literal(str(ref), datatype=_XSD_INT))
        years = [e["pub_year"] for e in evidence if e.get("pub_year")]
        if years:
            self._set(r, _FIRST_YEAR, ox.Literal(str(min(years)), datatype=_XSD_INT))
            self._set(r, _LAST_YEAR, ox.Literal(str(max(years)), datatype=_XSD_INT))
        # asserted_by provenance links (additive, deduped by URI)
        have = {q.object.value for q in self.store.quads_for_pattern(r, _ASSERTED_BY, None, None)}
        for e in evidence:
            uri = e.get("paper_uri")
            if uri and uri not in have:
                self.store.add(ox.Quad(r, _ASSERTED_BY, ox.NamedNode(uri)))
                have.add(uri)

    # ---- FUNCTIONAL is empty; mechanism retained for the clinical extension ----
    def _functional_violation(self, relation, s_node):
        if relation not in ont.FUNCTIONAL:
            return None
        objs = {q.object.value for q in self.store.quads_for_pattern(s_node, ox.NamedNode(ont.relation_uri(relation)), None, None)}
        return f"functional-violation: {s_node.value} {relation} {sorted(objs)}" if len(objs) > 1 else None

    def _flag_conflicts(self, relation, s_node, o_node):
        for other in ont.mutex_partners(relation):
            partner = ox.Triple(s_node, ox.NamedNode(ont.relation_uri(other)), o_node)
            r2 = self._reifier_for(partner)
            if r2 is not None:
                this = ox.Triple(s_node, ox.NamedNode(ont.relation_uri(relation)), o_node)
                r1 = self._reifier_for(this)
                self._set(r1, _CONTESTED, ox.Literal("true", datatype=_XSD_BOOL))
                self._set(r2, _CONTESTED, ox.Literal("true", datatype=_XSD_BOOL))
                return (f"contradiction: {s_node.value} {relation}/{other} "
                        f"{o_node.value} (both retained)")
        return self._functional_violation(relation, s_node)

    # ---- public write ----
    def add_relation(self, subject, subject_type, relation, obj, object_type, evidence):
        # parse-don't-validate boundary: coerce the blob into a typed Evidence ONCE.
        # Malformed input (bad pmid / out-of-range relevance) raises here, before
        # anything touches the store; downstream only sees validated fields.
        evidence = Evidence.model_validate(evidence).model_dump()
        s = self._node(subject_type, subject)
        o = self._node(object_type, obj)
        p = ox.NamedNode(ont.relation_uri(relation))
        claim = ox.Triple(s, p, o)
        self.store.add(ox.Quad(s, p, o))  # base fact (idempotent in oxigraph)
        r = self._reifier_for(claim)
        if r is None:
            r = ox.BlankNode()
            self.store.add(ox.Quad(r, _REIFIES, claim))
            ev_list = []
        else:
            ev_list = self._evidence(r)
        ev_list.append(evidence)
        self._write_annotations(r, ev_list)
        return self._flag_conflicts(relation, s, o)

    # ---- test/inspection helper (also used by query in Task 6) ----
    def _claim_meta(self, s_type, s_name, relation, o_type, o_name):
        claim = self._claim_triple(s_type, s_name, relation, o_type, o_name)
        r = self._reifier_for(claim)
        if r is None:
            return None
        return self._reifier_meta(r)

    def _reifier_meta(self, r) -> dict:
        m = {"confidence": 0.0, "confidence_lb": 0.0, "support": 0, "refute": 0,
             "first_year": None, "last_year": None, "contested": False}
        for q in self.store.quads_for_pattern(r, None, None, None):
            local = q.predicate.value.replace(ont.KG, "")
            v = getattr(q.object, "value", None)
            if local in ("confidence", "confidence_lb", "alpha", "beta"):
                m[local] = float(v)
            elif local in ("support", "refute", "first_year", "last_year"):
                m[local] = int(v)
            elif local == "contested":
                m[local] = (v == "true")
        return m
