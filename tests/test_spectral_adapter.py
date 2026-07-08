import networkx as nx
import pyoxigraph as ox
from agent_graph.spectral.graph_adapter import build_entity_graph


def _triple(s, p, o):
    return ox.Quad(ox.NamedNode(s), ox.NamedNode(p), ox.NamedNode(o))


def test_build_entity_graph_nodes_and_edges():
    store = ox.Store()
    E = "http://ex/entity/"; P = "http://ex/claim/affects"
    store.add(_triple(E + "A", P, E + "B"))
    store.add(_triple(E + "B", P, E + "C"))
    store.add(_triple(E + "A", P, E + "C"))
    G = build_entity_graph(store)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3
    assert G.has_edge(E + "A", E + "B")
