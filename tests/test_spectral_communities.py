import networkx as nx
from agent_graph.spectral.communities import detect_communities


def test_louvain_finds_two_communities():
    G = nx.Graph()
    for a, b in [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y"), ("y", "z"), ("x", "z")]:
        G.add_edge(a, b, weight=1.0)
    G.add_edge("c", "x", weight=0.1)  # weak bridge
    labels = detect_communities(G, method="louvain", seed=0)
    assert len({labels[n] for n in G.nodes()}) == 2
    assert labels["a"] == labels["b"] == labels["c"]
    assert labels["x"] == labels["y"] == labels["z"]
    assert labels["a"] != labels["x"]
