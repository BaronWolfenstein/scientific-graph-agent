import networkx as nx
from agent_graph.spectral.centrality import bridging_centrality


def test_bridge_nodes_score_highest():
    G = nx.Graph()
    for a, b in [("a","b"),("b","c"),("a","c"),("x","y"),("y","z"),("x","z")]:
        G.add_edge(a, b, weight=1.0)
    G.add_edge("c", "x", weight=1.0)                 # c and x are the bridge
    bc = bridging_centrality(G)
    assert bc["c"] > bc["a"] and bc["x"] > bc["z"]
