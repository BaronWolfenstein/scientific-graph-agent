import pyoxigraph as ox
import networkx as nx
from agent_graph.spectral.snapshots import SpectralSnapshot, write_spectral_snapshot
from agent_graph.spectral.laplacian import spectral_gap
from agent_graph.spectral.communities import detect_communities

def _two_triangles_store():
    G = nx.Graph()
    for a, b in [("a","b"),("b","c"),("a","c"),("x","y"),("y","z"),("x","z")]:
        G.add_edge(a, b, weight=1.0)
    return G

def test_snapshot_is_written_and_queryable():
    store = ox.Store()
    G = _two_triangles_store()
    snap = SpectralSnapshot(
        gap=spectral_gap(G, k=3),
        communities=detect_communities(G, seed=0),
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )
    uri = write_spectral_snapshot(store, snap)
    assert uri.startswith("http")
    # the snapshot node carries a spectral_gap[0] annotation
    SG = "http://ex/spectral/gap0"
    got = list(store.quads_for_pattern(ox.NamedNode(uri), ox.NamedNode(SG), None, None))
    assert len(got) == 1
