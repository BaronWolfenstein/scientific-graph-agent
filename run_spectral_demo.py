"""python run_spectral_demo.py — spectral layer over a toy claim graph."""
import pyoxigraph as ox
from agent_graph.spectral import (
    build_entity_graph, spectral_gap, detect_communities,
    bridging_centrality, SpectralSnapshot, write_spectral_snapshot,
)

def _seed_store():
    store = ox.Store(); E = "http://ex/e/"; P = "http://ex/claim/affects"
    edges = [("A","B"),("B","C"),("A","C"),("X","Y"),("Y","Z"),("X","Z"),("C","X")]
    for s, o in edges:
        store.add(ox.Quad(ox.NamedNode(E+s), ox.NamedNode(P), ox.NamedNode(E+o)))
    return store

def main():
    store = _seed_store()
    G = build_entity_graph(store)
    print("gap:", [round(v, 4) for v in spectral_gap(G, k=3)])
    comm = detect_communities(G, seed=0)
    print("communities:", comm)
    bc = bridging_centrality(G)
    print("top bridge:", max(bc, key=bc.get))
    uri = write_spectral_snapshot(
        store, SpectralSnapshot(spectral_gap(G, 3), comm, G.number_of_nodes(), G.number_of_edges()))
    print("snapshot:", uri)

if __name__ == "__main__":
    main()
