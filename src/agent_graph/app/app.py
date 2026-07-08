"""Gradio UI over the analysis layer — a thin wrapper so all logic stays in
`analysis.py` (which is unit-tested). Launch: `python -m agent_graph.app`.

Renders, for the current knowledge graph: a claims table (textual / structural /
combined confidence), a spectral summary (modularity gap + top bridge entities),
community assignments, and a community-colored network plot.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless — safe for servers and tests
import matplotlib.pyplot as plt
import networkx as nx

from agent_graph.app.analysis import AnalysisResult, analyze_store, seed_example_store
from agent_graph.spectral import build_entity_graph

_CLAIM_HEADERS = ["subject", "relation", "object", "textual", "structural",
                  "combined", "papers", "contested"]


def _short(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def claims_rows(result: AnalysisResult):
    return [[c.subject, c.relation, c.object, c.textual, c.structural,
             c.combined, c.support, "⚠" if c.contested else ""]
            for c in result.claims]


def spectral_summary(result: AnalysisResult) -> str:
    n_comm = len(set(result.communities.values())) if result.communities else 0
    gap2 = result.spectral_gap[1] if len(result.spectral_gap) > 1 else float("nan")
    bridges = ", ".join(f"{_short(u)} ({v:.2f})" for u, v in result.top_bridges[:3])
    return (
        f"**Graph:** {result.n_nodes} entities, {result.n_edges} relations, "
        f"**{n_comm} communities**\n\n"
        f"**Algebraic connectivity (λ₂):** {gap2:.3f} — lower ⇒ more modular "
        f"(clearer community split)\n\n"
        f"**Top bridge entities:** {bridges or '—'}"
    )


def network_figure(kg, result: AnalysisResult):
    G = build_entity_graph(kg.store)
    fig, ax = plt.subplots(figsize=(7, 5))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center")
        ax.axis("off")
        return fig
    pos = nx.spring_layout(G, seed=0, weight="weight")
    comms = result.communities
    colors = [comms.get(n, -1) for n in G.nodes()]
    sizes = [300 + 120 * G.degree(n) for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           cmap=plt.cm.tab10, vmin=0, vmax=9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: _short(n) for n in G.nodes()}, font_size=8)
    ax.set_title("Claim graph — nodes colored by community, sized by degree")
    ax.axis("off")
    fig.tight_layout()
    return fig


def build_demo(store_factory=seed_example_store):
    """Construct the Gradio Blocks app. `store_factory` returns a KnowledgeGraph;
    the default seeds the offline example so the app needs no API key."""
    import gradio as gr

    with gr.Blocks(title="Scientific Graph — structure & confidence") as demo:
        gr.Markdown(
            "# Scientific claim graph — structure & confidence\n"
            "Extracts ontology-constrained claims, scores each with the confidence "
            "toolkit (textual × structural), and surfaces the spectral structure "
            "(communities, modularity, bridge entities)."
        )
        with gr.Row():
            query = gr.Textbox(
                label="Entities to focus on (space-separated; blank = whole graph)",
                placeholder="e.g. Carvedilol CML",
            )
            btn = gr.Button("Analyze", variant="primary")
        summary = gr.Markdown()
        plot = gr.Plot()
        table = gr.Dataframe(headers=_CLAIM_HEADERS, label="Claims by combined confidence")

        def run(q):
            kg = store_factory()
            entities = [t for t in q.split()] if q.strip() else None
            result = analyze_store(kg, entities)
            return spectral_summary(result), network_figure(kg, result), claims_rows(result)

        btn.click(run, inputs=query, outputs=[summary, plot, table])
        demo.load(run, inputs=query, outputs=[summary, plot, table])  # populate on open
    return demo


def main():
    build_demo().launch()


if __name__ == "__main__":
    main()
