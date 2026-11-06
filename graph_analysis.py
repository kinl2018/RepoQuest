import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

def build_graph(files: list[Path], edges: list[tuple[Path, Path]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(files)
    graph.add_edges_from(edges)
    return graph

def find_cycles(graph: nx.DiGraph) -> list[list]:
    return list(nx.simple_cycles(graph))

def save_graph(graph: nx.DiGraph, output: Path, root: Path | None = None):
    plt.figure(figsize=(12, 8))
    labels = {}
    for node in graph.nodes:
        if root is not None and isinstance(node, Path):
            labels[node] = str(node.relative_to(root))
        else:
            labels[node] = str(node)

    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos=pos, with_labels=False, arrows=True, node_size=500, arrowsize=18)
    label_pos = {node: (x, y + 0.07) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(graph, pos=label_pos, labels=labels, font_size=9)
    plt.savefig(output, bbox_inches="tight")
    plt.close()