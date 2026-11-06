from pathlib import Path
from graph_analysis import build_graph
import networkx as nx

def test_build_graph():
    a = Path("a.py")
    b = Path("b.py")
    c = Path("c.py")

    files = [a, b, c]
    edges = [(a, b)]

    graph = build_graph(files, edges)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 1
    assert graph.has_edge(a, b)
    assert not graph.has_edge(b, a)

from graph_analysis import find_cycles
def test_find_cycles():
    a, b, c = "a.py", "b.py", "c.py"

    graph = nx.DiGraph()
    graph.add_edges_from([
        (a, b),
        (b, c),
        (c, a)
    ])

    cycles = find_cycles(graph)

    assert len(cycles) == 1
    assert set(cycles[0]) == {a, b, c}

def test_find_cycles_no_cycle():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("a.py", "b.py"),
        ("b.py", "c.py"),
    ])

    cycles = find_cycles(graph)

    assert cycles == []


from graph_analysis import save_graph

def test_save_graph(tmp_path):
    graph = nx.DiGraph()
    graph.add_edge("a.py", "b.py")

    output = tmp_path / "graph.png"

    save_graph(graph, output)

    assert output.is_file()
    assert output.stat().st_size > 0
