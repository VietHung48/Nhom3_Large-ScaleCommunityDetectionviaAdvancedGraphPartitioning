import networkx as nx
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
"""Đường dẫn đến thư mục dữ liệu."""

def load_facebook():
    """Nạp đồ thị Facebook gộp sẵn thành một đồ thị vô hướng."""
    return nx.read_edgelist(DATA / "facebook_combined.txt", nodetype=int)
def load_youtube():
    """Nạp đồ thị YouTube gộp sẵn thành một đồ thị vô hướng."""
    return nx.read_edgelist(DATA / "com-youtube.ungraph.txt", nodetype=int)

def graph_stats(G,clustering = False):
    """Thống kê cho bảng dataset trong báo cáo."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    s = {
        "nodes": n,
        "edges": m,
        "avg_degree": round(2 * m / n, 2),
        "connected_components": nx.number_connected_components(G),
    }
    if clustering:
        s["avg_clustering"] = round(nx.average_clustering(G), 4)
    return s


if __name__ == "__main__":
    G = load_youtube()
    for k, v in graph_stats(G).items():
        print(f"{k}: {v}")