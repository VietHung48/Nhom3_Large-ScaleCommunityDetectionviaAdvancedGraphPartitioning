import random
import networkx as nx
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "python-louvain-master"))

import community as community_louvain
from utils import load_facebook

def coarsen_once(G, rng):
    """Một cấp heavy-edge matching.
    Trả về (H, mapping): H là đồ thị thu nhỏ, mapping[đỉnh cũ] = siêu đỉnh."""
    nodes = list(G.nodes())
    rng.shuffle(nodes)

    mapping, matched, cid = {}, set(), 0
    for v in nodes:
        if v in matched:
            continue
        best, best_w = None, -1
        for u, d in G[v].items():
            if u == v or u in matched:
                continue
            w = d.get("weight", 1)
            if w > best_w:
                best, best_w = u, w
        matched.add(v)
        mapping[v] = cid
        if best is not None:
            matched.add(best)
            mapping[best] = cid
        cid += 1

    H = nx.Graph()
    H.add_nodes_from(range(cid), vw=0)
    for v, c in mapping.items():
        H.nodes[c]["vw"] += G.nodes[v].get("vw", 1)

    for u, v, d in G.edges(data=True):      # self-loop của G cũng được duyệt 1 lần
        a, b, w = mapping[u], mapping[v], d.get("weight", 1)
        if H.has_edge(a, b):                # a == b thì chính là cộng vào self-loop
            H[a][b]["weight"] += w
        else:
            H.add_edge(a, b, weight=w)
    return H, mapping


def build_hierarchy(G, min_nodes=200, max_levels=30, min_shrink=0.95, seed=0):
    """Tạo chuỗi G0 -> G1 -> ... -> Gk.
    Trả về (levels, maps): maps[i] ánh xạ đỉnh của levels[i] sang levels[i+1]."""
    rng = random.Random(seed)
    G0 = nx.Graph()
    G0.add_nodes_from(G.nodes(), vw=1)
    G0.add_edges_from((u, v, {"weight": d.get("weight", 1)})
                      for u, v, d in G.edges(data=True))

    levels, maps = [G0], []
    while levels[-1].number_of_nodes() > min_nodes and len(maps) < max_levels:
        H, m = coarsen_once(levels[-1], rng)
        if H.number_of_nodes() > min_shrink * levels[-1].number_of_nodes():
            break                           # không co thêm được nữa
        levels.append(H)
        maps.append(m)
    return levels, maps


def project(labels, mapping):
    """Chiếu nhãn từ đồ thị thô về đồ thị mịn hơn một cấp (dùng cho uncoarsening)."""
    return {v: labels[c] for v, c in mapping.items()}


if __name__ == "__main__":
    

    G = load_facebook()
    levels, maps = build_hierarchy(G)

    print("== Kích thước từng cấp ==")
    for i, H in enumerate(levels):
        tw = sum(d["weight"] for _, _, d in H.edges(data=True))
        print(f"Level {i}: {H.number_of_nodes():5d} đỉnh, "
              f"{H.number_of_edges():6d} cạnh, tổng trọng số = {tw}")

    print("\n== Kiểm tra bảo toàn ==")
    # 1) tổng trọng số cạnh (kể cả self-loop) không đổi
    totals = {sum(d["weight"] for _, _, d in H.edges(data=True)) for H in levels}
    print("Tổng trọng số bảo toàn:", len(totals) == 1)
    # 2) tổng vw = số đỉnh gốc
    print("Tổng vw bảo toàn:",
          all(sum(vw for _, vw in H.nodes(data="vw")) == G.number_of_nodes()
              for H in levels))
    # 3) Q của cùng một phân hoạch phải bằng nhau ở mọi cấp
    coarse_part = community_louvain.best_partition(levels[-1], weight="weight",
                                                   random_state=0)
    labels = coarse_part
    q_coarse = community_louvain.modularity(labels, levels[-1], weight="weight")
    for i in range(len(maps) - 1, -1, -1):
        labels = project(labels, maps[i])
        q = community_louvain.modularity(labels, levels[i], weight="weight")
        print(f"Q tại level {i}: {q:.6f}  (level {len(levels)-1}: {q_coarse:.6f})")