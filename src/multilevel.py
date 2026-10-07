import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "python-louvain-master"))

import community as community_louvain
from utils import load_facebook
from coarsen import build_hierarchy, project

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def multilevel_no_refine(G, seed=0, min_nodes=200):
    """Pha 1: coarsen. Pha 2: Louvain trên đồ thị nhỏ nhất.
    Pha 3 (uncoarsen): chiếu nhãn ngược về G0, CHƯA refine."""
    t0 = time.perf_counter()
    levels, maps = build_hierarchy(G, min_nodes=min_nodes, seed=seed)
    t_coarsen = time.perf_counter() - t0

    t1 = time.perf_counter()
    labels = community_louvain.best_partition(
        levels[-1], weight="weight", random_state=seed)
    t_init = time.perf_counter() - t1

    t2 = time.perf_counter()
    for i in range(len(maps) - 1, -1, -1):
        labels = project(labels, maps[i])
    t_project = time.perf_counter() - t2

    q = community_louvain.modularity(labels, levels[0], weight="weight")
    return {
        "seed": seed,
        "levels": len(levels),
        "coarsest_nodes": levels[-1].number_of_nodes(),
        "num_communities": len(set(labels.values())),
        "modularity": q,
        "t_coarsen": round(t_coarsen, 3),
        "t_initial": round(t_init, 3),
        "t_project": round(t_project, 3),
        "time_total": round(t_coarsen + t_init + t_project, 3),
    }


if __name__ == "__main__":
    G = load_facebook()
    rows = [multilevel_no_refine(G, seed=s) for s in range(5)]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    base = pd.read_csv(RESULTS / "baseline.csv")
    print("\n== So sánh trung bình ==")
    print(f"Louvain gốc      : Q = {base['modularity'].mean():.4f}, "
          f"time = {base['time_sec'].mean():.3f}s")
    print(f"Multilevel (chưa refine): Q = {df['modularity'].mean():.4f}, "
          f"time = {df['time_total'].mean():.3f}s")

    df.to_csv(RESULTS / "multilevel_no_refine.csv", index=False)