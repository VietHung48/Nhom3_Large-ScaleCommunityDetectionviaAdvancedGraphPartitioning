import time
import sys
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "python-louvain-master"))

import community as community_louvain
from utils import load_facebook, load_youtube
from coarsen import build_hierarchy, project
from refine import refine


def get_partition_regions(labels, k):
    """Chia các cộng đồng thành k vùng cân bằng (greedy bin-packing)."""
    from collections import Counter
    sizes = Counter(labels.values())
    # Sắp xếp cộng đồng theo kích thước giảm dần
    sorted_comms = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
    # Gán cộng đồng vào vùng đang nhỏ nhất
    region_size = [0] * k
    comm_to_region = {}
    for comm, size in sorted_comms:
        min_region = region_size.index(min(region_size))
        comm_to_region[comm] = min_region
        region_size[min_region] += size
    return comm_to_region, region_size


def refine_region(args):
    """Refine một vùng độc lập, chạy trong một process riêng."""
    sub, region_labels, region_id = args          # nhận đồ thị con, không nhận G
    print(f"Process {region_id}: {len(region_labels)} đỉnh", flush=True)
    t = time.perf_counter()
    refined = refine(sub, region_labels)          # bỏ .subgraph().copy()
    print(f"Process {region_id}: xong sau {time.perf_counter()-t:.1f}s", flush=True)
    return refined


def parallel_refine(G, labels, k=4):
    comm_to_region, region_size = get_partition_regions(labels, k)
    print(f"Chia thành {k} vùng, kích thước: {region_size}")

    regions = defaultdict(dict)
    for v, c in labels.items():
        regions[comm_to_region[c]][v] = c

    args = []
    for r in range(k):
        sub = G.subgraph(regions[r].keys()).copy()   # cắt ở process cha
        args.append((sub, regions[r], r))
    del regions

    with Pool(processes=k) as pool:
        results = pool.map(refine_region, args)
    del args

    merged = {}
    for refined in results:
        merged.update(refined)
    return merged


def multilevel_parallel(G, seed=0, min_nodes=200, k=4):
    """Pipeline đầy đủ với refine song song."""
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
        labels = parallel_refine(levels[i], labels, k=k)
        print(f"Level {i}: refine song song xong", flush=True)
    t_parallel = time.perf_counter() - t2

    q = community_louvain.modularity(labels, levels[0], weight="weight")
    return {
        "seed": seed,
        "k_processes": k,
        "levels": len(levels),
        "coarsest_nodes": levels[-1].number_of_nodes(),
        "num_communities": len(set(labels.values())),
        "modularity": q,
        "t_coarsen": round(t_coarsen, 3),
        "t_initial": round(t_init, 3),
        "t_parallel_refine": round(t_parallel, 3),
        "time_total": round(t_coarsen + t_init + t_parallel, 3),
    }


if __name__ == "__main__":
    from utils import ROOT
    import pandas as pd

    RESULTS = ROOT / "results"
    RESULTS.mkdir(exist_ok=True)

    # dataset = sys.argv[1] if len(sys.argv) > 1 else "facebook"
    # G = load_facebook() if dataset == "facebook" else load_youtube()
    
    dataset = "youtube"
    G = load_youtube()
    rows = []
    # Đo speedup: chạy với 1, 2, 4 process
    for k in [1, 2, 4]:
        print(f"\n=== {k} process(es) ===")
        r = multilevel_parallel(G, seed=0, k=k)
        rows.append(r)
        print(r)
        pd.DataFrame(rows).to_csv(RESULTS / f"parallel_{dataset}.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n== Bảng Speedup ==")
    base_time = df[df["k_processes"] == 1]["time_total"].values[0]
    df["speedup"] = base_time / df["time_total"]
    print(df[["k_processes", "modularity", "time_total", "speedup"]].to_string(index=False))
    df.to_csv(RESULTS / f"parallel_{dataset}.csv", index=False)