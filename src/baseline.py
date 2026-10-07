import time
import pandas as pd
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "python-louvain-master"))

import community as community_louvain
from utils import load_facebook
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def run_baseline(G, seeds=(0, 1, 2, 3, 4)):
    rows = []
    for s in seeds:
        t = time.perf_counter()
        part = community_louvain.best_partition(G, random_state=s)
        elapsed = time.perf_counter() - t
        rows.append({
            "seed": s,
            "modularity": community_louvain.modularity(part, G),
            "num_communities": len(set(part.values())),
            "time_sec": round(elapsed, 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    G = load_facebook()
    df = run_baseline(G)
    print(df.to_string(index=False))
    print("\nTrung bình:")
    print(df[["modularity", "num_communities", "time_sec"]].mean().round(4))
    df.to_csv(RESULTS / "baseline.csv", index=False)