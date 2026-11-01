from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # chỉ lưu file ảnh, không mở cửa sổ
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

DATASETS = {"facebook": "Facebook", "youtube": "YouTube"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.bottom": False,
    "ytick.left": False,
    "font.size": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def vn(x, digits=3):
    """Định dạng số theo kiểu báo cáo: dấu phẩy thập phân."""
    return f"{x:.{digits}f}".replace(".", ",")


def seconds(x):
    return f"{vn(x, 3)}s" if x < 100 else f"{x:.0f}s"


def comma_ticks(axis, digits):
    """Dùng dấu phẩy thập phân cho nhãn trục, khớp với nhãn số liệu."""
    axis.set_major_formatter(FuncFormatter(lambda x, _: vn(x, digits)))


def first_row(name):
    """Dòng đầu của một file kết quả (các file *_<dataset>.csv chỉ chạy seed 0)."""
    return pd.read_csv(RESULTS / name).iloc[0]


def load_methods(dataset):
    """Gom các phương pháp của một dataset về cùng một dạng để vẽ."""
    base = first_row(f"baseline_{dataset}.csv")
    no_refine = first_row(f"multilevel_no_refine_{dataset}.csv")
    rows = [
        {"name": "Louvain gốc", "q": base["modularity"], "coarsen": 0,
         "louvain": base["time_sec"], "refine": 0, "total": base["time_sec"]},
        {"name": "Chưa tinh chỉnh", "q": no_refine["modularity"],
         "coarsen": no_refine["t_coarsen"], "louvain": no_refine["t_initial"],
         "refine": no_refine["t_project"], "total": no_refine["time_total"]},
    ]
    sequential = RESULTS / f"multilevel_refine_{dataset}.csv"
    if sequential.exists():  # chạy refine trực tiếp, không qua Pool
        r = first_row(sequential.name)
        rows.append({"name": "Tinh chỉnh tuần tự", "q": r["modularity"],
                     "coarsen": r["t_coarsen"], "louvain": r["t_initial"],
                     "refine": r["t_refine"], "total": r["time_total"]})
    for _, r in pd.read_csv(RESULTS / f"parallel_{dataset}.csv").iterrows():
        rows.append({"name": f"Tinh chỉnh qua Pool, k={int(r['k_processes'])}",
                     "q": r["modularity"], "coarsen": r["t_coarsen"],
                     "louvain": r["t_initial"], "refine": r["t_parallel_refine"],
                     "total": r["time_total"]})
    return pd.DataFrame(rows)


def plot_modularity():
    """Q của từng phương pháp. Trục hai bên cùng độ rộng 0,08 để so được chênh lệch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (dataset, title), xlim in zip(axes, DATASETS.items(),
                                          [(0.78, 0.86), (0.66, 0.74)]):
        df = load_methods(dataset)
        df = df[df["name"] != "Tinh chỉnh tuần tự"]  # cùng Q với k=1
        y = range(len(df))
        ax.hlines(y, xlim[0], df["q"], color=GRID, linewidth=1)
        ax.scatter(df["q"], y, s=70, color=BLUE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        for yi, q in zip(y, df["q"]):
            ax.annotate(vn(q, 4), (q, yi), xytext=(9, 0),
                        textcoords="offset points", va="center", color=INK, zorder=4,
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
        ax.axvline(df["q"].iloc[0], color=MUTED, linestyle="--", linewidth=1)
        ax.set_yticks(list(y), df["name"])
        ax.invert_yaxis()
        ax.set_xlim(xlim)
        comma_ticks(ax.xaxis, 2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Modularity Q (trục không bắt đầu từ 0; nét đứt: Louvain gốc)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_modularity.png")
    plt.close(fig)


def plot_time_phases():
    """Thời gian chia theo pha. Hai dataset chênh nhau hàng trăm lần nên mỗi panel một thang."""
    phases = [("coarsen", "Thu nhỏ", BLUE),
              ("louvain", "Louvain", ORANGE),
              ("refine", "Chiếu nhãn + tinh chỉnh", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        df = load_methods(dataset)
        y = range(len(df))
        left = pd.Series(0.0, index=df.index)
        for col, label, color in phases:
            ax.barh(y, df[col], left=left, height=0.62, color=color, label=label,
                    edgecolor=SURFACE, linewidth=1.5)
            left = left + df[col]
        for yi, total in zip(y, df["total"]):
            ax.annotate(seconds(total), (total, yi), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK)
        ax.set_yticks(list(y), df["name"])
        ax.invert_yaxis()
        ax.set_xlim(0, df["total"].max() * 1.2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Thời gian (giây)")
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "02_time_phases.png")
    plt.close(fig)


def plot_speedup():
    """Speedup T(1)/T(k), tính cho toàn pipeline và riêng pha refine song song."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        df = pd.read_csv(RESULTS / f"parallel_{dataset}.csv")
        k = df["k_processes"]
        total = df["time_total"].iloc[0] / df["time_total"]
        refine = df["t_parallel_refine"].iloc[0] / df["t_parallel_refine"]
        ax.plot(k, k, color=MUTED, linestyle="--", linewidth=1,
                label="Lý tưởng (speedup = k)")
        ax.plot(k, total, color=BLUE, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, label="Toàn pipeline")
        ax.plot(k, refine, color=ORANGE, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, label="Riêng pha refine")
        for i in range(1, len(df)):  # giá trị cao hơn ghi phía trên, thấp hơn phía dưới
            above = total.iloc[i] >= refine.iloc[i]
            for s, up in ((total.iloc[i], above), (refine.iloc[i], not above)):
                ax.annotate(f"{vn(s, 2)}x", (k.iloc[i], s),
                            xytext=(0, 9 if up else -9), textcoords="offset points",
                            ha="center", va="bottom" if up else "top", color=INK)
        ax.set_xticks(k)
        ax.set_ylim(0, 4.3)
        comma_ticks(ax.yaxis, 1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Số tiến trình k")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Speedup so với k=1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "03_speedup.png")
    plt.close(fig)


def plot_seed_stability():
    """Q qua 5 seed trên Facebook, từ baseline.csv và multilevel_no_refine.csv."""
    groups = [("Louvain gốc", "baseline.csv"),
              ("Chưa tinh chỉnh", "multilevel_no_refine.csv")]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for y, (name, path) in enumerate(groups):
        df = pd.read_csv(RESULTS / path)
        other = df[df["seed"] != 0]
        seed0 = df[df["seed"] == 0]
        ax.scatter(other["modularity"], [y] * len(other), s=70, color=BLUE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5,
                   label="Seed 1-4" if y == 0 else None)
        ax.scatter(seed0["modularity"], [y] * len(seed0), s=70, color=ORANGE, zorder=4,
                   edgecolor=SURFACE, linewidth=1.5,
                   label="Seed 0 (số trong bảng kết quả)" if y == 0 else None)
        mean = df["modularity"].mean()
        ax.vlines(mean, y - 0.25, y + 0.25, color=INK, linewidth=2, zorder=2,
                  label="Trung bình" if y == 0 else None)
        ax.annotate(f"TB {vn(mean, 4)}", (mean, y), xytext=(0, 17),
                    textcoords="offset points", ha="center", color=INK)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(1.7, -0.7)
    ax.set_xlim(0.778, 0.84)
    comma_ticks(ax.xaxis, 2)
    ax.set_title("Facebook: Q qua 5 seed", loc="left")
    ax.set_xlabel("Modularity Q (trục không bắt đầu từ 0)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), labelcolor=INK_2)
    fig.savefig(FIGURES / "04_seed_stability.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_modularity()
    plot_time_phases()
    plot_speedup()
    plot_seed_stability()
    print(f"Đã lưu biểu đồ vào {FIGURES}")
