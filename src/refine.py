

def compute_partition_info(G):
    """Tính tổng trọng số cạnh trong đồ thị (dùng cho ΔQ)."""
    return sum(d.get("weight", 1) for _, _, d in G.edges(data=True))


def refine(G, labels, max_passes=10, level=None):
    """Local moving: thử chuyển từng đỉnh sang cộng đồng hàng xóm nếu ΔQ > 0.
    Lặp tối đa max_passes lần hoặc đến khi không còn đỉnh nào di chuyển.
    
    """
    if level is not None:
        print(f"Refine level {level}: {G.number_of_nodes()} đỉnh", flush=True)
        
    m2 = 2 * compute_partition_info(G)  # 2m
    if m2 == 0:
        return labels

    labels = dict(labels)  # copy để không sửa bản gốc

    # Tính degree và internal weight cho mỗi cộng đồng
    community_degree = {}    # tổng bậc của cộng đồng c
    community_internal = {}  # tổng trọng số cạnh nội bộ của c

    for v in G.nodes():
        c = labels[v]
        deg = sum(d.get("weight", 1) for _, d in G[v].items())
        community_degree[c] = community_degree.get(c, 0) + deg

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        if labels[u] == labels[v]:
            c = labels[u]
            community_internal[c] = community_internal.get(c, 0) + w
            if u == v:  # self-loop đếm một lần
                continue
            community_internal[c] = community_internal.get(c, 0) + w

    for _ in range(max_passes):
        moved = 0
        for v in G.nodes():
            c_old = labels[v]
            deg_v = sum(d.get("weight", 1) for _, d in G[v].items())

            # Tính trọng số cạnh từ v tới từng cộng đồng hàng xóm
            neighbor_weight = {}
            for u, d in G[v].items():
                if u == v:
                    continue
                c = labels[u]
                w = d.get("weight", 1)
                neighbor_weight[c] = neighbor_weight.get(c, 0) + w

            # ΔQ khi rời c_old
            k_v_old = neighbor_weight.get(c_old, 0)
            dq_remove = (k_v_old / m2
                         - deg_v * (community_degree.get(c_old, 0) - deg_v) / (m2 ** 2))

            # Tìm cộng đồng tốt nhất để chuyển sang
            best_c, best_dq = c_old, 0
            for c_new, k_v_new in neighbor_weight.items():
                if c_new == c_old:
                    continue
                dq_add = (k_v_new / m2
                          - deg_v * community_degree.get(c_new, 0) / (m2 ** 2))
                dq = dq_add - dq_remove
                if dq > best_dq:
                    best_dq = dq
                    best_c = c_new

            if best_c != c_old:
                # Cập nhật community_degree
                community_degree[c_old] = community_degree.get(c_old, 0) - deg_v
                community_degree[best_c] = community_degree.get(best_c, 0) + deg_v
                # Cập nhật community_internal
                community_internal[c_old] = community_internal.get(c_old, 0) - k_v_old
                community_internal[best_c] = community_internal.get(best_c, 0) + neighbor_weight.get(best_c, 0)
                labels[v] = best_c
                moved += 1

        if moved == 0:
            break

    return labels