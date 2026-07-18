from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from config import Y_END, Y_START, FLAT_START_BLOCKS, STEP_INTERVAL


def bfs_order(skel, start_z, start_x, bridge_max_gap=8):
    """
    Обходит скелет BFS от ближайшего к (start_z, start_x) пикселя.
    Возвращает dict: (z, x) -> реальное расстояние от старта (с учётом диагоналей √2).

    Скелет после skeletonize иногда рвётся на несколько несвязных кусков
    (тонкие/пересекающиеся участки трассы). BFS по 8-связности видит только
    свою компоненту — все точки в других кусках остаются вне dist_map.
    Раньше это тихо подставлялось как max_dist в compute_pixel_data, отсюда
    провалы высоты на трассе (кусок вдруг "телепортируется" в конец трассы).

    Здесь после основного BFS оставшиеся недостижимые точки сшиваются с уже
    пройденной частью — ищем ближайшую пару (непройденная точка, пройденная
    точка) и продолжаем BFS от неё с расстоянием = расстояние_пройденной +
    евклидов разрыв. Повторяем, пока все точки не разобраны или разрыв
    больше bridge_max_gap (тогда мостик не кидаем, чтобы не притягивать
    случайный мусор с другого конца картинки).
    """
    h, w = skel.shape
    skel_pts = np.array(np.where(skel > 0)).T
    if len(skel_pts) == 0:
        return {}

    tree = cKDTree(skel_pts)
    _, idx = tree.query([start_z, start_x])
    sz, sx = skel_pts[idx]

    dist = {(sz, sx): 0.0}
    queue = deque([(sz, sx, 0.0)])

    # шаг по диагонали = sqrt(2), по прямой = 1
    dirs = [
        (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
        (0, -1, 1.0),                   (0, 1, 1.0),
        (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414),
    ]

    def expand(queue):
        while queue:
            z, x, d = queue.popleft()
            for dz, dx, step in dirs:
                nz, nx = z + dz, x + dx
                if 0 <= nz < h and 0 <= nx < w and skel[nz, nx] and (nz, nx) not in dist:
                    dist[(nz, nx)] = d + step
                    queue.append((nz, nx, d + step))

    expand(queue)

    # ── сшивка оторванных фрагментов ────────────────────────────────────────
    # За раунд подхватываем СРАЗУ ВСЕ непройденные точки, у которых разрыв
    # до уже пройденной части <= bridge_max_gap (а не одну точку за раз) —
    # иначе при большом числе мелких оторванных огрызков скелета (шумная
    # маска) это O(число_фрагментов) полных пересборок cKDTree и генерация
    # может казаться "зависшей" на много минут.
    all_pts_set = {tuple(p) for p in skel_pts}
    unreached = all_pts_set - dist.keys()

    while unreached:
        reached_arr = np.array(list(dist.keys()))
        unreached_list = list(unreached)
        unreached_arr = np.array(unreached_list)
        reached_tree = cKDTree(reached_arr)
        gaps, r_idx = reached_tree.query(unreached_arr)

        bridge_mask = gaps <= bridge_max_gap
        if not bridge_mask.any():
            # Остальные фрагменты слишком далеко — не выдумываем связь,
            # просто прекращаем (эти точки не входят в трассу).
            break

        # Обрабатываем кандидатов от самых близких к самым далёким — так
        # более короткие мостики успевают "накрыть" соседей через expand()
        # ещё до того, как до них дойдёт очередь как до отдельных мостиков.
        candidates_idx = np.where(bridge_mask)[0]
        candidates_idx = candidates_idx[np.argsort(gaps[candidates_idx])]

        newly_bridged = deque()
        for idx in candidates_idx:
            bridge_pt = tuple(unreached_arr[idx])
            if bridge_pt in dist:
                continue  # уже подхвачен через expand() соседнего мостика в этом же раунде
            connect_to = tuple(reached_arr[r_idx[idx]])
            base_d = dist[connect_to] + float(gaps[idx])
            dist[bridge_pt] = base_d
            newly_bridged.append((bridge_pt[0], bridge_pt[1], base_d))

        expand(newly_bridged)

        unreached = all_pts_set - dist.keys()

    return dist


def compute_y_start(max_dist, flat_start_blocks=FLAT_START_BLOCKS,
                     step_interval=STEP_INTERVAL, y_end=Y_END):
    """
    Рассчитывает Y_START на основе длины трассы.
    Каждые 2*step_interval блоков (block+slab) = 1 блок высоты.
    """
    effective_len = max_dist - flat_start_blocks
    pairs = effective_len / (2 * step_interval)
    y_start = round(y_end + pairs)
    return y_start, effective_len, pairs


def compute_pixel_data(mask, skel_pts, dist_map, tree_skel, y_start=Y_START,
                        y_end=Y_END, flat_start_blocks=FLAT_START_BLOCKS,
                        step_interval=STEP_INTERVAL):
    """
    Для каждого пикселя трассы вычисляет:
    - pixel_y: Y в Minecraft
    - pixel_slab: True если на этом участке ставим slab

    Логика:
    - первые flat_start_blocks блоков от старта: Y=y_start, slab=False
    - дальше: каждые step_interval блоков чередуем block/slab,
      каждая полная пара (block+slab) опускает Y на 1.
    """
    max_dist = max(dist_map.values()) or 1

    pixel_y = {}
    pixel_slab = {}

    road_pts = np.array(np.where(mask > 128)).T
    _, nn_idx = tree_skel.query(road_pts)

    for i, (rz, rx) in enumerate(road_pts):
        ns = tuple(skel_pts[nn_idx[i]])
        d = dist_map.get(ns, max_dist)

        if d < flat_start_blocks:
            pixel_y[(rz, rx)] = y_start
            pixel_slab[(rz, rx)] = False
        else:
            effective_d = d - flat_start_blocks
            cycle = effective_d // step_interval
            in_slab = (cycle % 2) == 1

            pairs_done = cycle // 2
            y = y_start - pairs_done
            y = max(y, y_end)

            pixel_y[(rz, rx)] = y
            pixel_slab[(rz, rx)] = in_slab

    return pixel_y, pixel_slab