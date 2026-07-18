import numpy as np
from scipy.spatial import cKDTree


def detect_curvature(skel_pts, dist_map, window=15):
    """
    Для каждой точки скелета считает кривизну (насколько резко меняется направление).
    Возвращает dict: (z,x) -> curvature (0 = прямая, больше = круче поворот).
    """
    tree = cKDTree(skel_pts)
    curvature = {}

    for pt in skel_pts:
        pz, px = pt
        idxs = tree.query_ball_point(pt, window)
        nb = skel_pts[idxs]
        if len(nb) < 3:
            curvature[(pz, px)] = 0.0
            continue
        centered = nb - nb.mean(axis=0)
        _, s, _ = np.linalg.svd(centered)
        # Отношение второго сингулярного значения к первому = "ширина" разброса
        ratio = s[1] / s[0] if s[0] > 1e-6 else 0.0
        curvature[(pz, px)] = float(ratio)

    return curvature


def classify_turn_series(skel_pts, dist_map, curvature, curve_threshold=0.12,
                          series_gap=15, min_series_turns=2):
    """
    Определяет какие точки относятся к "серии поворотов" (шиканы, S-образные участки)
    в отличие от одиночных поворотов.

    Логика:
    - Точка считается поворотом если curvature > curve_threshold
    - Поворотные точки группируются по близости вдоль трассы (по dist_map)
    - Если в группе несколько отдельных пиков поворота близко друг к другу (gap < series_gap)
      и пиков >= min_series_turns — это серия (шикана), иначе одиночный поворот

    Возвращает dict: (z,x) -> True если точка часть серии поворотов (НЕ делаем обрыв),
                              False если одиночный поворот или прямая (можно обрыв)
    """
    # Сортируем точки по расстоянию вдоль трассы
    pts_with_dist = [(dist_map.get((int(z), int(x)), 0), (int(z), int(x)))
                      for z, x in skel_pts]
    pts_with_dist.sort(key=lambda v: v[0])

    # Находим "пики" поворотов — локальные максимумы кривизны выше порога
    is_turn = [curvature.get(p, 0.0) > curve_threshold for _, p in pts_with_dist]

    # Группируем последовательные True-сегменты (отдельные повороты)
    turn_segments = []  # список (start_dist, end_dist, points)
    cur_seg = []
    for (d, p), t in zip(pts_with_dist, is_turn):
        if t:
            cur_seg.append((d, p))
        else:
            if cur_seg:
                turn_segments.append(cur_seg)
                cur_seg = []
    if cur_seg:
        turn_segments.append(cur_seg)

    # Группируем сегменты в серии, если расстояние между ними < series_gap
    series_flags = {}  # (z,x) -> bool (True = часть серии)

    i = 0
    while i < len(turn_segments):
        group = [turn_segments[i]]
        j = i + 1
        while j < len(turn_segments):
            prev_end = group[-1][-1][0]
            next_start = turn_segments[j][0][0]
            if next_start - prev_end < series_gap:
                group.append(turn_segments[j])
                j += 1
            else:
                break

        is_series = len(group) >= min_series_turns
        for seg in group:
            for _, p in seg:
                series_flags[p] = is_series

        i = j

    return series_flags


def expand_series_flag_to_road(mask, skel_pts, tree_skel, series_flags):
    """Расширяет флаг 'серия поворотов' с скелета на всю ширину трассы."""
    road_pts = np.array(np.where(mask > 128)).T
    if len(road_pts) == 0:
        return {}
    _, nn_idx = tree_skel.query(road_pts)
    result = {}
    for i, (rz, rx) in enumerate(road_pts):
        ns = tuple(skel_pts[nn_idx[i]])
        result[(rz, rx)] = series_flags.get(ns, False)
    return result