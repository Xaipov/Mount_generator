import numpy as np
from scipy.spatial import cKDTree


def compute_road_direction(skel_pts, dist_map, window=10):
    """
    Для каждой точки скелета вычисляет направление движения (вектор) на основе
    соседних точек по dist_map (упорядоченных по расстоянию от старта).

    Возвращает dict: (z,x) -> (dz, dx) нормализованный вектор направления.
    """
    tree = cKDTree(skel_pts)
    directions = {}

    pt_to_dist = dist_map
    for pt in skel_pts:
        pz, px = pt
        idxs = tree.query_ball_point(pt, window)
        neighbors = skel_pts[idxs]

        d_self = pt_to_dist.get((pz, px), 0)
        # Точки немного дальше и немного ближе по трассе
        ahead = None
        behind = None
        best_ahead_diff = 1e9
        best_behind_diff = 1e9

        for n in neighbors:
            nz, nx = n
            dn = pt_to_dist.get((nz, nx))
            if dn is None:
                continue
            diff = dn - d_self
            if diff > 0 and diff < best_ahead_diff:
                best_ahead_diff = diff
                ahead = (nz, nx)
            elif diff < 0 and -diff < best_behind_diff:
                best_behind_diff = -diff
                behind = (nz, nx)

        if ahead is not None and behind is not None:
            dz = ahead[0] - behind[0]
            dx = ahead[1] - behind[1]
        elif ahead is not None:
            dz = ahead[0] - pz
            dx = ahead[1] - px
        elif behind is not None:
            dz = pz - behind[0]
            dx = px - behind[1]
        else:
            dz, dx = 0, 1

        norm = (dz**2 + dx**2) ** 0.5
        if norm < 1e-6:
            directions[(pz, px)] = (0.0, 1.0)
        else:
            directions[(pz, px)] = (dz / norm, dx / norm)

    return directions


def get_side(direction, point_offset):
    """
    direction: (dz, dx) — направление движения по трассе
    point_offset: (oz, ox) — смещение точки от центра скелета

    Возвращает 'left' или 'right' относительно направления движения
    (право от направления движения = поворот вектора направления на -90°).
    """
    dz, dx = direction
    oz, ox = point_offset
    # Перпендикуляр (право) = (dx, -dz) повёрнутый вектор движения
    # cross product z-component: dz*ox - dx*oz определяет сторону
    cross = dz * ox - dx * oz
    return 'right' if cross > 0 else 'left'
