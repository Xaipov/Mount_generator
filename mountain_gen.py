import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, median_filter
from opensimplex import OpenSimplex
from config import Y_END, WORLD_MIN_Y
from mountain_config import (
    NOISE_SCALE, NOISE_AMPLITUDE, NOISE_SEED,
    MACRO_NOISE_SCALE, MACRO_NOISE_AMPLITUDE, MACRO_NOISE_SEED, MACRO_PAD_MIN,
    NEAR_ROAD_RADIUS, MOUNTAIN_PAD, MOUNTAIN_SLOPE,
    LEFT_CARVE_DEPTH, RIGHT_CLIFF_DROP, EDGE_FADE_MARGIN, EDGE_FADE_FLOOR_OFFSET,
    FINISH_TAPER_END_MARGIN, FINISH_TAPER_LENGTH, STEEP_SLOPE_THRESHOLD,
    MOUNTAIN_SMOOTH_SIGMA,
)


def generate_noise_field(shape, seed=NOISE_SEED, scale=NOISE_SCALE, amplitude=NOISE_AMPLITUDE):
    """Мелкий шум поверхности (текстура) — высокая частота, небольшая амплитуда."""
    gen = OpenSimplex(seed=seed)
    h, w = shape
    xs = np.arange(w) * scale
    ys = np.arange(h) * scale
    return (gen.noise2array(xs, ys) * amplitude).astype(np.float32)


def generate_macro_noise_field(shape, seed=MACRO_NOISE_SEED, scale=MACRO_NOISE_SCALE,
                               amplitude=MACRO_NOISE_AMPLITUDE):
    """
    Крупный шум формы горы — низкая частота, большая амплитуда.
    """
    gen = OpenSimplex(seed=seed)
    h, w = shape
    xs = np.arange(w) * scale
    ys = np.arange(h) * scale
    return (gen.noise2array(xs, ys) * amplitude).astype(np.float32)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_mountain_height(mask, skel_pts, dist_map, tree_skel, pixel_y,
                            noise_field, macro_noise_field=None,
                            side_map=None, series_map=None,
                            curvature=None,
                            mountain_pad=MOUNTAIN_PAD,
                            mountain_slope=MOUNTAIN_SLOPE,
                            near_radius=NEAR_ROAD_RADIUS,
                            left_carve_depth=LEFT_CARVE_DEPTH,
                            right_cliff_drop=RIGHT_CLIFF_DROP,
                            edge_fade_margin=EDGE_FADE_MARGIN,
                            edge_fade_floor_offset=EDGE_FADE_FLOOR_OFFSET,
                            finish_taper_end_margin=FINISH_TAPER_END_MARGIN,
                            finish_taper_length=FINISH_TAPER_LENGTH,
                            steep_slope_threshold=STEEP_SLOPE_THRESHOLD,
                            mountain_smooth_sigma=MOUNTAIN_SMOOTH_SIGMA,
                            flat_base_y=None):
    """
    Вычисляет высоту горы через проекцию вдоль трассы (не по прямой!).
    """

    base_y = max(Y_END - 20, WORLD_MIN_Y)

    h, w = mask.shape
    if flat_base_y is None:
        flat_base_y = int(np.mean(list(pixel_y.values()))) if pixel_y else 0

    if macro_noise_field is None:
        macro_noise_field = np.zeros((h, w), dtype=np.float32)

    road_pts = np.array(np.where(mask > 128)).T
    if len(road_pts) == 0:
        empty = np.full((h, w), flat_base_y, dtype=np.int32)
        return empty, np.zeros((h, w), dtype=bool)

    # ── Используем скелет как основу ─────────────
    skel_tree = cKDTree(skel_pts)
    all_pts = np.array([(z, x) for z in range(h) for x in range(w)])
    dists_to_skel, nn_skel_idx = skel_tree.query(all_pts, workers=-1)
    nearest_skel_pt = skel_pts[nn_skel_idx]

    # Расстояние вдоль трассы от старта
    max_route_dist = max(dist_map.values()) if dist_map else 0.0
    route_dist_flat = np.zeros(len(all_pts), dtype=np.float32)
    for i in range(len(all_pts)):
        ns = tuple(nearest_skel_pt[i])
        route_dist_flat[i] = dist_map.get(ns, max_route_dist)

    # ── Маппинг скелет → side/series ──────────────
    road_tree = cKDTree(road_pts)
    _, nn_skel_to_road = road_tree.query(skel_pts)

    skel_side = {}
    skel_series = {}
    side_map = side_map or {}
    series_map = series_map or {}

    for i, sp in enumerate(skel_pts):
        rp = road_pts[nn_skel_to_road[i]]
        rp_key = (int(rp[0]), int(rp[1]))
        skel_side[tuple(sp)] = side_map.get(rp_key)
        skel_series[tuple(sp)] = series_map.get(rp_key, False)

    # ── Вычисляем высоту ────────────────────────
    mountain_y_flat = np.zeros(len(all_pts), dtype=np.float32)

    for i in range(len(all_pts)):
        z, x = all_pts[i]
        d_to_skel = float(dists_to_skel[i])
        ns = tuple(nearest_skel_pt[i])

        # Находим ближайшую точку дороги
        rp = road_pts[nn_skel_to_road[nn_skel_idx[i]]]
        rp_key = (int(rp[0]), int(rp[1]))
        road_y = pixel_y.get(rp_key, flat_base_y)

        side = skel_side.get(ns)
        is_series = skel_series.get(ns, False)
        noise_val = noise_field[z, x]

        # Локальный пик над точкой дороги
        macro_val = macro_noise_field[rp_key[0], rp_key[1]]
        cur_curvature = curvature.get(ns, 0.0) if curvature else 0.0

        # ─ ПРАВКА 1: После 30 блоков от старта гора ниже трассы на 1 блок ──
        current_route_dist = route_dist_flat[i]
        adjusted_road_y = road_y
        if current_route_dist >= 30:
            adjusted_road_y = road_y - 1

        # ── Расчет целевой высоты "стены" (target) ──
        if cur_curvature < 0.08:  # прямая
            target = adjusted_road_y + 2  # минимальная высота
        elif is_series:  # серия поворотов
            target = adjusted_road_y + 3  # почти нет стены
        else:  # одиночный поворот
            local_pad = max(MACRO_PAD_MIN, mountain_pad + macro_val)
            if side == 'left':
                target = adjusted_road_y + local_pad - left_carve_depth
            else:
                target = adjusted_road_y + local_pad

        # ── ПРАВКА 2: Убрали "лишнее расстояние" (плоскую полку) ──
        # Теперь переход от дороги к стене идет сразу (через smoothstep)
        if d_to_skel <= near_radius:
            t = _smoothstep(d_to_skel / near_radius) if near_radius > 0 else 1.0
            val = adjusted_road_y + (target - adjusted_road_y) * t
        else:
            slope_drop = (d_to_skel - near_radius) * mountain_slope
            val = target - slope_drop + noise_val
            if val < base_y:
                val = base_y

        mountain_y_flat[i] = val

    raw_height = mountain_y_flat.reshape(h, w)
    route_dist_grid = route_dist_flat.reshape(h, w)
    dist_to_skel_grid = dists_to_skel.reshape(h, w)

    far_mask = dist_to_skel_grid > near_radius

    # ── Сглаживание ───────────────────────────────
    if mountain_smooth_sigma > 0:
        smoothed_height = gaussian_filter(raw_height, sigma=mountain_smooth_sigma)
        raw_height = np.where(far_mask, smoothed_height, raw_height)

        smoothed_route_dist = gaussian_filter(route_dist_grid, sigma=mountain_smooth_sigma)
        route_dist_grid = np.where(far_mask, smoothed_route_dist, route_dist_grid)

    result = np.round(raw_height).astype(np.int32)
    result = median_filter(result, size=5)

    floor_y = base_y + EDGE_FADE_FLOOR_OFFSET
    factor = np.ones((h, w), dtype=np.float32)

    if edge_fade_margin > 0:
        zz, xx = np.indices((h, w))
        edge_dist = np.minimum.reduce([zz, xx, h - 1 - zz, w - 1 - xx]).astype(np.float32)
        if mountain_smooth_sigma > 0:
            edge_dist = gaussian_filter(edge_dist, sigma=mountain_smooth_sigma)
        t_edge = np.clip(edge_dist / edge_fade_margin, 0.0, 1.0)
        factor = np.minimum(factor, t_edge * t_edge * (3 - 2 * t_edge))

    if finish_taper_length > 0:
        dist_to_finish = max_route_dist - route_dist_grid
        t_finish = np.clip(
            (dist_to_finish - finish_taper_end_margin) / finish_taper_length, 0.0, 1.0
        )
        factor = np.minimum(factor, t_finish * t_finish * (3 - 2 * t_finish))

    faded = floor_y + (result.astype(np.float32) - floor_y) * factor

    if mountain_smooth_sigma > 0:
        smoothed_faded = gaussian_filter(faded, sigma=mountain_smooth_sigma)
        faded = np.where(far_mask, smoothed_faded, faded)

    result = np.round(faded).astype(np.int32)

    # Маска крутых склонов
    gz, gx = np.gradient(faded)
    slope_mag = np.sqrt(gz ** 2 + gx ** 2)
    steep_mask = slope_mag > steep_slope_threshold

    return result, steep_mask