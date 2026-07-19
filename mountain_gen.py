import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, median_filter, distance_transform_edt
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

# Должно совпадать с DIRT_SHOULDER_WIDTH в mountain_commands.py
SHOULDER_WIDTH = 4


def generate_noise_field(shape, seed=NOISE_SEED, scale=NOISE_SCALE, amplitude=NOISE_AMPLITUDE):
    gen = OpenSimplex(seed=seed)
    h, w = shape
    xs = np.arange(w) * scale
    ys = np.arange(h) * scale
    return (gen.noise2array(xs, ys) * amplitude).astype(np.float32)


def generate_macro_noise_field(shape, seed=MACRO_NOISE_SEED, scale=MACRO_NOISE_SCALE, amplitude=MACRO_NOISE_AMPLITUDE):
    gen = OpenSimplex(seed=seed)
    h, w = shape
    xs = np.arange(w) * scale
    ys = np.arange(h) * scale
    return (gen.noise2array(xs, ys) * amplitude).astype(np.float32)


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
    base_y = max(Y_END - 20, WORLD_MIN_Y)
    h, w = mask.shape
    if flat_base_y is None:
        flat_base_y = int(np.mean(list(pixel_y.values()))) if pixel_y else 0

    if macro_noise_field is None:
        macro_noise_field = np.zeros((h, w), dtype=np.float32)

    road_pts = np.array(np.where(mask > 128)).T
    if len(road_pts) == 0:
        return np.full((h, w), flat_base_y, dtype=np.int32), np.zeros((h, w), dtype=bool)

    # ── Карта road_y: для каждого пикселя — высота ближайшего асфальта ──────────
    road_y_map = np.full((h, w), flat_base_y, dtype=np.float32)
    for (rz, rx), ry in pixel_y.items():
        road_y_map[rz, rx] = float(ry)

    road_mask_bool = mask > 128
    dist_to_road, nn_indices = distance_transform_edt(~road_mask_bool, return_indices=True)
    nearest_road_y = road_y_map[nn_indices[0], nn_indices[1]]

    # Зона асфальт + обочина — эти пиксели должны быть СТРОГО на уровне road_y
    shoulder_zone = dist_to_road <= SHOULDER_WIDTH

    # ── Высота горы: от уровня асфальта вниз по склону ──────────────────────────
    # Склон начинается после обочины (dist > SHOULDER_WIDTH)
    effective_dist = np.maximum(0.0, dist_to_road - SHOULDER_WIDTH)
    mountain_y = nearest_road_y - effective_dist * mountain_slope

    # ── Шум только за пределами обочины ─────────────────────────────────────────
    noise_blend = np.clip((dist_to_road - SHOULDER_WIDTH - 1.0) / 4.0, 0.0, 1.0)
    mountain_y += noise_field * 0.5 * noise_blend
    mountain_y += macro_noise_field * 0.3 * noise_blend

    # ── Clamp: нигде не выше асфальта ────────────────────────────────────────────
    mountain_y = np.minimum(mountain_y, nearest_road_y)

    # ── Сглаживание только вдали от обочины ──────────────────────────────────────
    if mountain_smooth_sigma > 0:
        smoothed = gaussian_filter(mountain_y, sigma=mountain_smooth_sigma)
        blend_radius = float(SHOULDER_WIDTH + 3)
        t_blend = np.clip((dist_to_road - SHOULDER_WIDTH) / blend_radius, 0.0, 1.0)
        mountain_y = mountain_y * (1 - t_blend) + smoothed * t_blend

    mountain_y = median_filter(mountain_y, size=3)

    # ── Clamp после фильтров ──────────────────────────────────────────────────────
    mountain_y = np.minimum(mountain_y, nearest_road_y)

    # ── Не уходим ниже base_y ────────────────────────────────────────────────────
    mountain_y = np.maximum(mountain_y, float(base_y))

    # ── Затухание у краёв холста ─────────────────────────────────────────────────
    floor_y = float(base_y + EDGE_FADE_FLOOR_OFFSET)
    factor = np.ones((h, w), dtype=np.float32)

    if edge_fade_margin > 0:
        zz, xx = np.indices((h, w))
        edge_dist = np.minimum.reduce([zz, xx, h - 1 - zz, w - 1 - xx]).astype(np.float32)
        t_edge = np.clip(edge_dist / edge_fade_margin, 0.0, 1.0)
        factor = np.minimum(factor, t_edge * t_edge * (3 - 2 * t_edge))

    faded = floor_y + (mountain_y - floor_y) * factor
    result = np.round(faded).astype(np.int32)

    # ── Принудительно восстанавливаем зону дорога+обочина после edge_fade ────────
    # edge_fade не должен трогать эту зону — она всегда на уровне асфальта
    shoulder_road_y = np.round(nearest_road_y).astype(np.int32)
    result[shoulder_zone] = np.clip(shoulder_road_y[shoulder_zone], int(base_y), None)

    gz, gx = np.gradient(result.astype(np.float32))
    slope_mag = np.sqrt(gz ** 2 + gx ** 2)
    steep_mask = slope_mag > steep_slope_threshold

    return result, steep_mask