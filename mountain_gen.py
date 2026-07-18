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
    # Новые параметры для генерации команд
    SHELL_THICKNESS, MAX_RADIUS, SLOPE_FACTOR,
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
    Вычисляет высоту горы через проекцию вдоль трассы.
    Карта высот остается естественной, защита дороги работает через road_buffer_mask в mountain_commands.py.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter, median_filter

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

    # 1. Создаем базовую карту высот, инициализированную минимальной высотой
    mountain_y = np.full((h, w), base_y, dtype=np.float32)

    # 2. "Разливаем" высоту от каждой точки дороги, беря максимум
    radius = int(near_radius * 2.5)

    for i, rp in enumerate(road_pts):
        rz, rx = rp
        rp_key = (int(rz), int(rx))
        road_h = pixel_y.get(rp_key, flat_base_y)

        # Границы окна вокруг точки дороги
        z_min = max(0, rz - radius)
        z_max = min(h, rz + radius + 1)
        x_min = max(0, rx - radius)
        x_max = min(w, rx + radius + 1)

        # Создаем 2D сетку координат
        zz, xx = np.ogrid[z_min:z_max, x_min:x_max]

        # 2D массив расстояний
        dist = np.sqrt((zz - rz) ** 2 + (xx - rx) ** 2)

        # 2D булева маска
        valid = dist <= radius

        # Рассчитываем высоту
        drop = dist * mountain_slope
        local_h = road_h + drop

        # Массив-кандидат
        candidate_heights = np.where(valid, local_h, base_y)

        # Обновляем глобальную карту, беря максимум
        mountain_y[z_min:z_max, x_min:x_max] = np.maximum(
            mountain_y[z_min:z_max, x_min:x_max],
            candidate_heights
        )

    # 3. Добавляем шум
    mountain_y += noise_field * 0.5
    mountain_y += macro_noise_field * 0.3

    # 4. Сглаживаем ВСЮ карту высот
    if mountain_smooth_sigma > 0:
        mountain_y = gaussian_filter(mountain_y, sigma=mountain_smooth_sigma)

    # Дополнительное медианное сглаживание
    mountain_y = median_filter(mountain_y, size=3)

    # 5. Затухание к краям карты
    floor_y = base_y + EDGE_FADE_FLOOR_OFFSET
    factor = np.ones((h, w), dtype=np.float32)

    if edge_fade_margin > 0:
        zz, xx = np.indices((h, w))
        edge_dist = np.minimum.reduce([zz, xx, h - 1 - zz, w - 1 - xx]).astype(np.float32)
        t_edge = np.clip(edge_dist / edge_fade_margin, 0.0, 1.0)
        factor = np.minimum(factor, t_edge * t_edge * (3 - 2 * t_edge))

    # Применяем затухание
    faded = floor_y + (mountain_y - floor_y) * factor

    # Финальное округление
    result = np.round(faded).astype(np.int32)

    # 6. Маска крутых склонов
    gz, gx = np.gradient(result.astype(np.float32))
    slope_mag = np.sqrt(gz ** 2 + gx ** 2)
    steep_mask = slope_mag > steep_slope_threshold

    return result, steep_mask


def generate_mountain_commands(mountain_y, steep_mask, config, skel_to_pixel=None):
    """
    Генерирует команды для датапака на основе карты высот.

    КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Полностью заполняет объем горы от base_y до верха,
    чтобы не было воздушных карманов и "полок".

    Args:
        mountain_y: 2D numpy array с высотами (h, w)
        steep_mask: 2D numpy array с маской крутых склонов
        config: объект конфигурации
        skel_to_pixel: dict для конвертации (z, x) → (mc_x, mc_z), если нужно

    Returns:
        list строк с командами Minecraft
    """
    import random

    Y_END = getattr(config, 'Y_END', -58)
    WORLD_MIN_Y = getattr(config, 'WORLD_MIN_Y', -64)
    SHELL_THICKNESS = getattr(config, 'SHELL_THICKNESS', 1)

    base_y = max(Y_END - 20, WORLD_MIN_Y)

    h, w = mountain_y.shape
    commands = []

    # Проходим по всем пикселям карты высот
    for z in range(h):
        for x in range(w):
            top_y = int(mountain_y[z, x])

            # Пропускаем если высота ниже базовой
            if top_y <= base_y:
                continue

            # Определяем Minecraft координаты
            # Если skel_to_pixel предоставлен, используем его
            if skel_to_pixel and (z, x) in skel_to_pixel:
                mc_x, mc_z = skel_to_pixel[(z, x)]
            else:
                # По умолчанию: x → mc_x, z → mc_z (с центром в 0,0)
                mc_x = x - w // 2
                mc_z = z - h // 2

            # 1. ЗАПОЛНЯЕМ ЯДРО ГОРЫ (камень) — это решает проблему воздушных карманов!
            core_bottom = base_y
            core_top = top_y - SHELL_THICKNESS

            if core_top >= core_bottom:
                # ОДНА команда fill на всю колонку = быстро и без пустот
                commands.append(f"fill {mc_x} {core_bottom} {mc_z} {mc_x} {core_top} {mc_z} stone")

            # 2. КОРКА ПОВЕРХНОСТИ (ровно 1 блок)
            # 85% трава, 15% камень для разнообразия
            is_steep = steep_mask[z, x] if isinstance(steep_mask, np.ndarray) else False

            if is_steep:
                # На крутых склонах — камень
                surface_block = "stone"
            else:
                # На пологих — в основном трава
                surface_block = "grass_block" if random.random() > 0.15 else "stone"

            commands.append(f"setblock {mc_x} {top_y} {mc_z} {surface_block}")

    return commands