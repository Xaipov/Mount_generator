import numpy as np
import random
from scipy.ndimage import binary_dilation

from mountain_config import (
    FLAT_SHELL_THICKNESS, STEEP_SHELL_THICKNESS, ROAD_CLEARANCE
)
from config import Y_END
STACK_SIZE = 64
SHELL_THICKNESS = 1  # толщина корки горы (блоков от поверхности вниз)


def format_as_stacks(count):
    """Форматирует количество блоков в 'стаки + остаток'."""
    if count <= 0:
        return "0 блоков"

    stacks = count // STACK_SIZE
    remainder = count % STACK_SIZE

    if stacks == 0:
        return f"{remainder} блоков"
    elif remainder == 0:
        return f"{stacks} стак(ов)"
    else:
        return f"{stacks} стак(ов) + {remainder} блоков"


def generate_mountain_commands(mountain_y, offset_x, offset_z, y_end, max_fill_len=32,
                               steep_mask=None, mask=None, pixel_y=None):
    """
    Генерирует команды для ПОЛОЙ горы.
    Гора начинается от уровня дороги (асфальт на одном уровне с землей).
    """
    base_y = y_end
    h, w = mountain_y.shape
    if steep_mask is None:
        steep_mask = np.zeros((h, w), dtype=bool)

    road_mask = np.zeros((h, w), dtype=bool)
    if mask is not None:
        road_mask = mask > 128

    commands = []
    floor_segments = []
    current_floor_seg = None

    for z in range(h):
        for x in range(w):
            # Пропускаем ТОЛЬКО саму дорогу (не защитную зону)
            if road_mask[z, x]:
                continue

            top_y = int(mountain_y[z, x])
            is_steep = bool(steep_mask[z, x])

            if top_y <= base_y:
                continue

            mc_x = x + offset_x
            mc_z = z + offset_z

            # Динамическая толщина корки
            total_crust_thickness = STEEP_SHELL_THICKNESS if is_steep else FLAT_SHELL_THICKNESS
            stone_layers = total_crust_thickness - 1

            # Нижняя граница корки
            shell_bottom = max(base_y + 1, top_y - stone_layers)

            # Строим каменную часть корки
            if shell_bottom <= top_y - 1:
                commands.append(f"fill {mc_x} {shell_bottom} {mc_z} {mc_x} {top_y - 1} {mc_z} stone")

            # Верхний блок поверхности
            if is_steep:
                surface_block = "stone"
            else:
                surface_block = "grass_block" if random.random() > 0.15 else "stone"
            commands.append(f"setblock {mc_x} {top_y} {mc_z} {surface_block}")

            # Собираем данные для сплошного пола
            if current_floor_seg is None:
                current_floor_seg = {'x1': mc_x, 'x2': mc_x, 'z': mc_z}
            elif current_floor_seg['z'] == mc_z and current_floor_seg['x2'] + 1 == mc_x:
                current_floor_seg['x2'] = mc_x
            else:
                floor_segments.append(current_floor_seg)
                current_floor_seg = {'x1': mc_x, 'x2': mc_x, 'z': mc_z}

    if current_floor_seg is not None:
        floor_segments.append(current_floor_seg)

    # Генерируем СПЛОШНОЙ ПОЛ
    for seg in floor_segments:
        commands.append(f"fill {seg['x1']} {base_y} {seg['z']} {seg['x2']} {base_y} {seg['z']} stone")

    return commands


def count_mountain_resources(mountain_y, steep_mask=None, mask=None):
    """
    Подсчитывает количество блоков для ПОЛОЙ горы с толстой коркой.
    """
    base_y = Y_END
    h, w = mountain_y.shape
    if steep_mask is None:
        steep_mask = np.zeros((h, w), dtype=bool)

    road_mask = np.zeros((h, w), dtype=bool)
    if mask is not None:
        road_mask = mask > 128

    stone_count = 0
    dirt_count = 0
    surface_count = 0

    for z in range(h):
        for x in range(w):
            if road_mask[z, x]:
                continue

            top = int(mountain_y[z, x])
            is_steep = bool(steep_mask[z, x])
            dirt_layers = 1 if is_steep else 4

            if top <= base_y + 1:
                continue

            column_height = top - base_y + 1

            # Поверхность (1 блок)
            surface_count += 1

            # Грязь
            dirt_height = min(dirt_layers, column_height - 1)
            dirt_count += max(0, dirt_height)

            # Камень
            if column_height <= SHELL_THICKNESS:
                stone_height = column_height - 1 - dirt_height
            else:
                stone_height = SHELL_THICKNESS - 1 - dirt_height
                stone_height += 1  # +1 за пол на base_y

            stone_count += max(0, stone_height)

    # Рандом: 85% трава, 15% камень
    grass_count = int(surface_count * 0.85)
    stone_surface_count = surface_count - grass_count

    return {
        'stone': stone_count,
        'stone_str': format_as_stacks(stone_count),
        'dirt': dirt_count,
        'dirt_str': format_as_stacks(dirt_count),
        'grass': grass_count,
        'grass_str': format_as_stacks(grass_count),
        'stone_surface': stone_surface_count,
        'stone_surface_str': format_as_stacks(stone_surface_count),
    }