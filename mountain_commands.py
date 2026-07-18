import numpy as np
import random

from mountain_config import STONE_BLOCK, DIRT_BLOCK, GRASS_BLOCK, DIRT_LAYERS, STEEP_DIRT_LAYERS
from config import WORLD_MIN_Y, Y_END

STACK_SIZE = 64
SHELL_THICKNESS = 7  # толщина корки горы (блоков от поверхности вниз)


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
    Генерирует команды для ПОЛОЙ горы с толстой коркой (25 блоков).

    Структура колонки (сверху вниз):
    - 1 блок поверхности (трава 85% / камень 15%)
    - DIRT_LAYERS (4) или STEEP_DIRT_LAYERS (1) грязи
    - ~20 блоков камня (чтобы суммарно корка = 25 блоков)
    - ВОЗДУХ (полая часть, если колонка > 25 блоков)
    - 1 блок каменного пола на уровне base_y

    Если колонка ниже base_y + 1 — пропускаем (ниже уровня трассы не ставим).
    Если колонка ниже 25 блоков — ставим полностью (без пустоты внутри).
    """
    base_y = y_end  # уровень финиша трассы = пол горы
    h, w = mountain_y.shape
    if steep_mask is None:
        steep_mask = np.zeros((h, w), dtype=bool)

    road_mask = np.zeros((h, w), dtype=bool)
    if mask is not None:
        road_mask = mask > 128

    commands = []

    for z in range(h):
        row = mountain_y[z]
        steep_row = steep_mask[z]
        road_row = road_mask[z]

        start = 0
        cur_top = row[0]
        cur_steep = bool(steep_row[0])

        for x in range(1, w + 1):
            top = row[x] if x < w else None
            steep = bool(steep_row[x]) if x < w else None

            # Проверяем, есть ли трасса в этой колонке
            has_road = road_row[x - 1] if x <= w else False

            if top != cur_top or steep != cur_steep or x == w or has_road:
                # Если есть трасса — заполняем колонку до уровня трассы
                if has_road:
                    road_y = int(pixel_y.get((z, x - 1), Y_END))
                    # Заполняем до уровня трассы - 1 блок (чтобы трасса была сверху)
                    cur_top = road_y - 1
                    cur_steep = False  # под трассой не крутой склон

                    if x < w:
                        start = x
                        # Для следующей колонки берём нормальную высоту
                        cur_top = top
                        cur_steep = bool(steep_row[x])
                    continue

                x1, x2 = start, x - 1
                mc_x1 = x1 + offset_x
                mc_x2 = x2 + offset_x
                mc_z = z + offset_z

                dirt_layers = STEEP_DIRT_LAYERS if cur_steep else DIRT_LAYERS

                # Если колонка слишком низкая — пропускаем
                if cur_top <= base_y + 1:
                    if x < w:
                        start = x
                        cur_top = top
                        cur_steep = steep
                    continue

                # Высота колонки
                column_height = cur_top - base_y + 1

                # Определяем толщину каменной "стены"
                if column_height <= SHELL_THICKNESS:
                    # Низкая колонка — полностью заполняем (корка = вся колонка)
                    stone_thickness = column_height - 1 - dirt_layers
                    is_hollow = False
                else:
                    # Высокая колонка — полая внутри
                    stone_thickness = SHELL_THICKNESS - 1 - dirt_layers
                    is_hollow = True

                stone_thickness = max(0, stone_thickness)

                seg_start = mc_x1
                while seg_start <= mc_x2:
                    seg_end = min(seg_start + max_fill_len - 1, mc_x2)

                    # 1. Поверхность (рандом: 85% трава, 15% камень)
                    for bx in range(seg_start, seg_end + 1):
                        rand = random.random()
                        if rand < 0.85:
                            top_block = GRASS_BLOCK
                        else:
                            top_block = STONE_BLOCK
                        commands.append(f"setblock {bx} {int(cur_top)} {mc_z} {top_block}")

                    # 2. Слой грязи
                    dirt_top = cur_top - 1
                    dirt_bottom = cur_top - dirt_layers
                    if dirt_top >= dirt_bottom and dirt_bottom >= base_y + 1:
                        commands.append(
                            f"fill {seg_start} {int(dirt_bottom)} {mc_z} {seg_end} {int(dirt_top)} {mc_z} {DIRT_BLOCK}")

                    # 3. Каменная "стена"
                    stone_top = dirt_bottom - 1
                    stone_bottom = stone_top - stone_thickness + 1
                    if stone_thickness > 0 and stone_top >= base_y + 1:
                        stone_bottom = max(stone_bottom, base_y + 1)
                        if stone_bottom <= stone_top:
                            commands.append(
                                f"fill {seg_start} {int(stone_bottom)} {mc_z} {seg_end} {int(stone_top)} {mc_z} {STONE_BLOCK}")

                    # 4. Каменный "пол" на уровне base_y (только если колонка высокая)
                    if is_hollow:
                        commands.append(
                            f"fill {seg_start} {int(base_y)} {mc_z} {seg_end} {int(base_y)} {mc_z} {STONE_BLOCK}")

                    seg_start = seg_end + 1

                start = x
                cur_top = top
                cur_steep = steep

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