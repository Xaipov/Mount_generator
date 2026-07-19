import numpy as np
import random
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from mountain_config import (
    FLAT_SHELL_THICKNESS, STEEP_SHELL_THICKNESS, ROAD_CLEARANCE
)
from config import Y_END

DIRT_SHOULDER_WIDTH = 4


def _get_interior_mask(road_mask):
    """
    Возвращает маску пикселей внутри петель трассы.
    Логика: заливаем снаружи (от краёв холста), всё что не залито и не дорога — внутри.
    """
    h, w = road_mask.shape
    # Бинарная маска: 1 = можно идти (не дорога)
    free = ~road_mask

    # BFS/flood fill от всех краёв холста
    outside = np.zeros((h, w), dtype=bool)
    # Стартуем с граничных пикселей
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    outside[border & free] = True

    # Итеративная дилатация только по free-пикселям
    from scipy.ndimage import binary_fill_holes
    prev = None
    kernel = np.ones((3, 3), dtype=bool)
    while True:
        expanded = binary_dilation(outside, structure=kernel) & free
        if np.array_equal(expanded, outside):
            break
        outside = expanded

    # Внутри = free но не outside и не дорога
    interior = free & ~outside
    return interior


def generate_mountain_commands(mountain_y, offset_x, offset_z, y_end, max_fill_len=32,
                               steep_mask=None, mask=None, pixel_y=None):
    base_y = y_end
    h, w = mountain_y.shape
    if steep_mask is None:
        steep_mask = np.zeros((h, w), dtype=bool)

    road_mask = np.zeros((h, w), dtype=bool)
    if mask is not None:
        road_mask = mask > 128

    flat_base = int(np.mean(list(pixel_y.values()))) if pixel_y else 0
    road_y_map = np.full((h, w), flat_base, dtype=np.float32)
    if pixel_y:
        for (rz, rx), ry in pixel_y.items():
            road_y_map[rz, rx] = float(ry)

    dist_to_road, nn_indices = distance_transform_edt(~road_mask, return_indices=True)
    nearest_road_y = road_y_map[nn_indices[0], nn_indices[1]]

    shoulder_mask = (dist_to_road > 0) & (dist_to_road <= DIRT_SHOULDER_WIDTH)

    # Внутренность петель трассы — тоже dirt на уровне асфальта
    interior_mask = _get_interior_mask(road_mask)

    # Гора строится везде кроме дороги, обочины и внутренности петель
    mountain_build_mask = (~road_mask) & (~shoulder_mask) & (~interior_mask)

    commands = []
    floor_segments = []
    current_floor_seg = None

    # ── 1. Dirt-обочина (dist 1..4 от дороги) ───────────────────────────────────
    for z in range(h):
        for x in range(w):
            if not shoulder_mask[z, x]:
                continue
            road_y = int(nearest_road_y[z, x])
            mc_x = x + offset_x
            mc_z = z + offset_z
            if road_y > base_y:
                commands.append(f"fill {mc_x} {base_y} {mc_z} {mc_x} {road_y} {mc_z} minecraft:dirt")

    # ── 2. Внутренность петель — dirt на уровне ближайшего асфальта ──────────────
    for z in range(h):
        for x in range(w):
            if not interior_mask[z, x]:
                continue
            road_y = int(nearest_road_y[z, x])
            mc_x = x + offset_x
            mc_z = z + offset_z
            if road_y > base_y:
                commands.append(f"fill {mc_x} {base_y} {mc_z} {mc_x} {road_y} {mc_z} minecraft:dirt")

    # ── 3. Гора (снаружи) ────────────────────────────────────────────────────────
    for z in range(h):
        for x in range(w):
            if not mountain_build_mask[z, x]:
                continue

            top_y = int(mountain_y[z, x])
            is_steep = bool(steep_mask[z, x])

            if top_y <= base_y:
                continue

            mc_x = x + offset_x
            mc_z = z + offset_z

            # Сплошной столбик камня снизу доверху — никаких дыр
            if base_y + 1 <= top_y - 1:
                commands.append(f"fill {mc_x} {base_y + 1} {mc_z} {mc_x} {top_y - 1} {mc_z} minecraft:stone")

            surface_block = "minecraft:dirt" if random.random() > 0.15 else "minecraft:stone"
            commands.append(f"setblock {mc_x} {top_y} {mc_z} {surface_block}")

            if current_floor_seg is None:
                current_floor_seg = {'x1': mc_x, 'x2': mc_x, 'z': mc_z}
            elif current_floor_seg['z'] == mc_z and current_floor_seg['x2'] + 1 == mc_x:
                current_floor_seg['x2'] = mc_x
            else:
                floor_segments.append(current_floor_seg)
                current_floor_seg = {'x1': mc_x, 'x2': mc_x, 'z': mc_z}

    if current_floor_seg is not None:
        floor_segments.append(current_floor_seg)

    for seg in floor_segments:
        commands.append(f"fill {seg['x1']} {base_y} {seg['z']} {seg['x2']} {base_y} {seg['z']} minecraft:stone")

    return commands


def format_as_stacks(count):
    if count <= 0:
        return "0 блоков"
    stacks = count // 64
    remainder = count % 64
    if stacks == 0:
        return f"{remainder} блоков"
    elif remainder == 0:
        return f"{stacks} стак(ов)"
    else:
        return f"{stacks} стак(ов) + {remainder} блоков"


def count_mountain_resources(mountain_y, steep_mask=None, mask=None):
    base_y = Y_END
    h, w = mountain_y.shape
    if steep_mask is None:
        steep_mask = np.zeros((h, w), dtype=bool)

    road_mask = np.zeros((h, w), dtype=bool)
    if mask is not None:
        road_mask = mask > 128

    dist_to_road = distance_transform_edt(~road_mask)
    shoulder_mask = (dist_to_road > 0) & (dist_to_road <= DIRT_SHOULDER_WIDTH)
    interior_mask = _get_interior_mask(road_mask)

    stone_count = 0
    surface_count = 0
    dirt_shoulder_count = int((shoulder_mask | interior_mask).sum())

    for z in range(h):
        for x in range(w):
            if road_mask[z, x] or shoulder_mask[z, x] or interior_mask[z, x]:
                continue
            top_y = int(mountain_y[z, x])
            if top_y <= base_y:
                continue
            stone_count += max(0, top_y - 1 - base_y)
            stone_count += 1
            surface_count += 1

    grass_count = int(surface_count * 0.85)
    stone_surface_count = surface_count - grass_count
    total_stone = stone_count + stone_surface_count

    return {
        'stone': total_stone,
        'stone_str': format_as_stacks(total_stone),
        'dirt': dirt_shoulder_count,
        'dirt_str': format_as_stacks(dirt_shoulder_count),
        'grass': grass_count,
        'grass_str': format_as_stacks(grass_count),
        'stone_surface': stone_surface_count,
        'stone_surface_str': format_as_stacks(stone_surface_count),
    }