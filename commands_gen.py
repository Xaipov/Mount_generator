import numpy as np

from config import ROAD_BLOCK, SLAB_BLOCK, Y_END


def generate_commands(mask, pixel_y, pixel_slab, offset_x, offset_z):
    """Генерирует список Minecraft /fill команд по маске трассы с учётом высоты и slab-зон."""
    commands = []

    for z in range(mask.shape[0]):
        white_x = np.where(mask[z] > 128)[0]
        if len(white_x) == 0:
            continue

        start = white_x[0]
        prev = white_x[0]
        # ВАЖНО: приводим к int, чтобы не было -41.0
        base_y = int(pixel_y.get((z, int(white_x[0])), Y_END))
        is_slab = pixel_slab.get((z, int(white_x[0])), False)
        extended = np.append(white_x, [white_x[-1] + 2])

        for idx in range(1, len(extended)):
            x = int(extended[idx])
            # ВАЖНО: приводим к int
            cur_y = int(pixel_y.get((z, x), Y_END))
            cur_slab = pixel_slab.get((z, x), False)

            if x != prev + 1 or cur_y != base_y or cur_slab != is_slab:
                block = SLAB_BLOCK if is_slab else ROAD_BLOCK
                commands.append(
                    f"fill {start + offset_x} {base_y} {z + offset_z} "
                    f"{prev + offset_x} {base_y} {z + offset_z} {block}"
                )
                if x < mask.shape[1]:
                    start = x
                    prev = x
                    base_y = cur_y
                    is_slab = cur_slab
            else:
                prev = x

    return commands


def count_total_blocks(commands):
    """Считает суммарное количество блоков по всем fill-командам."""
    total = 0
    for c in commands:
        if not c.startswith("fill"):
            continue
        parts = c.split()
        x1, x2 = int(parts[1]), int(parts[4])
        total += abs(x2 - x1) + 1
    return total