import gzip
import random
import nbtlib
from nbtlib import Compound, Short, Int, IntArray, String, List, Byte

from mountain_config import STONE_BLOCK, DIRT_BLOCK, GRASS_BLOCK, DIRT_LAYERS, STEEP_DIRT_LAYERS
from config import Y_END

SHELL_THICKNESS = 25


def generate_schematic(mountain_y, mask, pixel_y, pixel_slab, offset_x, offset_z, y_end,
                       steep_mask=None, filename="akina.schem"):
    """
    Генерирует WorldEdit .schem файл (формат 1.13+).

    Используется:
    - В игре: //schem load akina → //paste
    - Или через WorldEdit GUI
    """
    base_y = y_end
    h, w = mountain_y.shape
    if steep_mask is None:
        steep_mask = np.zeros((h, w), dtype=bool)

    road_mask = np.zeros((h, w), dtype=bool)
    if mask is not None:
        road_mask = mask > 128

    # Находим bounding box
    min_x, max_x = 0, w - 1
    min_y = base_y
    max_y = int(mountain_y.max())
    min_z, max_z = 0, h - 1

    # Если трасса выше горы — расширяем bounding box
    if pixel_y:
        max_road_y = max(int(v) for v in pixel_y.values())
        max_y = max(max_y, max_road_y)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    length = max_z - min_z + 1

    print(f" Размеры схематика: {width}x{height}x{length}")
    print(
        f"📍 Bounding box: X[{min_x + offset_x}..{max_x + offset_x}] Y[{min_y}..{max_y}] Z[{min_z + offset_z}..{max_z + offset_z}]")

    # Палитра блоков и данные
    palette = {}
    block_data = [0] * (width * height * length)

    def get_state_id(block_name):
        if block_name not in palette:
            palette[block_name] = len(palette)
        return palette[block_name]

    def set_block(x, y, z, block_name):
        if x < 0 or x >= width or y < 0 or y >= height or z < 0 or z >= length:
            return
        state_id = get_state_id(block_name)
        # Порядок: X, Z, Y (для WorldEdit 1.13+)
        index = x + z * width + y * width * length
        block_data[index] = state_id

    # Заполняем блоки
    for z in range(h):
        for x in range(w):
            top = int(mountain_y[z, x])

            # Трасса
            if road_mask[z, x]:
                road_y = int(pixel_y.get((z, x), y_end))
                is_slab = pixel_slab.get((z, x), False)
                if is_slab:
                    block = "createdieselgenerators:asphalt_slab"
                else:
                    block = "createdieselgenerators:asphalt_block"
                set_block(x, road_y - min_y, z, block)
                continue

            # Гора
            if top <= base_y + 1:
                continue

            column_height = top - base_y + 1

            # Определяем толщину корки
            if column_height <= SHELL_THICKNESS:
                stone_thickness = column_height - 1 - DIRT_LAYERS
            else:
                stone_thickness = SHELL_THICKNESS - 1 - DIRT_LAYERS

            stone_thickness = max(0, stone_thickness)

            # Поверхность (рандом: 85% трава, 15% камень)
            rand = random.random()
            if rand < 0.85:
                top_block = GRASS_BLOCK
            else:
                top_block = STONE_BLOCK
            set_block(x, top - min_y, z, top_block)

            # Грязь
            for dy in range(1, DIRT_LAYERS + 1):
                y = top - dy
                if y >= base_y:
                    set_block(x, y - min_y, z, DIRT_BLOCK)

            # Камень (корка)
            for dy in range(DIRT_LAYERS + 1, DIRT_LAYERS + 1 + stone_thickness):
                y = top - dy
                if y >= base_y:
                    set_block(x, y - min_y, z, STONE_BLOCK)

            # Пол на base_y (если полая)
            if column_height > SHELL_THICKNESS:
                set_block(x, base_y - min_y, z, STONE_BLOCK)

    # Создаём палитру для NBT
    palette_nbt = Compound()
    for block_name, state_id in palette.items():
        palette_nbt[block_name] = Int(state_id)

    # Создаём NBT структуру
    nbt = Compound({
        'Version': Short(2),
        'Width': Short(width),
        'Height': Short(height),
        'Length': Short(length),
        'Offset': IntArray([0, 0, 0]),
        'Materials': String('Alpha'),
        'Palette': palette_nbt,
        'BlockData': IntArray(block_data),
        'TileEntities': List([]),
    })

    # Сохраняем в gzip
    nbt_file = nbtlib.File(nbt)
    with gzip.open(filename, 'wb') as f:
        nbt_file.save(f)

    print(f"✅ Схематик сохранён: {filename}")
    print(f"📦 Блоков: {sum(1 for v in block_data if v > 0):,}")
    print(f" Уникальных блоков: {len(palette)}")
    print(f"\n В игре:")
    print(f"1. Скопируй {filename} в .minecraft/schematics/")
    print(f"2. //schem load akina")
    print(f"3. //paste")