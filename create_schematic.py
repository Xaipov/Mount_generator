import random
import os

import numpy as np
import nbtlib
from nbtlib import Compound, Int, String, List

from config import Y_END
from mountain_config import STONE_BLOCK, DIRT_BLOCK, GRASS_BLOCK, DIRT_LAYERS, STEEP_DIRT_LAYERS

SHELL_THICKNESS = 25
CHUNK_SIZE = 50
DATA_VERSION = 3953  # 1.21.1


def generate_create_schematic(mountain_y, mask, pixel_y, pixel_slab, offset_x, offset_z, y_end,
                              steep_mask=None, filename="akina.nbt", output_dir="schematics"):
    """
    Генерирует .nbt файлы в ВАНИЛЬНОМ Structure-NBT формате (том же, что у
    structure-блока). Именно этот формат читает Create Schematicannon —
    его нужно положить в .minecraft/schematics/ и загрузить в Schematic Table.

    Раньше NBT собирался вручную через struct.pack в кастомную (и невалидную)
    структуру с плоским BlockData-массивом — это формат WorldEdit .schematic
    v1, Schematicannon его не понимает и/или падает при загрузке.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_y = y_end
    h, w = mountain_y.shape
    if steep_mask is None:
        steep_mask = np.zeros((h, w), dtype=bool)

    road_mask = np.zeros((h, w), dtype=bool)
    if mask is not None:
        road_mask = mask > 128

    min_x, max_x = 0, w - 1
    min_y = base_y
    max_y = int(mountain_y.max())
    min_z, max_z = 0, h - 1

    if pixel_y:
        max_road_y = max(int(v) for v in pixel_y.values())
        max_y = max(max_y, max_road_y)

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    length = max_z - min_z + 1

    print(f"📐 Общий размер: {width}x{height}x{length}")
    print(f"📁 Папка вывода: {output_dir}/")

    num_parts = (length + CHUNK_SIZE - 1) // CHUNK_SIZE

    if num_parts == 1:
        filepath = os.path.join(output_dir, filename)
        _save_create_part(
            mountain_y, mask, pixel_y, pixel_slab, steep_mask, road_mask,
            offset_x, offset_z, min_x, max_x, min_y, max_y, min_z, max_z,
            base_y, filepath
        )
    else:
        print(f"📦 Разбиваю на {num_parts} частей (по {CHUNK_SIZE} блоков по Z)...")
        for i in range(num_parts):
            part_z_start = min_z + i * CHUNK_SIZE
            part_z_end = min(part_z_start + CHUNK_SIZE - 1, max_z)

            part_filename = filename.replace('.nbt', f'_part{i + 1}.nbt')
            filepath = os.path.join(output_dir, part_filename)
            print(f"\nЧасть {i + 1}/{num_parts}: Z[{part_z_start}..{part_z_end}]")

            _save_create_part(
                mountain_y, mask, pixel_y, pixel_slab, steep_mask, road_mask,
                offset_x, offset_z, min_x, max_x, min_y, max_y, part_z_start, part_z_end,
                base_y, filepath
            )

        print(f"\n✅ Все части сохранены в: {os.path.abspath(output_dir)}/")
        print(f"🎮 В игре:")
        print(f"   1. Скопируй файлы из {output_dir}/ в .minecraft/schematics/")
        print(f"   2. Поставь Schematic Table и Schematicannon")
        print(f"   3. Загрузи akina_part1 → заправь → активируй")
        print(f"   4. Повтори для каждой части")


def _build_column(x, z, top, mountain_y, road_mask, pixel_y, pixel_slab,
                  steep_mask, base_y, blocks_dict):
    """Заполняет один столбец (x,z) в blocks_dict: (x, y_local, z_local) -> block_name."""
    if road_mask[z, x]:
        road_y = int(pixel_y.get((z, x), Y_END))
        is_slab = pixel_slab.get((z, x), False)
        block = "createdieselgenerators:asphalt_slab" if is_slab else "createdieselgenerators:asphalt_block"
        return {(x, road_y, z): block}

    if top <= base_y + 1:
        return {}

    is_steep = bool(steep_mask[z, x])
    dirt_layers = STEEP_DIRT_LAYERS if is_steep else DIRT_LAYERS

    column_height = top - base_y + 1

    if column_height <= SHELL_THICKNESS:
        stone_thickness = column_height - 1 - dirt_layers
        is_hollow = False
    else:
        stone_thickness = SHELL_THICKNESS - 1 - dirt_layers
        is_hollow = True

    stone_thickness = max(0, stone_thickness)

    out = {}
    rand = random.random()
    top_block = GRASS_BLOCK if rand < 0.85 else STONE_BLOCK
    out[(x, top, z)] = top_block

    dirt_bottom = top - dirt_layers
    for dy in range(1, dirt_layers + 1):
        y = top - dy
        if y >= base_y:
            out[(x, y, z)] = DIRT_BLOCK

    stone_top = dirt_bottom - 1
    for dy in range(dirt_layers + 1, dirt_layers + 1 + stone_thickness):
        y = top - dy
        if y >= base_y:
            out[(x, y, z)] = STONE_BLOCK

    if is_hollow:
        out[(x, base_y, z)] = STONE_BLOCK

    return out


def _save_create_part(mountain_y, mask, pixel_y, pixel_slab, steep_mask, road_mask,
                      offset_x, offset_z, min_x, max_x, min_y, max_y, part_z_start, part_z_end,
                      base_y, filepath):
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    length = part_z_end - part_z_start + 1

    # (x_local, y_local, z_local) -> block_name, только непустые (воздух не храним)
    blocks_dict = {}

    for z in range(part_z_start, part_z_end + 1):
        z_local = z - part_z_start
        for x in range(max_x + 1):
            top = int(mountain_y[z, x])
            col = _build_column(x, z, top, mountain_y, road_mask, pixel_y, pixel_slab,
                                steep_mask, base_y, {})
            for (bx, by, bz), block in col.items():
                blocks_dict[(bx, by - min_y, z_local)] = block

    if not blocks_dict:
        print(f"  ⚠ {os.path.basename(filepath)}: нет блоков, файл не создан")
        return

    # ── Палитра ────────────────────────────────────────────────────────────
    palette_names = []
    palette_index = {}
    for block_name in blocks_dict.values():
        if block_name not in palette_index:
            palette_index[block_name] = len(palette_names)
            palette_names.append(block_name)

    palette_nbt = List[Compound]([
        Compound({'Name': String(name)}) for name in palette_names
    ])

    blocks_nbt = List[Compound]([
        Compound({
            'state': Int(palette_index[block_name]),
            'pos': List[Int]([Int(x), Int(y), Int(z)]),
        })
        for (x, y, z), block_name in blocks_dict.items()
    ])

    nbt = Compound({
        'DataVersion': Int(DATA_VERSION),
        'size': List[Int]([Int(width), Int(height), Int(length)]),
        'entities': List[Compound]([]),
        'blocks': blocks_nbt,
        'palette': palette_nbt,
    })

    nbt_file = nbtlib.File(nbt)
    nbt_file.save(filepath, gzipped=True)

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"  ✅ {os.path.basename(filepath)} ({len(blocks_dict):,} блоков, "
          f"{len(palette_names)} типов, {file_size_kb:.1f} KB)")