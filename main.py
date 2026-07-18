import numpy as np
from scipy.spatial import cKDTree
from create_schematic import generate_create_schematic

from config import (
    IMAGE_PATH, START_X, START_Z, Y_END, DATAPACK_DIR,
    FLAT_START_BLOCKS, STEP_INTERVAL, SPEED_BLS,
)
from image_processing import find_red_point, get_mask, build_skeleton
from height_calc import bfs_order, compute_y_start, compute_pixel_data
from commands_gen import generate_commands, count_total_blocks
from mountain_gen import (
    generate_noise_field, generate_macro_noise_field, compute_mountain_height,
)
from mountain_commands import generate_mountain_commands, count_mountain_resources, format_as_stacks
from road_direction import compute_road_direction, get_side
from turn_classifier import detect_curvature, classify_turn_series, expand_series_flag_to_road
from datapack_gen import create_datapack
from mountain_preview import visualize_full


def main():
    print("Загрузка и сглаживание...")
    mask = get_mask(IMAGE_PATH)
    if mask is None:
        print(f"❌ Не удалось загрузить изображение: {IMAGE_PATH}")
        return

    red = find_red_point(IMAGE_PATH)
    if red:
        origin_x, origin_z = red
        print(f"✓ Красная точка: пиксель ({origin_x},{origin_z}) → MC X={START_X} Z={START_Z}")
    else:
        origin_x, origin_z = 0, 0
        print("⚠ Красная точка не найдена, используем (0, 0)")

    print(f"✓ Пикселей трассы: {(mask > 128).sum()}")

    print("Скелетизация...")
    skel = build_skeleton(mask)
    skel_pts = np.array(np.where(skel > 0)).T

    print("BFS от старта...")
    dist_map = bfs_order(skel, origin_z, origin_x)
    max_dist = max(dist_map.values()) if dist_map else 0
    print(f"✓ Длина скелета: {round(max_dist)} блоков")

    total_sec = round(max_dist) / SPEED_BLS
    minutes = int(total_sec // 60)
    seconds = int(total_sec % 60)
    print(f"✓ Расчётное время прохождения (~{SPEED_BLS} бл/с): {minutes}м {seconds}с")

    y_start, effective_len, pairs = compute_y_start(max_dist)
    print(f"✓ Y_START: эфф.длина={effective_len}, пар={pairs} → Y_START={y_start}")

    print("Вычисление высот и slab-зон...")
    tree_skel = cKDTree(skel_pts)
    pixel_y, pixel_slab = compute_pixel_data(mask, skel_pts, dist_map, tree_skel, y_start)
    slab_count = sum(1 for v in pixel_slab.values() if v)
    block_count = len(pixel_slab) - slab_count
    print(f"✓ Пикселей со slab: {slab_count}, без: {block_count}")

    print("Генерация команд трассы...")
    offset_x = START_X - origin_x
    offset_z = START_Z - origin_z
    road_commands = generate_commands(mask, pixel_y, pixel_slab, offset_x, offset_z)
    print(f"✓ road_commands: {len(road_commands)} команд")
    print(f"✓ Блоков трассы будет поставлено: {count_total_blocks(road_commands):,}")

    print("\n=== ГЕНЕРАЦИЯ ГОРЫ ===")
    print("Определение лево/право...")
    side_map = {}
    directions = compute_road_direction(skel_pts, dist_map)
    road_pts = np.array(np.where(mask > 128)).T
    _, nn_idx = tree_skel.query(road_pts)
    for i, (rz, rx) in enumerate(road_pts):
        ns = tuple(skel_pts[nn_idx[i]])
        direction = directions.get(ns, (0.0, 1.0))
        offset = (rz - ns[0], rx - ns[1])
        side_map[(rz, rx)] = get_side(direction, offset)
    left_count = sum(1 for v in side_map.values() if v == 'left')
    print(f"✓ Left: {left_count}, Right: {len(side_map) - left_count}")

    print("Классификация поворотов (одиночные vs серии)...")
    curvature = detect_curvature(skel_pts, dist_map)
    series_flags = classify_turn_series(skel_pts, dist_map, curvature)
    series_map = expand_series_flag_to_road(mask, skel_pts, tree_skel, series_flags)
    series_count = sum(1 for v in series_map.values() if v)
    print(f"✓ В сериях поворотов: {series_count}, одиночные/прямые: {len(series_map) - series_count}")

    print(f"✓ Y_START={y_start}, генерирую шум...")
    noise_field = generate_noise_field(mask.shape)
    macro_noise_field = generate_macro_noise_field(mask.shape)

    print("Вычисление высоты горы (может занять время)...")
    mountain_y, steep_mask = compute_mountain_height(
        mask, skel_pts, dist_map, tree_skel, pixel_y, noise_field,
        macro_noise_field=macro_noise_field,
        side_map=side_map, series_map=series_map,
        curvature=curvature,
    )
    print(f"✓ Крутых (со скалой) пикселей: {steep_mask.sum()}")

    print("Генерация команд горы...")
    mountain_commands = generate_mountain_commands(
        mountain_y, offset_x, offset_z, Y_END,
        steep_mask=steep_mask, mask=mask, pixel_y=pixel_y,
    )
    print(f"✓ Команд горы: {len(mountain_commands):,}")

    # Подсчет ресурсов горы
    print("\n=== РЕСУРСЫ ДЛЯ ГОРЫ ===")
    mountain_resources = count_mountain_resources(mountain_y, steep_mask=steep_mask, mask=mask)
    print(f"  Камень (основание): {mountain_resources['stone']:,} → {mountain_resources['stone_str']}")
    print(f"  Земля: {mountain_resources['dirt']:,} → {mountain_resources['dirt_str']}")
    print(f"  Трава (85%): {mountain_resources['grass']:,} → {mountain_resources['grass_str']}")
    print(
        f"  Камень (поверхность, 15%): {mountain_resources['stone_surface']:,} → {mountain_resources['stone_surface_str']}")

    # Подсчет ресурсов трассы
    asphalt_blocks = sum(1 for v in pixel_slab.values() if not v) if pixel_slab else 0
    asphalt_slabs = sum(1 for v in pixel_slab.values() if v) if pixel_slab else 0

    asphalt_str = format_as_stacks(asphalt_blocks)
    slab_str = format_as_stacks(asphalt_slabs)

    print(f"\n=== РЕСУРСЫ ДЛЯ ТРАССЫ ===")
    print(f"  Асфальт (блок): {asphalt_blocks:,} → {asphalt_str}")
    print(f"  Асфальт (плита): {asphalt_slabs:,} → {slab_str}")

    total_blocks = (
            mountain_resources['stone'] + mountain_resources['dirt'] +
            mountain_resources['grass'] + mountain_resources['stone_surface'] +
            asphalt_blocks + asphalt_slabs
    )
    total_str = format_as_stacks(total_blocks)
    print(f"\n=== ИТОГО ВСЕХ БЛОКОВ: {total_blocks:,} → {total_str} ===")

    all_commands = mountain_commands + road_commands
    print(f"\n✓ Всего команд (гора+трасса): {len(all_commands):,}")

    create_datapack(all_commands, y_start, all_commands=all_commands, road_commands=road_commands)

    print("\n=== ВИЗУАЛИЗАЦИЯ ===")
    visualize_full(
        mask, dist_map, pixel_y, pixel_slab, origin_x, origin_z, y_start,
        max_dist, len(road_commands), mountain_y, side_map, series_map,
        mountain_resources=mountain_resources, curvature=curvature,
        total_blocks=total_blocks,
    )
    print("\n=== ГЕНЕРАЦИЯ СХЕМАТИКА (CREATE NBT) ===")

    generate_create_schematic(
        mountain_y, mask, pixel_y, pixel_slab, offset_x, offset_z, Y_END,
        steep_mask=steep_mask, filename="akina.nbt", output_dir="schematics"
    )


    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("=" * 60)
    print(f"\n📁 Датапак: {DATAPACK_DIR}")
    print("\n🎮 В игре:")
    print("1. /reload")
    print("2. /function akina:test")
    print("3. /function akina:start_build")
    print("\n🔧 Доп. команды:")
    print("   /function akina:clear_build   — снос всего (гора + трасса)")
    print("   /function akina:clear_road    — снос только трассы")
    print("   /function akina:rebuild_road  — перестройка только трассы")


if __name__ == "__main__":
    main()