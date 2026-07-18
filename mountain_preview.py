import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from config import IMAGE_PATH, START_X, START_Z, Y_END, FLAT_START_BLOCKS, STEP_INTERVAL, SPEED_BLS
from image_processing import find_red_point, get_mask, build_skeleton
from height_calc import bfs_order, compute_y_start, compute_pixel_data
from mountain_gen import generate_noise_field, generate_macro_noise_field, compute_mountain_height
from road_direction import compute_road_direction, get_side
from turn_classifier import detect_curvature, classify_turn_series, expand_series_flag_to_road
from mountain_config import MOUNTAIN_PAD, NEAR_ROAD_RADIUS


def build_side_map(mask, skel_pts, dist_map, tree_skel):
    """Строит dict (z,x на трассе) -> 'left'/'right' для каждого пикселя трассы."""
    directions = compute_road_direction(skel_pts, dist_map)

    road_pts = np.array(np.where(mask > 128)).T
    _, nn_idx = tree_skel.query(road_pts)

    side_map = {}
    for i, (rz, rx) in enumerate(road_pts):
        ns = tuple(skel_pts[nn_idx[i]])
        direction = directions.get(ns, (0.0, 1.0))
        offset = (rz - ns[0], rx - ns[1])
        side_map[(rz, rx)] = get_side(direction, offset)

    return side_map


def visualize_full(mask, dist_map, pixel_y, pixel_slab, origin_x, origin_z, y_start,
                   max_dist, block_count, mountain_y, side_map, series_map,
                   mountain_resources=None, curvature=None, total_blocks=0):
    """
    Единое превью на 6 панелей с полной статистикой ресурсов.
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # ── Ряд 1: трасса ──────────────────────────────

    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].set_title('Маска трассы (сглаженная)')
    axes[0, 0].plot(origin_x, origin_z, 'ro', markersize=8, label='Старт')
    axes[0, 0].legend()
    axes[0, 0].axis('off')

    height_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    if pixel_y:
        y_vals = list(pixel_y.values())
        y_min, y_max = min(y_vals), max(y_vals)
        rng = max(1, y_max - y_min)
        for (rz, rx), y in pixel_y.items():
            t = (y - y_min) / rng
            height_rgb[rz, rx] = [int(255 * t), 0, int(255 * (1 - t))]
    axes[0, 1].imshow(height_rgb)
    axes[0, 1].set_title('Высота трассы: красный=высоко, синий=низко')
    axes[0, 1].plot(origin_x, origin_z, 'yo', markersize=8)
    axes[0, 1].axis('off')

    if dist_map:
        sorted_pts = sorted(dist_map.items(), key=lambda kv: kv[1])
        distances, y_values = [], []
        for (z, x), d in sorted_pts:
            y = pixel_y.get((z, x), Y_END)
            distances.append(d)
            y_values.append(y)

        axes[0, 2].fill_between(distances, y_values, min(y_values) - 2, alpha=0.2, color='steelblue')
        axes[0, 2].plot(distances, y_values, color='royalblue', linewidth=1.5)
        axes[0, 2].axhline(y_start, color='red', linestyle='--', linewidth=1, label=f'Старт Y={y_start}')
        axes[0, 2].axhline(Y_END, color='blue', linestyle='--', linewidth=1, label=f'Финиш Y={Y_END}')
        axes[0, 2].axvline(FLAT_START_BLOCKS, color='orange', linestyle=':', linewidth=1,
                           label=f'Конец плоской зоны ({FLAT_START_BLOCKS} бл.)')
        axes[0, 2].set_xlabel('Расстояние по трассе (пиксели)')
        axes[0, 2].set_ylabel('Y (Minecraft)')
        axes[0, 2].set_title('Профиль высоты трассы (вид сбоку)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].invert_yaxis()

    if max_dist > 0:
        total_sec = round(max_dist) / SPEED_BLS
        minutes = int(total_sec // 60)
        seconds = int(total_sec % 60)

        # Подсчет блоков трассы
        asphalt_blocks = sum(1 for v in pixel_slab.values() if not v) if pixel_slab else 0
        asphalt_slabs = sum(1 for v in pixel_slab.values() if v) if pixel_slab else 0

        stats = "\n".join([
            f"Длина трассы: {round(max_dist)} блоков",
            f"Высота старта: Y={y_start}",
            f"Высота финиша: Y={Y_END}",
            f"Перепад: {y_start - Y_END} блоков",
            f"Плоская зона: {FLAT_START_BLOCKS} блоков",
            f"Интервал спуска: {STEP_INTERVAL} блоков",
            f"Время (~{SPEED_BLS} бл/с): {minutes}м {seconds}с",
            f"Блоков трассы: {block_count}",
        ])
        axes[0, 2].text(
            0.02, 0.02, stats,
            transform=axes[0, 2].transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        )

        # Ресурсы
        if mountain_resources and total_blocks > 0:
            resource_stats = "\n".join([
                f"=== РЕСУРСЫ ===",
                f"Асфальт: {asphalt_blocks:,} ({asphalt_blocks // 64}ст+{asphalt_blocks % 64})",
                f"Асф. плита: {asphalt_slabs:,} ({asphalt_slabs // 64}ст+{asphalt_slabs % 64})",
                f"Камень (осн.): {mountain_resources['stone']:,}",
                f"  → {mountain_resources['stone_str']}",
                f"Земля: {mountain_resources['dirt']:,}",
                f"  → {mountain_resources['dirt_str']}",
                f"Трава: {mountain_resources['grass']:,}",
                f"  → {mountain_resources['grass_str']}",
                f"Камень (пов.): {mountain_resources['stone_surface']:,}",
                f"  → {mountain_resources['stone_surface_str']}",
                f"ИТОГО: {total_blocks:,} ({total_blocks // 64}ст+{total_blocks % 64})",
            ])
            axes[0, 2].text(
                0.02, 0.55, resource_stats,
                transform=axes[0, 2].transAxes,
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            )

    # ── Ряд 2: гора ──────────────────────────────

    im0 = axes[1, 0].imshow(mountain_y, cmap='terrain')
    axes[1, 0].set_title('Итоговая высота горы (Y)')
    axes[1, 0].plot(origin_x, origin_z, 'ro', markersize=6)
    plt.colorbar(im0, ax=axes[1, 0])
    axes[1, 0].axis('off')

    overlay = mountain_y.astype(float)
    rng2 = overlay.max() - overlay.min()
    norm = (overlay - overlay.min()) / (rng2 if rng2 > 0 else 1)
    vis_rgb = plt.cm.terrain(norm)[:, :, :3]
    road_mask = mask > 128
    vis_rgb[road_mask] = [0, 0, 0]
    axes[1, 1].imshow(vis_rgb)
    axes[1, 1].set_title('Гора + трасса (чёрным) — как будет выглядеть')
    axes[1, 1].plot(origin_x, origin_z, 'ro', markersize=6)
    axes[1, 1].axis('off')

    side_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for (rz, rx), side in side_map.items():
        if series_map.get((rz, rx), False):
            side_vis[rz, rx] = [255, 255, 0]
        elif side == 'left':
            side_vis[rz, rx] = [0, 120, 255]
        else:
            side_vis[rz, rx] = [255, 60, 60]
    axes[1, 2].imshow(side_vis)
    axes[1, 2].set_title('Синий=врезка слева  Красный=обрыв справа  Жёлтый=серия поворотов')
    axes[1, 2].plot(origin_x, origin_z, 'wo', markersize=6)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('akina_full_preview.png', dpi=100)
    plt.show()
    print("✓ Полное превью сохранено: akina_full_preview.png")


def main():
    print("Загрузка трассы...")
    mask = get_mask(IMAGE_PATH)
    red = find_red_point(IMAGE_PATH)
    origin_x, origin_z = red if red else (0, 0)

    skel = build_skeleton(mask)
    skel_pts = np.array(np.where(skel > 0)).T

    dist_map = bfs_order(skel, origin_z, origin_x)
    max_dist = max(dist_map.values())

    y_start, _, _ = compute_y_start(max_dist)
    tree_skel = cKDTree(skel_pts)
    pixel_y, pixel_slab = compute_pixel_data(mask, skel_pts, dist_map, tree_skel, y_start)

    print("Определение лево/право...")
    side_map = build_side_map(mask, skel_pts, dist_map, tree_skel)
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

    # === Превью ===
    visualize_full(
        mask, dist_map, pixel_y, pixel_slab, origin_x, origin_z, y_start,
        max_dist, 0, mountain_y, side_map, series_map,
        curvature=curvature,
    )


if __name__ == "__main__":
    main()