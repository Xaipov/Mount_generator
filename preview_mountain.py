import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from config import IMAGE_PATH, START_X, START_Z
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
    print(f"✓ В сериях поворотов: {series_count}, одиночные/прямые: {len(series_map)-series_count}")

    print(f"✓ Y_START={y_start}, генерирую шум...")
    noise_field = generate_noise_field(mask.shape)
    macro_noise_field = generate_macro_noise_field(mask.shape)

    print("Вычисление высоты горы (может занять время)...")
    mountain_y, steep_mask = compute_mountain_height(
        mask, skel_pts, dist_map, tree_skel, pixel_y, noise_field,
        macro_noise_field=macro_noise_field,
        side_map=side_map, series_map=series_map,
    )
    print(f"✓ Крутых (со скалой) пикселей: {steep_mask.sum()}")

    # === Превью ===
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    im0 = axes[0].imshow(mountain_y, cmap='terrain')
    axes[0].set_title('Высота горы (Y)')
    axes[0].plot(origin_x, origin_z, 'ro', markersize=6)
    plt.colorbar(im0, ax=axes[0])
    axes[0].axis('off')

    overlay = mountain_y.astype(float)
    vis_rgb = plt.cm.terrain((overlay - overlay.min()) / (overlay.max() - overlay.min()))[:, :, :3]
    road_mask = mask > 128
    vis_rgb[road_mask] = [0, 0, 0]
    axes[1].imshow(vis_rgb)
    axes[1].set_title('Гора + трасса (чёрным)')
    axes[1].plot(origin_x, origin_z, 'ro', markersize=6)
    axes[1].axis('off')

    # Лево(синий)/право(красный) + серии(жёлтым)
    side_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for (rz, rx), side in side_map.items():
        if series_map.get((rz, rx), False):
            side_vis[rz, rx] = [255, 255, 0]   # серия — жёлтый
        elif side == 'left':
            side_vis[rz, rx] = [0, 120, 255]   # лево — синий (врезка)
        else:
            side_vis[rz, rx] = [255, 60, 60]   # право — красный (обрыв)
    axes[2].imshow(side_vis)
    axes[2].set_title('Синий=лево(врезка) Красный=право(обрыв) Жёлтый=серия поворотов')
    axes[2].plot(origin_x, origin_z, 'wo', markersize=6)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('mountain_preview.png', dpi=100)
    plt.show()
    print("✓ Превью сохранено: mountain_preview.png")


if __name__ == "__main__":
    main()
