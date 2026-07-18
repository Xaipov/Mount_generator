import matplotlib.pyplot as plt
import numpy as np

from config import Y_END, FLAT_START_BLOCKS, STEP_INTERVAL, SPEED_BLS


def visualize(mask, dist_map, pixel_y, pixel_slab, origin_x, origin_z,
              y_start, max_dist=0, block_count=0):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Маска трассы
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Маска трассы (сглаженная)')
    axes[0].plot(origin_x, origin_z, 'ro', markersize=8, label='Старт')
    axes[0].legend()
    axes[0].axis('off')

    # 2. Градиент высоты (красный = высоко, синий = низко)
    height_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    if pixel_y:
        y_vals = list(pixel_y.values())
        y_min, y_max = min(y_vals), max(y_vals)
        rng = max(1, y_max - y_min)
        for (rz, rx), y in pixel_y.items():
            t = (y - y_min) / rng
            height_rgb[rz, rx] = [int(255 * t), 0, int(255 * (1 - t))]
    axes[1].imshow(height_rgb)
    axes[1].set_title('Высота: красный = высоко, синий = низко')
    axes[1].plot(origin_x, origin_z, 'yo', markersize=8, label='Старт')
    axes[1].legend()
    axes[1].axis('off')

    # 3. Профиль высоты (вид сбоку) + статистика
    if dist_map:
        sorted_pts = sorted(dist_map.items(), key=lambda kv: kv[1])
        distances, y_values = [], []
        for (z, x), d in sorted_pts:
            y = pixel_y.get((z, x), Y_END)
            distances.append(d)
            y_values.append(y)

        axes[2].fill_between(distances, y_values, min(y_values) - 2, alpha=0.2, color='steelblue')
        axes[2].plot(distances, y_values, color='royalblue', linewidth=1.5)
        axes[2].axhline(y_start, color='red', linestyle='--', linewidth=1, label=f'Старт Y={y_start}')
        axes[2].axhline(Y_END, color='blue', linestyle='--', linewidth=1, label=f'Финиш Y={Y_END}')
        axes[2].axvline(FLAT_START_BLOCKS, color='orange', linestyle=':', linewidth=1,
                         label=f'Конец плоской зоны ({FLAT_START_BLOCKS} бл.)')
        axes[2].set_xlabel('Расстояние по трассе (пиксели)')
        axes[2].set_ylabel('Y (Minecraft)')
        axes[2].set_title('Профиль высоты (вид сбоку)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].invert_yaxis()

    if max_dist > 0:
        total_sec = round(max_dist) / SPEED_BLS
        minutes = int(total_sec // 60)
        seconds = int(total_sec % 60)
        stats = "\n".join([
            f"Длина трассы: {round(max_dist)} блоков",
            f"Высота старта: Y={y_start}",
            f"Высота финиша: Y={Y_END}",
            f"Перепад: {y_start - Y_END} блоков",
            f"Плоская зона: {FLAT_START_BLOCKS} блоков",
            f"Интервал спуска: {STEP_INTERVAL} блоков",
            f"Время прохождения (~{SPEED_BLS} бл/с): {minutes}м {seconds}с",
            f"Блоков затрачено: {block_count}",
        ])
        axes[2].text(
            0.02, 0.02, stats,
            transform=axes[2].transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig('akina_preview.png', dpi=100)
    plt.show()
    print("✓ Превью сохранено: akina_preview.png")