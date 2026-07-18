import json
import os
import shutil

from config import (
    NAMESPACE, PACK_FORMAT, DATAPACK_DIR, BLOCKS_PER_TICK,
    ROAD_BLOCK, START_X, START_Z, Y_START,
)


def create_datapack(commands, y_start=Y_START, all_commands=None, road_commands=None):
    """Создаёт датапак."""
    if os.path.exists(DATAPACK_DIR):
        shutil.rmtree(DATAPACK_DIR)

    func_dir = os.path.join(DATAPACK_DIR, "data", NAMESPACE, "function")
    os.makedirs(func_dir, exist_ok=True)

    # ── pack.mcmeta ───────────────────────────────────────────────────────────
    with open(os.path.join(DATAPACK_DIR, "pack.mcmeta"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(
            {"pack": {"pack_format": PACK_FORMAT, "description": "Akina Auto-Build"}},
            ensure_ascii=False,
        ))

    # ── test.mcfunction ───────────────────────────────────────────────────────
    with open(os.path.join(func_dir, "test.mcfunction"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f'tellraw @a [{{"text":"[TEST] Origin X={START_X} Y={y_start} Z={START_Z}","color":"yellow"}}]\n')
        f.write(f"fill {START_X} {y_start} {START_Z} {START_X} {y_start} {START_Z} {ROAD_BLOCK}\n")

    # ── build_N.mcfunction — части команд ────────────────────────────────────
    chunks = [commands[i:i + BLOCKS_PER_TICK] for i in range(0, len(commands), BLOCKS_PER_TICK)]
    total_parts = len(chunks)

    for part_num, chunk in enumerate(chunks):
        with open(os.path.join(func_dir, f"build_{part_num}.mcfunction"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(chunk))
            f.write("\n")
            if part_num < total_parts - 1:
                f.write(f"schedule function {NAMESPACE}:build_{part_num + 1} 1t\n")
            else:
                f.write(f'tellraw @a [{{"text":"[AKINA] Готово!","color":"gold","bold":true}}]\n')

    # ── start_build.mcfunction ────────────────────────────────────────────────
    start_msg = json.dumps([{"text": "[AKINA] Начинаю строительство...", "color": "green", "bold": True}])
    with open(os.path.join(func_dir, "start_build.mcfunction"), "w", encoding="utf-8", newline="\n") as f:
        f.write(f"tellraw @a {start_msg}\n")
        f.write("gamerule maxCommandChainLength 10000000\n")
        f.write(f"function {NAMESPACE}:build_0\n")

    # ─ clear_build.mcfunction — быстрый снос ───────────────────────────────
    if all_commands:
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for cmd in all_commands:
            if cmd.startswith('fill') or cmd.startswith('setblock'):
                parts = cmd.split()
                if cmd.startswith('fill'):
                    coords = [int(parts[1]), int(parts[2]), int(parts[3]),
                              int(parts[4]), int(parts[5]), int(parts[6])]
                else:  # setblock
                    coords = [int(parts[1]), int(parts[2]), int(parts[3])] * 2

                min_x = min(min_x, coords[0], coords[3])
                max_x = max(max_x, coords[0], coords[3])
                min_y = min(min_y, coords[1], coords[4])
                max_y = max(max_y, coords[1], coords[4])
                min_z = min(min_z, coords[2], coords[5])
                max_z = max(max_z, coords[2], coords[5])

        min_x -= 5
        max_x += 5
        min_y -= 5
        max_y += 5
        min_z -= 5
        max_z += 5

        with open(os.path.join(func_dir, "clear_build.mcfunction"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f'tellraw @a [{{"text":"[AKINA] Снос постройки...","color":"red","bold":true}}]\n')
            f.write(f"fill {min_x} {min_y} {min_z} {max_x} {max_y} {max_z} air\n")
            f.write(f'tellraw @a [{{"text":"[AKINA] Готово!","color":"green","bold":true}}]\n')

        print(f"✓ Команда сноса: /function {NAMESPACE}:clear_build")
        print(f"  Bounding box: X[{min_x}..{max_x}] Y[{min_y}..{max_y}] Z[{min_z}..{max_z}]")

    # ── rebuild_road.mcfunction — перестройка только трассы ──────────────
    if road_commands:
        road_chunks = [road_commands[i:i + BLOCKS_PER_TICK] for i in range(0, len(road_commands), BLOCKS_PER_TICK)]
        total_road_parts = len(road_chunks)

        for part_num, chunk in enumerate(road_chunks):
            with open(os.path.join(func_dir, f"rebuild_road_{part_num}.mcfunction"), "w", encoding="utf-8",
                      newline="\n") as f:
                f.write("\n".join(chunk))
                f.write("\n")
                if part_num < total_road_parts - 1:
                    f.write(f"schedule function {NAMESPACE}:rebuild_road_{part_num + 1} 1t\n")
                else:
                    f.write(f'tellraw @a [{{"text":"[AKINA] Трасса перестроена!","color":"green","bold":true}}]\n')

        with open(os.path.join(func_dir, "rebuild_road.mcfunction"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f'tellraw @a [{{"text":"[AKINA] Начинаю перестройку трассы...","color":"yellow","bold":true}}]\n')
            f.write("gamerule maxCommandChainLength 10000000\n")
            f.write(f"function {NAMESPACE}:rebuild_road_0\n")

        print(f"✓ Команда перестройки трассы: /function {NAMESPACE}:rebuild_road")
        print(f"  Частей трассы: {total_road_parts}")

    # ── clear_road.mcfunction — снос только трассы ────────────────────────
    if road_commands:
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for cmd in road_commands:
            if cmd.startswith('fill'):
                parts = cmd.split()
                coords = [int(parts[1]), int(parts[2]), int(parts[3]),
                          int(parts[4]), int(parts[5]), int(parts[6])]

                min_x = min(min_x, coords[0], coords[3])
                max_x = max(max_x, coords[0], coords[3])
                min_y = min(min_y, coords[1], coords[4])
                max_y = max(max_y, coords[1], coords[4])
                min_z = min(min_z, coords[2], coords[5])
                max_z = max(max_z, coords[2], coords[5])

        min_x -= 2
        max_x += 2
        min_y -= 2
        max_y += 2
        min_z -= 2
        max_z += 2

        with open(os.path.join(func_dir, "clear_road.mcfunction"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f'tellraw @a [{{"text":"[AKINA] Снос трассы...","color":"red","bold":true}}]\n')
            f.write(f"fill {min_x} {min_y} {min_z} {max_x} {max_y} {max_z} air\n")
            f.write(f'tellraw @a [{{"text":"[AKINA] Трасса снесена!","color":"green","bold":true}}]\n')

        print(f"✓ Команда сноса трассы: /function {NAMESPACE}:clear_road")
        print(f"  Bounding box трассы: X[{min_x}..{max_x}] Y[{min_y}..{max_y}] Z[{min_z}..{max_z}]")

    print(f"✓ Всего команд: {len(commands)}, частей: {total_parts}")
    print(f"✓ В игре: /reload → /function akina:test → /function akina:start_build")