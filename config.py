# === НАСТРОЙКИ ===

IMAGE_PATH = "akina_clean.png"
ROAD_BLOCK = "createdieselgenerators:asphalt_block"
SLAB_BLOCK = "createdieselgenerators:asphalt_slab"



Y_END   = -58   # Y у конца трассы (финиш)
Y_START = 0     # считается автоматически, не менять вручную
WORLD_MIN_Y = -58  # минимальная высота мира; поменяй, если у тебя расширенный мир

FLAT_START_BLOCKS = 32   # плоская зона у старта (блоков)
STEP_INTERVAL     = 16   # каждые N блоков по скелету — чередование block/slab

START_X = 0
START_Z = 0

NAMESPACE    = "akina"
PACK_FORMAT  = 48          # Minecraft 1.21 / 1.21.1
DATAPACK_DIR = "akina_datapack"
BLOCKS_PER_TICK  = 200
PROGRESS_EVERY   = 5
SMOOTHING_ITERATIONS = 0

SPEED_BLS = 18  # средняя скорость игрока (блоков/сек), используется для расчёта времени
