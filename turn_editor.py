#!/usr/bin/env python3
"""
turn_editor.py — PySide6-редактор ручной разметки поворотов трассы Akina.

Запуск:
    python turn_editor.py

Управление холстом:
    ЛКМ              — рисовать выбранным инструментом
    ПКМ              — сбросить пиксели под кистью на авто-определение
    Средняя кн. drag — панорамирование
    Колесо мыши      — масштаб
    Ctrl+S           — сохранить
    Ctrl+Z           — отменить последнее действие

Интеграция с main.py — добавь после classify_turn_series / expand_series_flag_to_road:

    from turn_editor import apply_manual_annotations
    series_map = apply_manual_annotations(series_map)
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# ── PySide6 ──────────────────────────────────────────────────────────────────
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor, QCursor, QImage, QKeySequence, QPainter,
    QPixmap, QShortcut, QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QRadioButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

# ── Локальные модули проекта ──────────────────────────────────────────────────
from config import IMAGE_PATH
from height_calc import bfs_order, compute_pixel_data, compute_y_start
from image_processing import build_skeleton, find_red_point, get_mask
from turn_classifier import (
    classify_turn_series,
    detect_curvature,
    expand_series_flag_to_road,
)

# ── Константы ─────────────────────────────────────────────────────────────────
SAVE_PATH = "turn_annotations.json"
MAX_UNDO  = 60       # глубина отмены

# RGBA-цвета слоя аннотаций
_A = 210
COLORS: dict[str, tuple[int, int, int, int]] = {
    "auto":     ( 90,  90,  90, 150),  # серый      — авто без вмешательства
    "series":   (255, 210,   0, _A),   # жёлтый     — серия поворотов (шикана)
    "turn":     (255,  55,  55, _A),   # красный     — одиночный поворот
    "straight": ( 30, 200,  60, _A),   # зелёный     — прямая / без обрыва
}
SKEL_COLOR   = (  0, 230,   0, 255)   # зелёный — скелет
ORIGIN_COLOR = (255,   0,   0, 255)   # красный — точка старта


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательная функция (можно импортировать без Qt)
# ══════════════════════════════════════════════════════════════════════════════

def apply_manual_annotations(
    auto_series_map: dict,
    path: str = SAVE_PATH,
) -> dict:
    """
    Загружает ручную разметку из JSON и применяет поверх авто-карты.

    Возвращает итоговый series_map: (z, x) -> bool.
    Если файл не найден — возвращает auto_series_map без изменений.
    """
    if not Path(path).exists():
        return auto_series_map

    with open(path, encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)

    result = dict(auto_series_map)
    overridden = 0
    for key_str, mode in raw.items():
        z, x = map(int, key_str.split(","))
        key = (z, x)
        if mode == "series":
            result[key] = True
            overridden += 1
        elif mode in ("turn", "straight"):
            result[key] = False
            overridden += 1
        # mode == "auto" → оставляем авто-значение
    print(f"[turn_editor] применено ручных аннотаций: {overridden}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Canvas — виджет холста
# ══════════════════════════════════════════════════════════════════════════════

class Canvas(QLabel):
    """
    Интерактивный холст: отображение трассы + рисование аннотаций.

    Координатная система:
      img_col = x (пиксель по горизонтали)
      img_row = z (пиксель по вертикали)
    Ключи словарей: (z, x) — как в mask, pixel_y и т.д.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── Данные трассы ───────────────────────────────────────────────────
        self._bg: QImage | None = None         # фоновое изображение маски
        self._road_pts: np.ndarray | None = None   # (N, 2) массив [z, x]
        self._road_tree: cKDTree | None = None
        self._skel_pts: np.ndarray | None = None
        self._origin_x = 0
        self._origin_z = 0

        # ── Аннотации ───────────────────────────────────────────────────────
        self.auto_series: dict[tuple[int, int], bool] = {}
        self.manual: dict[tuple[int, int], str] = {}   # 'series'|'turn'|'straight'|'auto'

        # ── Отмена ──────────────────────────────────────────────────────────
        self._undo_stack: deque[dict] = deque(maxlen=MAX_UNDO)

        # ── Режим рисования ─────────────────────────────────────────────────
        self.brush_size: int = 8
        self.draw_mode: str = "series"
        self._drawing = False

        # ── Вид (zoom + pan) ────────────────────────────────────────────────
        self.scale: float = 1.0
        self.offset = QPoint(0, 0)
        self._last_pos = QPoint()
        self._panning = False

        # ── Кэш оверлея ─────────────────────────────────────────────────────
        self._overlay: QImage | None = None
        self._dirty = False

    # ──────────────────────────────────────────────────────────────────────────
    # Загрузка данных
    # ──────────────────────────────────────────────────────────────────────────

    def set_data(
        self,
        mask: np.ndarray,
        skel_pts: np.ndarray,
        dist_map: dict,
        auto_series_map: dict[tuple[int, int], bool],
        origin_x: int,
        origin_z: int,
    ) -> None:
        h, w = mask.shape
        self._origin_x = origin_x
        self._origin_z = origin_z
        self._skel_pts = skel_pts
        self.auto_series = auto_series_map

        # Фон: полупрозрачная маска
        bg_arr = np.zeros((h, w, 4), dtype=np.uint8)
        road_mask = mask > 128
        bg_arr[road_mask] = [55, 55, 55, 255]
        bg_arr[~road_mask] = [18, 18, 18, 255]
        self._bg = self._arr_to_qimage(bg_arr)

        road_zy = np.array(np.where(road_mask)).T  # [[z, x], ...]
        self._road_pts = road_zy
        self._road_tree = cKDTree(road_zy)

        self.manual.clear()
        self._undo_stack.clear()
        self._rebuild_overlay()

    # ──────────────────────────────────────────────────────────────────────────
    # Отрисовка оверлея
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _arr_to_qimage(arr: np.ndarray) -> QImage:
        h, w, _ = arr.shape
        return QImage(
            arr.data.tobytes(), w, h, w * 4, QImage.Format_RGBA8888
        ).copy()

    def _rebuild_overlay(self) -> None:
        if self._bg is None:
            return
        h, w = self._bg.height(), self._bg.width()
        arr = np.zeros((h, w, 4), dtype=np.uint8)

        # Дорога
        if self._road_pts is not None:
            for z, x in self._road_pts:
                key = (int(z), int(x))
                mode = self.manual.get(key, "auto")
                if mode == "auto":
                    c = COLORS["series"] if self.auto_series.get(key, False) else COLORS["auto"]
                else:
                    c = COLORS[mode]
                arr[z, x] = c

        # Скелет поверх
        if self._skel_pts is not None:
            for z, x in self._skel_pts:
                arr[int(z), int(x)] = SKEL_COLOR

        # Точка старта
        oz, ox = self._origin_z, self._origin_x
        r = 5
        z1, z2 = max(0, oz - r), min(h, oz + r + 1)
        x1, x2 = max(0, ox - r), min(w, ox + r + 1)
        arr[z1:z2, x1:x2] = ORIGIN_COLOR

        self._overlay = self._arr_to_qimage(arr)
        self.update()

    # ──────────────────────────────────────────────────────────────────────────
    # Qt paintEvent
    # ──────────────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._bg is None or self._overlay is None:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)

        cx = self.width()  / 2.0 + self.offset.x()
        cy = self.height() / 2.0 + self.offset.y()
        iw = int(self._bg.width()  * self.scale)
        ih = int(self._bg.height() * self.scale)
        x0 = int(cx - iw / 2)
        y0 = int(cy - ih / 2)

        p.fillRect(0, 0, self.width(), self.height(), QColor(18, 18, 18))
        p.drawImage(QRect(x0, y0, iw, ih), self._bg)
        p.drawImage(QRect(x0, y0, iw, ih), self._overlay)
        p.end()

    # ──────────────────────────────────────────────────────────────────────────
    # Координатные преобразования
    # ──────────────────────────────────────────────────────────────────────────

    def _widget_to_img(self, pos: QPoint) -> tuple[int, int]:
        """Виджет-координаты → (img_col=x, img_row=z)."""
        if self._bg is None:
            return 0, 0
        cx = self.width()  / 2.0 + self.offset.x()
        cy = self.height() / 2.0 + self.offset.y()
        iw = self._bg.width()  * self.scale
        ih = self._bg.height() * self.scale
        ix = (pos.x() - (cx - iw / 2)) / self.scale
        iz = (pos.y() - (cy - ih / 2)) / self.scale
        return int(ix), int(iz)

    # ──────────────────────────────────────────────────────────────────────────
    # Рисование
    # ──────────────────────────────────────────────────────────────────────────

    def _paint_at(self, widget_pos: QPoint, mode: str) -> None:
        if self._road_tree is None or self._road_pts is None:
            return
        ix, iz = self._widget_to_img(widget_pos)
        radius = max(1.0, self.brush_size / self.scale)
        idxs = self._road_tree.query_ball_point([iz, ix], radius)
        if not idxs:
            return

        # Снимок для undo перед изменением
        snapshot = {k: v for k, v in self.manual.items()
                    if tuple(self._road_pts[i]) in
                    [(int(self._road_pts[j][0]), int(self._road_pts[j][1])) for j in idxs]}
        self._undo_stack.append(dict(self.manual))  # полный снимок (проще)

        changed = False
        for i in idxs:
            key = (int(self._road_pts[i][0]), int(self._road_pts[i][1]))
            if self.manual.get(key) != mode:
                self.manual[key] = mode
                changed = True
        if changed:
            self._rebuild_overlay()

    # ──────────────────────────────────────────────────────────────────────────
    # Обработка мыши
    # ──────────────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if self._bg is None:
            return
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self._paint_at(event.pos(), self.draw_mode)
        elif event.button() == Qt.RightButton:
            self._drawing = True
            self._paint_at(event.pos(), "auto")
        elif event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pos = event.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._bg is None:
            return
        if self._drawing:
            if event.buttons() & Qt.LeftButton:
                self._paint_at(event.pos(), self.draw_mode)
            elif event.buttons() & Qt.RightButton:
                self._paint_at(event.pos(), "auto")
        if self._panning:
            delta = event.pos() - self._last_pos
            self.offset += delta
            self._last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drawing = False
        self._panning = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        new_scale = max(0.08, min(30.0, self.scale * factor))

        # Зумим к позиции курсора
        pos = event.position()
        cx = self.width()  / 2.0 + self.offset.x()
        cy = self.height() / 2.0 + self.offset.y()
        # Корректируем offset, чтобы точка под курсором не уехала
        ratio = new_scale / self.scale
        dx = (pos.x() - cx) * (ratio - 1)
        dy = (pos.y() - cy) * (ratio - 1)
        self.offset -= QPoint(int(dx), int(dy))
        self.scale = new_scale
        self.update()

    # ──────────────────────────────────────────────────────────────────────────
    # Undo / Redo
    # ──────────────────────────────────────────────────────────────────────────

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self.manual = self._undo_stack.pop()
        self._rebuild_overlay()

    # ──────────────────────────────────────────────────────────────────────────
    # Fit / Reset
    # ──────────────────────────────────────────────────────────────────────────

    def fit_to_window(self) -> None:
        if self._bg is None:
            return
        sw = (self.width()  - 20) / self._bg.width()
        sh = (self.height() - 20) / self._bg.height()
        self.scale = max(0.08, min(sw, sh, 4.0))
        self.offset = QPoint(0, 0)
        self.update()

    def reset_manual(self) -> None:
        self._undo_stack.append(dict(self.manual))
        self.manual.clear()
        self._rebuild_overlay()

    # ──────────────────────────────────────────────────────────────────────────
    # Сохранение / загрузка
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str = SAVE_PATH) -> str:
        data = {f"{z},{x}": v for (z, x), v in self.manual.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def load(self, path: str = SAVE_PATH) -> None:
        if not Path(path).exists():
            return
        with open(path, encoding="utf-8") as f:
            raw: dict[str, str] = json.load(f)
        self._undo_stack.append(dict(self.manual))
        self.manual = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v
            for k, v in raw.items()
        }
        self._rebuild_overlay()

    # ──────────────────────────────────────────────────────────────────────────
    # Итоговая карта
    # ──────────────────────────────────────────────────────────────────────────

    def merged_series_map(self) -> dict[tuple[int, int], bool]:
        """Авто-карта + ручные переопределения → итоговый series_map."""
        result = dict(self.auto_series)
        for key, mode in self.manual.items():
            if mode == "series":
                result[key] = True
            elif mode in ("turn", "straight"):
                result[key] = False
            # mode == "auto" → ничего не меняем
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Статистика
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        total = len(self._road_pts) if self._road_pts is not None else 0
        manual_series   = sum(1 for v in self.manual.values() if v == "series")
        manual_turn     = sum(1 for v in self.manual.values() if v == "turn")
        manual_straight = sum(1 for v in self.manual.values() if v == "straight")
        merged = self.merged_series_map()
        total_series = sum(1 for v in merged.values() if v)
        return {
            "total":           total,
            "manual_series":   manual_series,
            "manual_turn":     manual_turn,
            "manual_straight": manual_straight,
            "total_series":    total_series,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MainWindow
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Akina — Turn Editor")
        self.resize(1400, 860)
        self._build_ui()
        # Загружаем трассу после того, как окно отрисовалось
        QTimer.singleShot(80, self._load_track)

    # ──────────────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # ── Холст ────────────────────────────────────────────────────────────
        self.canvas = Canvas()
        outer.addWidget(self.canvas, stretch=1)

        # ── Боковая панель ────────────────────────────────────────────────────
        panel = QWidget()
        panel.setFixedWidth(235)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        pv = QVBoxLayout(panel)
        pv.setAlignment(Qt.AlignTop)
        pv.setSpacing(8)
        outer.addWidget(panel)

        # ── Инструменты ───────────────────────────────────────────────────────
        grp_tools = QGroupBox("Инструмент  (ЛКМ рисует, ПКМ — авто)")
        gv = QVBoxLayout(grp_tools)
        self._rb_group = QButtonGroup(self)
        tools = [
            ("series",   "🟡  Серия поворотов  (шикана)"),
            ("turn",     "🔴  Одиночный поворот"),
            ("straight", "🟢  Прямая / нет обрыва"),
            ("auto",     "⬛  Авто  (убрать ручное)"),
        ]
        self._rbs: dict[str, QRadioButton] = {}
        for mode, label in tools:
            rb = QRadioButton(label)
            self._rb_group.addButton(rb)
            gv.addWidget(rb)
            self._rbs[mode] = rb
        self._rbs["series"].setChecked(True)
        self._rb_group.buttonClicked.connect(self._on_tool_click)
        pv.addWidget(grp_tools)

        # ── Размер кисти ──────────────────────────────────────────────────────
        grp_brush = QGroupBox("Кисть")
        bv = QVBoxLayout(grp_brush)
        self._lbl_brush = QLabel("8 пикс.")
        self._sld_brush = QSlider(Qt.Horizontal)
        self._sld_brush.setRange(1, 60)
        self._sld_brush.setValue(8)
        self._sld_brush.valueChanged.connect(self._on_brush_change)
        bv.addWidget(self._lbl_brush)
        bv.addWidget(self._sld_brush)
        pv.addWidget(grp_brush)

        # ── Легенда ───────────────────────────────────────────────────────────
        grp_leg = QGroupBox("Легенда")
        lv = QVBoxLayout(grp_leg)
        leg_text = (
            "🟡 Жёлтый  — серия (шикана)\n"
            "🔴 Красный — одиночный поворот\n"
            "🟢 Зелёный — прямая/без обрыва\n"
            "⬛ Серый   — авто-определение\n"
            "━━  Зелёная линия — скелет\n"
            "●   Красная точка — старт"
        )
        lbl_leg = QLabel(leg_text)
        lbl_leg.setWordWrap(True)
        lv.addWidget(lbl_leg)
        pv.addWidget(grp_leg)

        # ── Действия ──────────────────────────────────────────────────────────
        grp_act = QGroupBox("Действия")
        av = QVBoxLayout(grp_act)

        btn_save = QPushButton("💾  Сохранить  (Ctrl+S)")
        btn_save.clicked.connect(self._on_save)
        av.addWidget(btn_save)

        btn_load = QPushButton("📂  Загрузить разметку")
        btn_load.clicked.connect(self._on_load)
        av.addWidget(btn_load)

        btn_undo = QPushButton("↩  Отменить  (Ctrl+Z)")
        btn_undo.clicked.connect(self.canvas.undo)
        av.addWidget(btn_undo)

        btn_reset = QPushButton("♻️  Сбросить всё на авто")
        btn_reset.clicked.connect(self._on_reset)
        av.addWidget(btn_reset)

        btn_fit = QPushButton("🔍  По размеру окна")
        btn_fit.clicked.connect(self.canvas.fit_to_window)
        av.addWidget(btn_fit)

        pv.addWidget(grp_act)

        # ── Статус / подсказки ────────────────────────────────────────────────
        self._lbl_status = QLabel("⏳ Загрузка трассы...")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        pv.addWidget(self._lbl_status)

        pv.addStretch(1)

        # ── Подсказки управления ──────────────────────────────────────────────
        hint = QLabel(
            "<small>"
            "Колесо — масштаб к курсору<br>"
            "Ср.кн.+drag — панорама<br>"
            "ПКМ — авто (стереть ручное)"
            "</small>"
        )
        hint.setWordWrap(True)
        pv.addWidget(hint)

        # ── Горячие клавиши ───────────────────────────────────────────────────
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._on_save)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.canvas.undo)
        QShortcut(QKeySequence("1"), self).activated.connect(
            lambda: self._select_tool("series"))
        QShortcut(QKeySequence("2"), self).activated.connect(
            lambda: self._select_tool("turn"))
        QShortcut(QKeySequence("3"), self).activated.connect(
            lambda: self._select_tool("straight"))
        QShortcut(QKeySequence("4"), self).activated.connect(
            lambda: self._select_tool("auto"))
        QShortcut(QKeySequence("F"), self).activated.connect(
            self.canvas.fit_to_window)

    # ──────────────────────────────────────────────────────────────────────────
    # Загрузка трассы
    # ──────────────────────────────────────────────────────────────────────────

    def _load_track(self) -> None:
        self._lbl_status.setText("⏳ Загрузка изображения...")
        QApplication.processEvents()

        try:
            mask = get_mask(IMAGE_PATH)
            if mask is None:
                self._lbl_status.setText(f"❌ Файл не найден:\n{IMAGE_PATH}")
                return

            self._lbl_status.setText("⏳ Скелетизация и BFS...")
            QApplication.processEvents()

            red = find_red_point(IMAGE_PATH)
            ox, oz = (red if red else (0, 0))

            skel = build_skeleton(mask)
            skel_pts = np.array(np.where(skel > 0)).T

            dist_map = bfs_order(skel, oz, ox)
            max_dist = max(dist_map.values()) if dist_map else 0

            self._lbl_status.setText("⏳ Вычисление высот...")
            QApplication.processEvents()

            y_start, _, _ = compute_y_start(max_dist)
            tree_skel = cKDTree(skel_pts)
            pixel_y, _ = compute_pixel_data(mask, skel_pts, dist_map, tree_skel, y_start)

            self._lbl_status.setText("⏳ Авто-классификация поворотов...")
            QApplication.processEvents()

            curvature = detect_curvature(skel_pts, dist_map)
            series_flags = classify_turn_series(skel_pts, dist_map, curvature)
            auto_map = expand_series_flag_to_road(mask, skel_pts, tree_skel, series_flags)

            self.canvas.set_data(mask, skel_pts, dist_map, auto_map, ox, oz)
            self.canvas.fit_to_window()

            # Автозагрузка сохранённой разметки
            if Path(SAVE_PATH).exists():
                self.canvas.load(SAVE_PATH)
                note = f"📂 Загружена разметка: {SAVE_PATH}"
            else:
                note = "Разметка не найдена, используется авто"

            road_px = int((mask > 128).sum())
            auto_series_pct = (
                100 * sum(1 for v in auto_map.values() if v) / max(1, road_px)
            )
            self._update_status(
                f"✅ Трасса загружена\n"
                f"Длина: {round(max_dist)} бл. | Y_start: {y_start}\n"
                f"Авто-серий: {auto_series_pct:.1f}% пикселей трассы\n"
                f"{note}"
            )

        except Exception as exc:
            self._lbl_status.setText(f"❌ Ошибка загрузки:\n{exc}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Обработчики
    # ──────────────────────────────────────────────────────────────────────────

    def _on_tool_click(self) -> None:
        for mode, rb in self._rbs.items():
            if rb.isChecked():
                self.canvas.draw_mode = mode
                break

    def _select_tool(self, mode: str) -> None:
        self._rbs[mode].setChecked(True)
        self.canvas.draw_mode = mode

    def _on_brush_change(self, val: int) -> None:
        self.canvas.brush_size = val
        self._lbl_brush.setText(f"{val} пикс.")

    def _on_save(self) -> None:
        path = self.canvas.save()
        st = self.canvas.stats()
        self._update_status(
            f"💾 Сохранено: {path}\n"
            f"Ручных: серий={st['manual_series']}, "
            f"поворотов={st['manual_turn']}, "
            f"прямых={st['manual_straight']}\n"
            f"Итого серий: {st['total_series']} пикс. из {st['total']}"
        )

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить разметку", ".", "JSON (*.json)"
        )
        if path:
            self.canvas.load(path)
            self._update_status(f"📂 Загружено: {path}")

    def _on_reset(self) -> None:
        self.canvas.reset_manual()
        self._update_status("♻️ Ручная разметка сброшена (авто)")

    def _update_status(self, text: str) -> None:
        self._lbl_status.setText(text)


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Тёмная тема через QPalette
    from PySide6.QtGui import QPalette
    pal = QPalette()
    bg  = QColor(30, 30, 30)
    mid = QColor(50, 50, 50)
    fg  = QColor(220, 220, 220)
    acc = QColor(60, 140, 240)
    pal.setColor(QPalette.Window,          bg)
    pal.setColor(QPalette.WindowText,      fg)
    pal.setColor(QPalette.Base,            mid)
    pal.setColor(QPalette.AlternateBase,   bg)
    pal.setColor(QPalette.Text,            fg)
    pal.setColor(QPalette.Button,          mid)
    pal.setColor(QPalette.ButtonText,      fg)
    pal.setColor(QPalette.Highlight,       acc)
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
