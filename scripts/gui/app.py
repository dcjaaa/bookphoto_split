"""
图书盘点 GUI 模块（PyQt5）。

功能：
    - 启动/管理后端 FastAPI 进程
    - 选择书架照片 → 书脊分割 → OCR 书名识别 → 馆藏匹配
    - 图片预览叠加检测框，显示预测质量摘要
    - OCR 结果可编辑修正
    - 匹配结果导出 CSV

使用方式：
    python -m scripts.gui.app
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QRect, QSize
from PyQt5.QtGui import (
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QImage,
    QPalette,
    QBrush,
)
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QGridLayout,
)

from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

from scripts.gui.backend import BackendManager
from scripts.gui.client import (
    segment as api_segment,
    ocr_spine as api_ocr_spine,
    inventory as api_inventory,
    health as api_health,
    ClientError,
)
from scripts.utils.paths import GUI_OUTPUT_DIR, RAW_DIR, BOOK_LABELS_DIR
from scripts.match.inventory import evaluate_vs_ground_truth

# ---------------------------------------------------------------------------
# colour palette
# ---------------------------------------------------------------------------

COLOR_HIGH_CONF = QColor(0x27, 0xAE, 0x60, 220)     # green  — >= 0.8
COLOR_MED_CONF  = QColor(0xF3, 0x9C, 0x12, 220)     # orange — 0.5–0.8
COLOR_LOW_CONF  = QColor(0xE7, 0x4C, 0x3C, 220)     # red    — < 0.5
COLOR_NEEDS_REVIEW = QColor(0xFF, 0xF3, 0xCD)       # light yellow bg

# ---------------------------------------------------------------------------
# QSS theme — colorful light theme
# ---------------------------------------------------------------------------

QSS = """
/* ── globals ───────────────────────────────────────────── */
QMainWindow, QWidget#central {
    background-color: #F5F6FA;
    color: #2C3E50;
    font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC", sans-serif;
    font-size: 13px;
}

/* ── toolbar ──────────────────────────────────────────── */
QWidget#toolbar {
    background-color: #FFFFFF;
    border-bottom: 1px solid #E0E4E8;
    min-height: 48px;
    padding: 0 12px;
}

QWidget#actionbar {
    background-color: #FFFFFF;
    border-top: 1px solid #E0E4E8;
    min-height: 52px;
    padding: 0 12px;
}

/* ── buttons ──────────────────────────────────────────── */
QPushButton {
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    color: #FFFFFF;
    min-height: 32px;
}
QPushButton:hover {
    opacity: 0.9;
}
QPushButton:pressed {
    margin-top: 1px;
}
QPushButton:disabled {
    background-color: #E0E4E8 !important;
    color: #BDC3C7 !important;
}
QPushButton#open-btn {
    background-color: #00BFA5;
}
QPushButton#open-btn:hover {
    background-color: #00A68C;
}
QPushButton#nav-btn {
    background-color: #95A5A6;
    padding: 6px 14px;
}
QPushButton#nav-btn:hover {
    background-color: #7F8C8D;
}
QPushButton#nav-btn:disabled {
    background-color: #E0E4E8;
}
QPushButton#segment-btn {
    background-color: #4A90D9;
}
QPushButton#segment-btn:hover {
    background-color: #357ABD;
}
QPushButton#ocr-btn {
    background-color: #7C4DFF;
}
QPushButton#ocr-btn:hover {
    background-color: #6535E0;
}
QPushButton#match-btn {
    background-color: #4CAF50;
}
QPushButton#match-btn:hover {
    background-color: #388E3C;
}
QPushButton#save-btn {
    background-color: #FF9800;
}
QPushButton#save-btn:hover {
    background-color: #E68900;
}
QPushButton#export-btn {
    background-color: #607D8B;
}
QPushButton#export-btn:hover {
    background-color: #546E7A;
}

/* ── tabs ─────────────────────────────────────────────── */
QTabWidget::pane {
    background-color: #FFFFFF;
    border: 1px solid #E0E4E8;
    border-top: none;
    border-radius: 0 0 6px 6px;
}
QTabBar::tab {
    background-color: #F0F2F5;
    color: #7F8C8D;
    padding: 10px 20px;
    border: 1px solid #E0E4E8;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #4A90D9;
    border-bottom: 2px solid #4A90D9;
}
QTabBar::tab:hover:!selected {
    background-color: #E8EAEE;
    color: #2C3E50;
}

/* ── tables ───────────────────────────────────────────── */
QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F8F9FB;
    border: 1px solid #E0E4E8;
    border-radius: 6px;
    gridline-color: #EEF0F4;
    selection-background-color: #E3F0FF;
    selection-color: #2C3E50;
}
QTableWidget::item {
    padding: 6px 8px;
}
QHeaderView::section {
    background-color: #F0F2F5;
    color: #7F8C8D;
    font-weight: 600;
    padding: 8px 8px;
    border: none;
    border-bottom: 2px solid #E0E4E8;
    text-transform: uppercase;
    font-size: 11px;
}

/* ── group box ────────────────────────────────────────── */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E0E4E8;
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 12px 12px;
    font-weight: 600;
    color: #2C3E50;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 10px;
    margin-left: 8px;
    background-color: #FFFFFF;
    border: 1px solid #E0E4E8;
    border-radius: 4px;
}

/* ── slider ───────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 6px;
    background: #E0E4E8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #4A90D9;
    border: 2px solid #FFFFFF;
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 10px;
}
QSlider::sub-page:horizontal {
    background: #90CAF9;
    border-radius: 3px;
}

/* ── list widget ──────────────────────────────────────── */
QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #E0E4E8;
    border-radius: 6px;
    outline: none;
}
QListWidget::item {
    padding: 8px 10px;
    border-bottom: 1px solid #F0F2F5;
}
QListWidget::item:selected {
    background-color: #E3F0FF;
    color: #2C3E50;
}

/* ── scroll area ──────────────────────────────────────── */
QScrollArea {
    border: 1px solid #E0E4E8;
    border-radius: 6px;
    background-color: #F5F6FA;
}

/* ── status bar ───────────────────────────────────────── */
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E0E4E8;
    color: #7F8C8D;
    font-size: 12px;
    padding: 4px;
}
QStatusBar::item {
    border: none;
}

/* ── label ────────────────────────────────────────────── */
QLabel {
    color: #2C3E50;
}

/* ── slider label ─────────────────────────────────────── */
QLabel#conf-label {
    color: #4A90D9;
    font-weight: 700;
    font-size: 14px;
}

/* ── splitter ─────────────────────────────────────────── */
QSplitter::handle {
    background-color: #E0E4E8;
    width: 2px;
    margin: 4px 0;
}

/* ── preview frame ────────────────────────────────────── */
QWidget#preview-frame {
    background-color: #F0F2F5;
    border: 1px solid #E0E4E8;
    border-radius: 8px;
}

/* ── backend LED ──────────────────────────────────────── */
QLed {
    min-width: 12px;
    min-height: 12px;
    max-width: 12px;
    max-height: 12px;
    border-radius: 6px;
}
"""

# ---------------------------------------------------------------------------
# worker threads (keep UI responsive during API calls)
# ---------------------------------------------------------------------------

class _ApiWorker(QThread):
    """Generic QThread that calls `fn(*args, **kwargs)` and emits result/error."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except ClientError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# image preview widget
# ---------------------------------------------------------------------------

class ImagePreviewWidget(QWidget):
    """Scrollable/zoomable image preview with optional detection-box overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._boxes: list[dict] = []
        self._scale = 1.0
        self._cached_scaled: QPixmap | None = None
        self._cached_w: int = 0
        self._cached_h: int = 0
        self.setMinimumSize(400, 300)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(0xF0, 0xF2, 0xF5))
        self.setPalette(pal)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def load_image(self, path: Path) -> None:
        """Load a new image and auto-fit to widget width."""
        img = QImage(str(path))
        if img.isNull():
            return
        pix = QPixmap.fromImage(img)
        if pix.isNull():
            return
        self._pixmap = pix
        self._boxes.clear()
        self._cached_scaled = None
        self._fit_scale()
        self.update()

    def set_boxes(self, boxes: list[dict]) -> None:
        """Set detection boxes (each has 'bbox' [x1,y1,x2,y2], 'confidence', optional 'polygon')."""
        self._boxes = boxes
        self.update()

    def clear(self) -> None:
        self._pixmap = None
        self._boxes.clear()
        self.update()

    # ------------------------------------------------------------------
    # paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        super().paintEvent(event)
        try:
            self._safe_paint()
        except Exception:
            pass  # PyQt5: any exception in paintEvent → qFatal

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap is not None:
            old_scale = self._scale
            self._fit_scale()
            if abs(self._scale - old_scale) > 0.01:
                self._cached_scaled = None
                self.update()

    def _ensure_cache(self):
        if self._pixmap is None or self._pixmap.isNull():
            return None, 0, 0
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        w = self.width()
        h = self.height()
        if self._cached_scaled is not None and self._cached_w == w and self._cached_h == h:
            return self._cached_scaled, self._ox, self._oy
        pw2 = int(pw * self._scale)
        ph2 = int(ph * self._scale)
        if pw2 <= 0 or ph2 <= 0:
            return None, 0, 0
        scaled = self._pixmap.scaled(pw2, ph2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cached_scaled = scaled
        self._cached_w = w
        self._cached_h = h
        self._ox = (w - scaled.width()) // 2
        self._oy = (h - scaled.height()) // 2
        return scaled, self._ox, self._oy

    def _safe_paint(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            p = QPainter(self)
            p.setPen(QColor("#BDC3C7"))
            p.setFont(QFont("Sans", 16))
            p.drawText(self.rect(), Qt.AlignCenter, "拖拽或点击📂选择图片")
            p.end()
            return
        if self._scale <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        scaled, ox, oy = self._ensure_cache()
        if scaled is None:
            p.end()
            return
        p.drawPixmap(ox, oy, scaled)

        if not self._boxes:
            p.end()
            return

        # draw masks + boxes
        sx = self._scale
        font = QFont("Sans", 9)
        p.setFont(font)
        for box in self._boxes:
            b = box["bbox"]
            conf = box.get("confidence", 0.0)
            x1 = b[0] * sx + ox
            y1 = b[1] * sx + oy
            x2 = b[2] * sx + ox
            y2 = b[3] * sx + oy
            if conf >= 0.8:
                color = COLOR_HIGH_CONF
            elif conf >= 0.5:
                color = COLOR_MED_CONF
            else:
                color = COLOR_LOW_CONF

            # polygon mask fill (semi-transparent)
            polygon = box.get("polygon")
            if polygon and len(polygon) >= 3:
                path = QPainterPath()
                px0 = polygon[0][0] * sx + ox
                py0 = polygon[0][1] * sx + oy
                path.moveTo(px0, py0)
                for px, py in polygon[1:]:
                    path.lineTo(px * sx + ox, py * sx + oy)
                path.closeSubpath()
                fill_color = QColor(color)
                fill_color.setAlpha(80)
                p.fillPath(path, QBrush(fill_color))

            # bbox rectangle
            pen = QPen(color, 2.0)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            # confidence label
            label = f"{conf:.2f}"
            p.fillRect(QRectF(x1, y1 - 16, 38, 16), color)
            p.setPen(Qt.white)
            p.drawText(QRectF(x1 + 2, y1 - 16, 36, 16), Qt.AlignVCenter | Qt.AlignLeft, label)

        p.end()

    def wheelEvent(self, event):
        if event.modifiers() != Qt.ControlModifier:
            event.ignore()
            return
        delta = event.angleDelta().y() / 120.0
        self._scale = max(0.1, min(5.0, self._scale + delta * 0.15))
        self.update()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _fit_scale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        pw = self._pixmap.width()
        if pw <= 0:
            return
        w = max(1, self.width() - 20)
        self._scale = max(0.01, min(1.0, w / pw))


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Book inventory desktop application."""

    def __init__(self, backend: BackendManager):
        super().__init__()
        self._backend = backend
        self._current_image: Path | None = None
        self._image_list: list[Path] = []
        self._image_index: int = -1
        self._all_detections: list[dict] = []
        self._ocr_books: list[dict] = []
        self._worker: _ApiWorker | None = None
        self._crop_pixmaps_full: list[QPixmap | None] = []  # full-size crop cache (lazy)
        self._has_unsaved: bool = False

        self.setWindowTitle("BookPhoto Split — 图书盘点")
        self.resize(1400, 850)
        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_stylesheet()
        self._connect_signals()
        self._setup_shortcuts()

        # auto-start backend
        self._on_start_backend()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        self._central = QWidget()
        self._central.setObjectName("central")
        self.setCentralWidget(self._central)
        root = QVBoxLayout(self._central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- toolbar ---
        self._toolbar_widget = QWidget()
        toolbar = QHBoxLayout(self._toolbar_widget)
        toolbar.setContentsMargins(16, 8, 16, 8)
        toolbar.setSpacing(8)

        self._btn_open = QPushButton("📂 选择图片")
        toolbar.addWidget(self._btn_open)

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(36)
        self._btn_prev.setToolTip("上一张 (←)")
        self._btn_prev.setEnabled(False)
        toolbar.addWidget(self._btn_prev)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(36)
        self._btn_next.setToolTip("下一张 (→)")
        self._btn_next.setEnabled(False)
        toolbar.addWidget(self._btn_next)

        self._current_image_label = QLabel("未选择图片")
        self._current_image_label.setStyleSheet(
            "color:#2C3E50;font-weight:bold;font-size:13px;padding:2px 8px;"
            "background:#E8F8F5;border-radius:3px;"
        )
        toolbar.addWidget(self._current_image_label)
        toolbar.addSpacing(8)

        toolbar.addSpacing(16)
        toolbar.addWidget(QLabel("置信度:"))
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(10, 100)
        self._conf_slider.setValue(25)
        self._conf_slider.setFixedWidth(160)
        toolbar.addWidget(self._conf_slider)

        self._conf_label = QLabel("0.25")
        self._conf_label.setFixedWidth(36)
        toolbar.addWidget(self._conf_label)
        toolbar.addSpacing(16)

        self._backend_led = QLabel()
        self._backend_led.setFixedSize(12, 12)
        self._backend_led.setStyleSheet("border-radius:6px;background-color:#95A5A6;")
        toolbar.addWidget(self._backend_led)
        self._backend_status = QLabel("后端未连接")
        self._backend_status.setStyleSheet("color:#7F8C8D;font-size:12px;")
        toolbar.addWidget(self._backend_status)
        toolbar.addStretch()

        root.addWidget(self._toolbar_widget)

        # --- body: splitter (image | tabs) ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        # -- left: image --
        self._preview = ImagePreviewWidget()
        splitter.addWidget(self._preview)

        # -- right: tabs --
        self._tabs = QTabWidget()

        # tab 0: detection summary
        self._tab_detect = QWidget()
        self._build_detect_tab()
        self._tabs.addTab(self._tab_detect, "🔍 检测结果")

        # tab 1: crop gallery
        self._tab_crop = QWidget()
        self._build_crop_tab()
        self._tabs.addTab(self._tab_crop, "✂️ 裁剪书脊")

        # tab 2: OCR
        self._tab_ocr = QWidget()
        self._build_ocr_tab()
        self._tabs.addTab(self._tab_ocr, "📖 OCR 结果")

        # tab 3: evaluation details
        self._tab_match = QWidget()
        self._build_match_tab()
        self._tabs.addTab(self._tab_match, "📊 评估详情")

        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        root.addWidget(splitter, 1)

        # --- action bar ---
        self._actionbar_widget = QWidget()
        btn_row = QHBoxLayout(self._actionbar_widget)
        btn_row.setContentsMargins(16, 8, 16, 8)
        btn_row.setSpacing(10)

        self._btn_segment = QPushButton("🔍 分割检测")
        self._btn_segment.setMinimumHeight(36)
        btn_row.addWidget(self._btn_segment)

        self._btn_ocr = QPushButton("📖 OCR 识别")
        self._btn_ocr.setMinimumHeight(36)
        self._btn_ocr.setEnabled(False)
        btn_row.addWidget(self._btn_ocr)

        self._btn_match = QPushButton("📊 评估对比")
        self._btn_match.setMinimumHeight(36)
        self._btn_match.setEnabled(False)
        btn_row.addWidget(self._btn_match)

        self._btn_retry_fail = QPushButton("🔄 重试失败")
        self._btn_retry_fail.setMinimumHeight(36)
        self._btn_retry_fail.setEnabled(False)
        btn_row.addWidget(self._btn_retry_fail)

        self._btn_retry_extra = QPushButton("🔄 重试多检")
        self._btn_retry_extra.setMinimumHeight(36)
        self._btn_retry_extra.setEnabled(False)
        btn_row.addWidget(self._btn_retry_extra)

        self._btn_save = QPushButton("💾 保存结果")
        self._btn_save.setMinimumHeight(36)
        self._btn_save.setEnabled(False)
        btn_row.addWidget(self._btn_save)

        self._btn_export = QPushButton("📤 导出 CSV")
        self._btn_export.setMinimumHeight(36)
        self._btn_export.setEnabled(False)
        btn_row.addWidget(self._btn_export)

        btn_row.addStretch()
        root.addWidget(self._actionbar_widget)

        # --- status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪 — 点击📂选择图片开始")

    # ------------------------------------------------------------------
    # tab builders
    # ------------------------------------------------------------------

    def _build_detect_tab(self) -> None:
        layout = QVBoxLayout(self._tab_detect)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # summary group
        gb = QGroupBox("预测质量")
        gb_lay = QVBoxLayout(gb)
        self._lbl_detect_summary = QLabel("尚未检测")
        self._lbl_detect_summary.setWordWrap(True)
        gb_lay.addWidget(self._lbl_detect_summary)
        layout.addWidget(gb)

        # detail table
        self._detect_table = QTableWidget(0, 3)
        self._detect_table.setHorizontalHeaderLabels(["序号", "置信度", "尺寸 (宽×高)"])
        self._detect_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._detect_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._detect_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self._detect_table)

    def _build_crop_tab(self) -> None:
        """Tab showing cropped spines: list on left, large preview on right."""
        layout = QVBoxLayout(self._tab_crop)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        hint = QLabel("分割后自动生成裁剪书脊。点击列表项查看大图，拖动置信度滑块可实时筛选。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;")
        layout.addWidget(hint)

        # master-detail splitter: list | preview
        splitter = QSplitter(Qt.Horizontal)

        # left: spine list
        self._crop_list = QListWidget()
        self._crop_list.setIconSize(QSize(40, 120))
        self._crop_list.setMinimumWidth(200)
        splitter.addWidget(self._crop_list)

        # right: large preview in scroll area
        self._crop_preview_scroll = QScrollArea()
        self._crop_preview_scroll.setWidgetResizable(True)
        self._crop_preview_label = QLabel()
        self._crop_preview_label.setAlignment(Qt.AlignCenter)
        self._crop_preview_label.setStyleSheet("background-color:#2D2D2D;")
        self._crop_preview_scroll.setWidget(self._crop_preview_label)
        splitter.addWidget(self._crop_preview_scroll)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        layout.addWidget(splitter, 1)

        # bottom: save buttons
        btn_row = QHBoxLayout()
        self._btn_save_crop_current = QPushButton("💾 保存当前书脊")
        self._btn_save_crop_current.setEnabled(False)
        btn_row.addWidget(self._btn_save_crop_current)

        self._btn_save_crop_all = QPushButton("💾 保存全部书脊")
        self._btn_save_crop_all.setEnabled(False)
        btn_row.addWidget(self._btn_save_crop_all)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _build_ocr_tab(self) -> None:
        layout = QVBoxLayout(self._tab_ocr)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._lbl_ocr_summary = QLabel("尚未 OCR 识别")
        self._lbl_ocr_summary.setWordWrap(True)
        layout.addWidget(self._lbl_ocr_summary)

        self._ocr_table = QTableWidget(0, 6)
        self._ocr_table.setHorizontalHeaderLabels(["序号", "置信度", "OCR 书名", "馆藏匹配", "书名标注", "结果"])
        self._ocr_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._ocr_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._ocr_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self._ocr_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ocr_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self._ocr_table)

    def _build_match_tab(self) -> None:
        layout = QVBoxLayout(self._tab_match)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._lbl_match_summary = QLabel("OCR 完成后点击「📊 评估对比」查看详情")
        self._lbl_match_summary.setWordWrap(True)
        layout.addWidget(self._lbl_match_summary)

        self._match_table = QTableWidget(0, 7)
        self._match_table.setHorizontalHeaderLabels([
            "序号", "OCR 书名", "馆藏匹配", "书名标注", "标注馆藏", "分数", "结果",
        ])
        self._match_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._match_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._match_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._match_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self._match_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._match_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self._match_table)

    # ==================================================================
    # signal wiring
    # ==================================================================

    def _connect_signals(self) -> None:
        self._btn_open.clicked.connect(self._on_open_image)
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        self._btn_segment.clicked.connect(self._on_segment)
        self._btn_ocr.clicked.connect(self._on_ocr)
        self._btn_match.clicked.connect(self._on_evaluate)
        self._btn_retry_fail.clicked.connect(self._on_retry_fail)
        self._btn_retry_extra.clicked.connect(self._on_retry_extra)
        self._ocr_table.cellDoubleClicked.connect(self._on_ocr_cell_double_click)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_prev.clicked.connect(self._on_prev_image)
        self._btn_next.clicked.connect(self._on_next_image)
        self._crop_list.currentRowChanged.connect(self._on_crop_selected)
        self._btn_save_crop_current.clicked.connect(self._on_save_crop_current)
        self._btn_save_crop_all.clicked.connect(self._on_save_crop_all)

    # ==================================================================
    # QSS stylesheet
    # ==================================================================

    def _apply_stylesheet(self) -> None:
        self._btn_open.setObjectName("open-btn")
        self._btn_prev.setObjectName("nav-btn")
        self._btn_next.setObjectName("nav-btn")
        self._btn_segment.setObjectName("segment-btn")
        self._btn_ocr.setObjectName("ocr-btn")
        self._btn_match.setObjectName("match-btn")
        self._btn_save.setObjectName("save-btn")
        self._btn_export.setObjectName("export-btn")
        self._conf_label.setObjectName("conf-label")
        self._toolbar_widget.setObjectName("toolbar")
        self._actionbar_widget.setObjectName("actionbar")
        self._preview.setObjectName("preview-frame")
        self.setStyleSheet(QSS)

    # ==================================================================
    # backend lifecycle
    # ==================================================================

    def _on_start_backend(self) -> None:
        self._backend_status.setText("启动中…")
        self._backend_led.setStyleSheet("border-radius:7px;background-color:#F39C12;")
        self._status_bar.showMessage("正在启动后端服务…")

        self._backend.start()
        ready = self._backend.wait_ready(timeout=15.0)
        if ready:
            info = self._backend.health_info()
            model_ok = info.get("model_exists", False)
            cat_ok = info.get("catalog_exists", False)
            ocr_n = info.get("ocr_results", 0)
            self._backend_status.setText(f"后端就绪 | 模型: {'✓' if model_ok else '✗'} | 目录: {'✓' if cat_ok else '✗'} | OCR: {ocr_n}")
            self._backend_led.setStyleSheet("border-radius:7px;background-color:#27AE60;")
            self._status_bar.showMessage("就绪 — 点击「选择图片」开始")
        else:
            self._backend_status.setText("后端启动失败")
            self._backend_led.setStyleSheet("border-radius:7px;background-color:#E74C3C;")
            self._status_bar.showMessage("后端启动失败，请检查环境")

    # ==================================================================
    # slot: open image
    # ==================================================================

    def _on_open_image(self) -> None:
        path = self._open_image_dialog()
        if not path:
            return
        self._navigate_to(Path(path))

    def _open_image_dialog(self) -> str | None:
        """Custom image picker dialog with live preview."""
        dlg = QDialog(self)
        dlg.setWindowTitle("选择书架照片")
        dlg.resize(860, 520)
        dlg.setMinimumSize(700, 400)

        # file list
        file_list = QListWidget()
        file_list.setFixedWidth(320)
        file_list.setStyleSheet("QListWidget{font-size:11pt;}")

        # preview
        preview = QLabel("← 点击左侧文件名预览")
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumWidth(400)
        preview.setStyleSheet(
            "QLabel{background:#E8ECF0; border:1px solid #BDC3C7; color:#7F8C8D; font-size:12pt; padding:10px;}"
        )

        # directory selector
        dir_edit = QLineEdit(str(RAW_DIR.resolve()))
        dir_edit.setPlaceholderText("图片文件夹路径…")
        dir_edit.setReadOnly(True)
        btn_browse = QPushButton("浏览…")
        btn_browse.setFixedWidth(70)

        dir_bar = QHBoxLayout()
        dir_bar.addWidget(QLabel("文件夹:"))
        dir_bar.addWidget(dir_edit, 1)
        dir_bar.addWidget(btn_browse)

        # buttons
        btn_ok = QPushButton("确定")
        btn_ok.setFixedWidth(90)
        btn_cancel = QPushButton("取消")
        btn_cancel.setFixedWidth(90)
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        btn_bar.addWidget(btn_ok)
        btn_bar.addWidget(btn_cancel)

        # file list + preview
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(file_list)
        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(dlg)
        layout.addLayout(dir_bar)
        layout.addWidget(splitter, 1)
        layout.addLayout(btn_bar)

        # --- load files ---
        def _load_dir(dir_path: Path) -> None:
            file_list.clear()
            preview.setText("← 点击左侧文件名预览")
            exts = {".jpg", ".jpeg", ".png"}
            files = sorted(
                [f for f in dir_path.iterdir() if f.suffix.lower() in exts],
                key=lambda f: int(f.stem) if f.stem.isdigit() else f.name,
            )
            for f in files:
                item = QListWidgetItem(f.name)
                item.setData(Qt.UserRole, str(f))
                item.setToolTip(str(f))
                file_list.addItem(item)

        _load_dir(RAW_DIR)

        def _on_selected(row: int) -> None:
            if row < 0:
                return
            item = file_list.item(row)
            path_str = item.data(Qt.UserRole)
            pix = QPixmap(path_str)
            if pix.isNull():
                return
            avail_w = preview.width() - 20
            avail_h = preview.height() - 20
            if avail_w <= 0:
                avail_w = 380
            if avail_h <= 0:
                avail_h = 400
            pix = pix.scaled(avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            preview.setPixmap(pix)

        file_list.currentRowChanged.connect(_on_selected)

        def _on_browse() -> None:
            path = QFileDialog.getExistingDirectory(dlg, "选择图片文件夹", str(RAW_DIR))
            if path:
                dir_edit.setText(path)
                _load_dir(Path(path))

        btn_browse.clicked.connect(_on_browse)
        btn_ok.clicked.connect(lambda: dlg.accept())
        btn_cancel.clicked.connect(lambda: dlg.reject())
        file_list.itemDoubleClicked.connect(lambda: dlg.accept())

        if dlg.exec_() == QDialog.Accepted:
            row = file_list.currentRow()
            if row >= 0:
                return file_list.item(row).data(Qt.UserRole)
        return None

    # ==================================================================
    # slot: confidence slider
    # ==================================================================

    def _on_conf_changed(self, value: int) -> None:
        val = value / 100.0
        self._conf_label.setText(f"{val:.2f}")
        if not self._all_detections:
            return
        filtered = [d for d in self._all_detections if d["confidence"] >= val]
        self._preview.set_boxes(filtered)
        self._update_detect_table(filtered)
        self._update_crop_visibility(filtered)
        self._status_bar.showMessage(f"置信度阈值: {val:.2f} | 显示 {len(filtered)}/{len(self._all_detections)} 个书脊")

    # ==================================================================
    # slot: segment
    # ==================================================================

    def _on_segment(self) -> None:
        if self._current_image is None:
            return
        conf = self._conf_slider.value() / 100.0
        self._set_busy(True, "正在分割检测书脊…")
        self._run_worker(api_segment, self._current_image, conf=conf)

    def _on_segment_done(self, result: dict) -> None:
        self._set_busy(False)
        boxes = result.get("boxes", [])
        count = result.get("count", len(boxes))
        self._all_detections = boxes
        self._has_unsaved = True
        self._ocr_books.clear()

        # apply current conf filter
        conf_thresh = self._conf_slider.value() / 100.0
        filtered = [d for d in boxes if d["confidence"] >= conf_thresh]

        self._preview.set_boxes(filtered)

        # summary
        if count == 0:
            self._lbl_detect_summary.setText("⚠️ 未检测到书脊，请调整置信度阈值或更换图片")
            self._detect_table.setRowCount(0)
        else:
            confs = [b["confidence"] for b in boxes]
            avg = sum(confs) / len(confs)
            high = sum(1 for c in confs if c >= 0.8)
            med = sum(1 for c in confs if 0.5 <= c < 0.8)
            low = sum(1 for c in confs if c < 0.5)
            self._lbl_detect_summary.setText(
                f"检测到 <b>{count}</b> 个书脊 | "
                f"平均置信度 <b>{avg:.3f}</b> | "
                f"高置信({high}) 中置信({med}) 低置信({low})"
            )
            self._update_detect_table(filtered)

        # generate crop thumbnails
        self._build_crop_gallery(boxes)

        self._ocr_table.setRowCount(0)
        self._lbl_ocr_summary.setText("尚未 OCR 识别")
        self._btn_ocr.setEnabled(count > 0)
        self._btn_save.setEnabled(count > 0)
        self._tabs.setCurrentIndex(0)
        self._status_bar.showMessage(f"分割完成: {count} 个书脊 (显示 {len(filtered)})")

    # ==================================================================
    # slot: OCR
    # ==================================================================

    def _update_detect_table(self, detections: list[dict]) -> None:
        """Update detection table with filtered detections."""
        self._detect_table.setRowCount(len(detections))
        for i, b in enumerate(detections):
            conf = b["confidence"]
            bbox = b["bbox"]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            no_item = QTableWidgetItem(str(i + 1))
            no_item.setTextAlignment(Qt.AlignCenter)
            self._detect_table.setItem(i, 0, no_item)

            conf_item = QTableWidgetItem(f"{conf:.4f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            if conf >= 0.8:
                conf_item.setForeground(COLOR_HIGH_CONF)
            elif conf >= 0.5:
                conf_item.setForeground(COLOR_MED_CONF)
            else:
                conf_item.setForeground(COLOR_LOW_CONF)
            self._detect_table.setItem(i, 1, conf_item)

            self._detect_table.setItem(i, 2, QTableWidgetItem(f"{w:.0f} × {h:.0f}"))

    def _build_crop_gallery(self, detections: list[dict]) -> None:
        """Generate crop thumbnails for the list (full crops are lazy)."""
        self._crop_list.clear()
        self._crop_pixmaps_full.clear()

        if self._preview._pixmap is None or self._preview._pixmap.isNull():
            return

        src_pix = self._preview._pixmap

        for idx, det in enumerate(detections):
            # small thumbnail for list icon only (full crop is lazy)
            thumb_pix = self._crop_spine_pixmap(src_pix, det, 40)
            conf = det["confidence"]

            item = QListWidgetItem()
            if thumb_pix is not None:
                item.setIcon(QIcon(thumb_pix))
            item.setText(f"#{idx + 1}  {conf:.2f}")
            item.setToolTip(f"书脊 {idx + 1} | 置信度 {conf:.4f}")
            self._crop_list.addItem(item)

        if self._crop_list.count() > 0:
            self._crop_list.setCurrentRow(0)

        self._btn_save_crop_all.setEnabled(self._crop_list.count() > 0)

    def _update_crop_visibility(self, filtered: list[dict]) -> None:
        """Show/hide crop list rows based on filtered detections."""
        filtered_set = {id(d) for d in filtered}
        for i, det in enumerate(self._all_detections):
            if i < self._crop_list.count():
                self._crop_list.setRowHidden(i, id(det) not in filtered_set)

        # if current row is hidden, find first visible
        current = self._crop_list.currentRow()
        if current >= 0 and self._crop_list.isRowHidden(current):
            for i in range(self._crop_list.count()):
                if not self._crop_list.isRowHidden(i):
                    self._crop_list.setCurrentRow(i)
                    return

    def _on_crop_selected(self, row: int) -> None:
        """Show selected crop in large preview (lazy-generated)."""
        if row < 0 or row >= len(self._all_detections):
            self._crop_preview_label.setPixmap(QPixmap())
            self._crop_preview_label.setText("分割后在此显示书脊裁剪图")
            self._btn_save_crop_current.setEnabled(False)
            return

        pix = self._get_full_crop(row)
        if pix is None or pix.isNull():
            self._crop_preview_label.setText("无法生成裁剪图")
            self._btn_save_crop_current.setEnabled(False)
            return

        avail_w = max(100, self._crop_preview_scroll.viewport().width() - 20)
        avail_h = max(100, self._crop_preview_scroll.viewport().height() - 20)
        scaled = pix.scaled(avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._crop_preview_label.setPixmap(scaled)
        self._btn_save_crop_current.setEnabled(True)

    def _save_crop_to_file(self, row: int) -> None:
        """Generate full-size crop on demand and save."""
        pix = self._get_full_crop(row)
        if pix is None or pix.isNull():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存书脊裁剪图",
            f"spine_{row + 1:03d}.png",
            "PNG (*.png);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog,
        )
        if not path:
            return
        pix.save(path, "PNG")
        self._status_bar.showMessage(f"已保存: {path}")

    def _on_save_crop_current(self) -> None:
        row = self._crop_list.currentRow()
        if row < 0:
            return
        self._save_crop_to_file(row)

    def _on_save_crop_all(self) -> None:
        conf_thresh = self._conf_slider.value() / 100.0
        visible_indices = [
            i for i, det in enumerate(self._all_detections)
            if det["confidence"] >= conf_thresh
        ]
        if not visible_indices:
            QMessageBox.information(self, "提示", "没有可保存的裁剪图")
            return

        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not path:
            return

        out_dir = Path(path)
        n = 0
        for i in visible_indices:
            pix = self._get_full_crop(i)
            if pix is not None and not pix.isNull():
                pix.save(str(out_dir / f"spine_{i + 1:03d}.png"), "PNG")
                n += 1

        self._status_bar.showMessage(f"已保存 {n} 张裁剪图到 {out_dir}")
        QMessageBox.information(self, "保存完成", f"已保存 {n} 张裁剪图到:\n{out_dir}")

    def _get_full_crop(self, idx: int) -> QPixmap | None:
        """Lazy generate and cache full-size crop."""
        if idx < 0 or idx >= len(self._all_detections):
            return None
        if idx < len(self._crop_pixmaps_full) and self._crop_pixmaps_full[idx] is not None:
            return self._crop_pixmaps_full[idx]
        src = self._preview._pixmap
        if src is None or src.isNull():
            return None
        det = self._all_detections[idx]
        pix = self._crop_spine_pixmap(src, det, 1000)
        while len(self._crop_pixmaps_full) <= idx:
            self._crop_pixmaps_full.append(None)
        self._crop_pixmaps_full[idx] = pix
        return pix

    def _crop_spine_pixmap(self, src: QPixmap, det: dict, thumb_w: int) -> QPixmap | None:
        """Crop a single spine from source pixmap using polygon clip. Returns thumbnail."""
        polygon = det.get("polygon")
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            return None

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cw, ch = x2 - x1, y2 - y1
        if cw <= 0 or ch <= 0:
            return None

        # create transparent crop with polygon clip
        crop = QPixmap(cw, ch)
        crop.fill(Qt.transparent)

        p = QPainter(crop)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # clip to polygon if available
        if polygon and len(polygon) >= 3:
            path = QPainterPath()
            path.moveTo(polygon[0][0] - x1, polygon[0][1] - y1)
            for px, py in polygon[1:]:
                path.lineTo(px - x1, py - y1)
            path.closeSubpath()
            p.setClipPath(path)

        # draw source region
        p.drawPixmap(0, 0, src, x1, y1, cw, ch)
        p.end()

        # scale to thumbnail (keep aspect ratio)
        if crop.isNull():
            return None
        ratio = thumb_w / cw
        thumb_h = max(1, int(ch * ratio))
        return crop.scaled(thumb_w, thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # ==================================================================
    # slot: OCR (并行推理, 5 并发)
    # ==================================================================

    _OCR_MAX_CONCURRENT = 10

    def _on_ocr(self, spine_indices: list[int] | None = None) -> None:
        if spine_indices is True or spine_indices is False:
            spine_indices = None  # QPushButton clicked 信号传 bool
        if not self._all_detections:
            QMessageBox.information(self, "提示", "请先执行分割检测")
            return

        conf_thresh = self._conf_slider.value() / 100.0
        all_dets = self._all_detections

        if spine_indices is not None:
            self._ocr_queue = list(spine_indices)
            if not hasattr(self, '_ocr_spine_results') or self._ocr_spine_results is None:
                self._ocr_spine_results = [None] * len(all_dets)
        else:
            self._ocr_queue = [i for i, d in enumerate(all_dets) if d["confidence"] >= conf_thresh]
            if not self._ocr_queue:
                QMessageBox.information(self, "提示", f"所有书脊置信度均低于 {conf_thresh:.2f}，无需 OCR")
                return
            self._ocr_spine_results = [None] * len(all_dets)
            for i, d in enumerate(all_dets):
                if d["confidence"] < conf_thresh:
                    self._ocr_spine_results[i] = {
                        "spine_idx": i, "book_name": f"已跳过(<{conf_thresh:.2f})",
                        "matched_name": None, "score": 0.0, "strategy": "skip_conf",
                        "needs_review": False,
                    }

        msg = f"重试 {len(self._ocr_queue)} 个书脊…" if spine_indices else f"OCR 识别书脊 (高于 {conf_thresh:.2f}, {self._OCR_MAX_CONCURRENT}并发)…"
        self._set_busy(True, msg)

        self._ocr_pending = len(self._ocr_queue)
        self._ocr_completed = 0
        self._ocr_next_qi = 0

        # 启动初始 N 个并发（错峰 200ms 避免同时触发限速）
        self._ocr_workers: list[_ApiWorker] = []  # 保持引用防 GC 崩溃
        import time as _time
        for i in range(min(self._OCR_MAX_CONCURRENT, self._ocr_pending)):
            if i > 0:
                _time.sleep(0.2)
            self._launch_ocr_worker()

    def _launch_ocr_worker(self) -> None:
        if self._ocr_next_qi >= self._ocr_pending:
            return
        qi = self._ocr_next_qi
        self._ocr_next_qi += 1
        idx = self._ocr_queue[qi]

        # 懒生成临时文件（不阻塞启动），400px JPEG 大幅提速
        src = self._preview._pixmap
        pix = self._crop_spine_pixmap(src, self._all_detections[idx], 400) if src else None
        if pix is None or pix.isNull():
            self._ocr_spine_results[idx] = {
                "spine_idx": idx, "book_name": "未识别",
                "matched_name": None, "score": 0.0, "strategy": "skip",
                "needs_review": False,
            }
            self._on_spine_complete()
            return
        tmp = Path(tempfile.mktemp(suffix=".jpg"))
        pix.save(str(tmp), "JPEG", quality=85)

        def _done(result):
            result["spine_idx"] = idx
            self._ocr_spine_results[idx] = result
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            self._on_spine_complete()

        def _err(err_str):
            self._ocr_spine_results[idx] = {
                "spine_idx": idx, "book_name": f"错误: {err_str[:30]}",
                "matched_name": None, "score": 0.0, "strategy": "error",
                "needs_review": False,
            }
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            self._on_spine_complete()

        w = _ApiWorker(api_ocr_spine, tmp)
        w.finished.connect(_done)
        w.error.connect(_err)
        self._ocr_workers.append(w)
        w.start()

    def _on_spine_complete(self) -> None:
        self._ocr_completed += 1
        self._status_bar.showMessage(f"OCR 中 {self._ocr_completed}/{self._ocr_pending}…")
        QApplication.processEvents()

        if self._ocr_completed >= self._ocr_pending:
            self._on_ocr_all_done()
        else:
            # 启动下一个 worker（保持并发数）
            self._launch_ocr_worker()

    def _on_ocr_all_done(self) -> None:
        # 清理 workers（防 QThread GC 崩溃）
        for w in getattr(self, '_ocr_workers', []):
            try: w.wait(100)
            except: pass
        self._ocr_workers.clear()

        self._set_busy(False)
        results = self._ocr_spine_results

        ocr_count = sum(1 for r in results if r and r.get("strategy") != "skip_conf")
        skip_count = sum(1 for r in results if r and r.get("strategy") == "skip_conf")

        self._ocr_table.setRowCount(len(results))
        for i, r in enumerate(results):
            conf = self._all_detections[i].get("confidence", 0.0) if i < len(self._all_detections) else 0
            is_skipped = r.get("strategy") == "skip_conf"

            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(f"{conf:.4f}"),
                QTableWidgetItem(r.get("book_name", "")),
                QTableWidgetItem(r.get("matched_name") or "—"),
                QTableWidgetItem(""),
                QTableWidgetItem(""),
            ]
            items[0].setTextAlignment(Qt.AlignCenter)
            items[1].setTextAlignment(Qt.AlignCenter)
            if is_skipped:
                items[0].setForeground(QColor("#888888"))
                items[1].setForeground(QColor("#888888"))
            for col, it in enumerate(items):
                self._ocr_table.setItem(i, col, it)

        self._btn_match.setEnabled(ocr_count > 0)
        self._has_unsaved = True
        self._tabs.setCurrentIndex(2)
        self._status_bar.showMessage(f"OCR 完成: {ocr_count} 个识别 / {skip_count} 个跳过")

        # 有失败/超时条目的启用重试按钮
        has_fail = any(r and r.get("strategy") in ("error", "skip") and r.get("book_name","").startswith(("错误","未识别"))
                       for r in results)
        self._btn_retry_fail.setEnabled(has_fail)

    def _on_evaluate(self) -> None:
        try:
            self._do_evaluate()
        except Exception as e:
            self._status_bar.showMessage(f"评估失败: {e}")
            self._set_busy(False)

    def _do_evaluate(self) -> None:
        if not hasattr(self, "_ocr_spine_results") or not self._ocr_spine_results:
            QMessageBox.information(self, "提示", "请先执行 OCR 识别")
            return

        photo_id = self._current_image.stem if self._current_image else ""
        gt_path = BOOK_LABELS_DIR / f"{photo_id}.json"
        if not gt_path.exists():
            QMessageBox.information(self, "提示", f"书名标注文件不存在: {gt_path.name}")
            return

        gt_raw = json.loads(gt_path.read_text(encoding="utf-8"))
        # book_labels 用 "name"/"matched_name" 字段
        gt_data = {"books": []}
        for b in gt_raw.get("books", []):
            gt_data["books"].append({
                "original_ocr_name": b.get("name", ""),
                "matched_name": b.get("matched_name"),
                "count": b.get("count", 1),
            })

        eval_result = evaluate_vs_ground_truth(self._ocr_spine_results, gt_data)
        self._eval_per_spine = eval_result["per_spine"]  # 存下来供 retry_extra 用

        summary = eval_result["summary"]
        acc = summary["accuracy"] * 100
        count_acc = summary.get("counting_accuracy", 0) * 100
        prec = summary.get("precision", 0) * 100
        rec = summary.get("recall", 0) * 100
        skipped_info = f" | 跳过(阈值以下) <b>{summary.get('skipped',0)}</b>" if summary.get("skipped", 0) > 0 else ""
        self._lbl_ocr_summary.setText(
            f"识别准确率 <b>{acc:.1f}%</b> ({summary['correct']}/{summary['total_gt']}) | "
            f"计数准确率 <b>{count_acc:.1f}%</b> | "
            f"精确率 <b>{prec:.1f}%</b> / 召回率 <b>{rec:.1f}%</b> | "
            f"漏检 {summary['missed']} / 多检 {summary['extra']}"
            f"{skipped_info}"
        )

        for i, ps in enumerate(eval_result["per_spine"]):
            result = ps.get("result", "")
            gt_name = ps.get("gt_name")
            score = ps.get("gt_score", 0.0)

            if result == "correct":
                result_text = "✓"
                gt_display = gt_name or "—"
                color = COLOR_HIGH_CONF
            elif result == "skipped":
                result_text = "—"
                gt_display = "已跳过"
                color = QColor("#888888")
            else:
                result_text = "✗ 多检"
                gt_display = f"最高: {gt_name} ({score:.2f})" if gt_name else "—"
                color = COLOR_LOW_CONF

            gt_item = QTableWidgetItem(gt_display)
            gt_item.setForeground(color)

            result_item = QTableWidgetItem(result_text)
            result_item.setTextAlignment(Qt.AlignCenter)
            result_item.setForeground(color)
            self._ocr_table.setItem(i, 4, gt_item)
            self._ocr_table.setItem(i, 5, result_item)

        # 填充评估详情 tab
        self._match_table.setRowCount(len(eval_result["per_spine"]))
        for i, ps in enumerate(eval_result["per_spine"]):
            result = ps.get("result", "")
            gt_name = ps.get("gt_name", "")
            gt_matched = ps.get("gt_matched", "")
            score = ps.get("gt_score", 0.0)
            ocr_name = ps.get("ocr_name", "")
            matched = ps.get("matched_name", "")
            match_round = ps.get("match_round", "")

            if result == "correct":
                result_text = "✓"
                color = COLOR_HIGH_CONF
            elif result == "skipped":
                result_text = "—"
                color = QColor("#888888")
            else:
                result_text = "✗"
                color = COLOR_LOW_CONF

            for col, (text, c) in enumerate([
                (str(i + 1), None),
                (ocr_name, None),
                (matched or "—", None),
                (gt_name or "—", color),
                (gt_matched or "—", None),
                (f"{score:.4f}", None),
                (result_text, color),
            ]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter if col in (0, 5, 6) else Qt.AlignLeft)
                if c: item.setForeground(c)
                self._match_table.setItem(i, col, item)

        self._lbl_match_summary.setText(
            f"准确率 <b>{acc:.1f}%</b> ({summary['correct']}/{summary['total_gt']}) | "
            f"漏检 <b>{summary['missed']}</b> | 多检 <b>{summary['extra']}</b>"
            f"{skipped_info}"
        )
        self._tabs.setCurrentIndex(3)  # 切到评估详情 tab

        # 启用重试按钮
        has_fail = any(r and r.get("strategy") in ("error", "skip") and r.get("book_name","").startswith(("错误","未识别"))
                       for r in (self._ocr_spine_results or []))
        self._btn_retry_fail.setEnabled(has_fail)
        # 多检: result="extra" 的
        if hasattr(self, '_eval_per_spine'):
            has_extra = any(ps.get("result") == "extra" for ps in self._eval_per_spine)
            self._btn_retry_extra.setEnabled(has_extra)

    def _on_retry_fail(self) -> None:
        """重试失败/超时的脊柱。"""
        try:
            retry = []
            for i, r in enumerate(self._ocr_spine_results or []):
                if r and r.get("strategy") in ("error", "skip") and r.get("book_name","").startswith(("错误","未识别")):
                    retry.append(i)
            if retry:
                self._on_ocr(spine_indices=retry)
        except Exception as e:
            self._status_bar.showMessage(f"重试失败: {e}")
            self._set_busy(False)

    def _on_retry_extra(self) -> None:
        """重试评估为多检(extra)的脊柱。"""
        try:
            if not hasattr(self, '_eval_per_spine'):
                return
            retry = [ps["spine_idx"] for ps in self._eval_per_spine if ps.get("result") == "extra"]
            if retry:
                self._on_ocr(spine_indices=retry)
        except Exception as e:
            self._status_bar.showMessage(f"重试失败: {e}")
            self._set_busy(False)

    def _on_ocr_cell_double_click(self, row: int, col: int) -> None:
        """双击 OCR 表行 → 单独重试该脊柱。"""
        try:
            if row < 0 or row >= len(self._all_detections):
                return
            reply = QMessageBox.question(self, "重试", f"单独重试第 {row+1} 号书脊？",
                                          QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._on_ocr(spine_indices=[row])
        except Exception as e:
            self._status_bar.showMessage(f"重试失败: {e}")
            self._set_busy(False)

    # ==================================================================
    # slot: match
    # ==================================================================

    def _on_match(self) -> None:
        # collect current OCR table data (post-edit)
        books: list[dict] = []
        for row in range(self._ocr_table.rowCount()):
            name_item = self._ocr_table.item(row, 0)
            cnt_item = self._ocr_table.item(row, 1)
            if name_item is None:
                continue
            name = name_item.text().strip()
            if not name:
                continue
            try:
                cnt = int(cnt_item.text()) if cnt_item else 1
            except ValueError:
                cnt = 1
            books.append({"book_name": name, "count": cnt})

        if not books:
            QMessageBox.information(self, "提示", "OCR 结果为空，请先执行 OCR 识别")
            return

        payload = [{"books": books}]
        self._set_busy(True, "正在馆藏匹配…")
        self._run_worker(
            api_inventory, payload, threshold=0.7,
        )

    def _on_match_done(self, result: dict) -> None:
        self._set_busy(False)
        match_log = result.get("match_log", [])
        book_counts = result.get("book_counts", {})
        total_copies = sum(book_counts.values())
        matched = sum(1 for m in match_log if m.get("matched_name"))
        total = len(match_log)
        match_rate = (matched / total * 100) if total > 0 else 0

        self._lbl_match_summary.setText(
            f"匹配率 <b>{match_rate:.1f}%</b> ({matched}/{total}) | "
            f"总册数 <b>{total_copies}</b> | "
            f"独立书名 <b>{len(book_counts)}</b>"
        )

        self._match_table.setRowCount(total)
        for i, m in enumerate(match_log):
            needs_review = m.get("needs_review", False)

            ocr_item = QTableWidgetItem(m.get("ocr_name", ""))
            matched_item = QTableWidgetItem(m.get("matched_name") or "—")
            score_item = QTableWidgetItem(f"{m.get('score', 0):.4f}")
            strategy_item = QTableWidgetItem(m.get("strategy", ""))
            review_item = QTableWidgetItem("⚠ 需确认" if needs_review else "✓")

            for col, it in enumerate([ocr_item, matched_item, score_item, strategy_item, review_item]):
                if needs_review:
                    it.setBackground(COLOR_NEEDS_REVIEW)
                self._match_table.setItem(i, col, it)

        self._btn_export.setEnabled(True)
        self._has_unsaved = True
        self._tabs.setCurrentIndex(3)
        self._status_bar.showMessage(f"匹配完成: {match_rate:.1f}%")

    # ==================================================================
    # slot: save all products
    # ==================================================================

    def _on_save(self) -> None:
        """Save annotated image + crops + OCR + match results to output/gui/<timestamp>/."""
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = GUI_OUTPUT_DIR / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []

        # 1. segmentation visualization (boxes + masks)
        vis = self._render_annotated()
        if vis is not None:
            vis_path = out_dir / "segmentation_vis.png"
            vis.save(str(vis_path), "PNG")
            saved.append(f"分割图: {vis_path.name}")

        # 2. cropped spines (transparent PNG, filtered by current conf)
        conf_thresh = self._conf_slider.value() / 100.0
        filtered_indices = [
            i for i, det in enumerate(self._all_detections)
            if det["confidence"] >= conf_thresh
        ]
        if filtered_indices:
            crops_dir = out_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            n_saved = 0
            for i in filtered_indices:
                pix = self._get_full_crop(i)
                if pix is not None and not pix.isNull():
                    pix.save(str(crops_dir / f"spine_{i + 1:03d}.png"), "PNG")
                    n_saved += 1
            if n_saved > 0:
                saved.append(f"裁剪书脊: {n_saved} 张 PNG")

        # 3. detection results JSON
        filtered = [self._all_detections[i] for i in filtered_indices]
        if filtered:
            det_data = {
                "image": str(self._current_image) if self._current_image else None,
                "count": len(filtered),
                "conf_threshold": conf_thresh,
                "detections": [
                    {
                        "bbox": d["bbox"],
                        "confidence": d["confidence"],
                        "label": d.get("label", "book"),
                        "polygon": d.get("polygon"),
                    }
                    for d in filtered
                ],
            }
            det_path = out_dir / "detection.json"
            det_path.write_text(
                json.dumps(det_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved.append(f"检测数据: {det_path.name}")

        # 4. OCR results (current table data)
        ocr_data = self._collect_ocr_data()
        if ocr_data:
            ocr_path = out_dir / "ocr_results.json"
            ocr_path.write_text(
                json.dumps(ocr_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved.append(f"OCR: {ocr_path.name}")

        # 5. match results (if available)
        match_data = self._collect_match_data()
        if match_data:
            match_path = out_dir / "match_results.json"
            match_path.write_text(
                json.dumps(match_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved.append(f"匹配: {match_path.name}")

        if not saved:
            QMessageBox.information(self, "提示", "没有可保存的结果，请先执行分割检测")
            return

        self._status_bar.showMessage(f"已保存到 {out_dir} ({', '.join(saved)})")
        self._has_unsaved = False
        QMessageBox.information(
            self, "保存完成",
            f"已保存 {len(saved)} 个文件到:\n{out_dir}\n\n{saved}",
        )

    def _render_annotated(self):
        """Render the original image with detection boxes + masks to a QPixmap."""
        if self._preview._pixmap is None:  # noqa
            return None
        src = self._preview._pixmap  # noqa
        result = QPixmap(src.size())
        result.fill(Qt.white)

        p = QPainter(result)
        p.setRenderHint(QPainter.Antialiasing)
        p.drawPixmap(0, 0, src)

        conf_thresh = self._conf_slider.value() / 100.0
        boxes = [d for d in self._all_detections if d["confidence"] >= conf_thresh]

        if boxes:
            font = QFont("Sans", max(10, int(src.width() / 300)))
            p.setFont(font)
            for box in boxes:
                b = box["bbox"]
                conf = box.get("confidence", 0.0)
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

                if conf >= 0.8:
                    color = COLOR_HIGH_CONF
                elif conf >= 0.5:
                    color = COLOR_MED_CONF
                else:
                    color = COLOR_LOW_CONF

                # polygon mask fill
                polygon = box.get("polygon")
                if polygon and len(polygon) >= 3:
                    path = QPainterPath()
                    path.moveTo(polygon[0][0], polygon[0][1])
                    for px, py in polygon[1:]:
                        path.lineTo(px, py)
                    path.closeSubpath()
                    fill_color = QColor(color)
                    fill_color.setAlpha(80)
                    p.fillPath(path, QBrush(fill_color))

                # bbox rectangle
                pen = QPen(color, max(3.0, src.width() / 500))
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

                # confidence label (scale with image size)
                lbl_h = max(18, int(src.width() / 150))
                p.fillRect(QRectF(x1, y1 - lbl_h, lbl_h * 3, lbl_h), color)
                p.setPen(Qt.white)
                p.drawText(QRectF(x1 + 3, y1 - lbl_h, lbl_h * 3 - 3, lbl_h),
                           Qt.AlignVCenter | Qt.AlignLeft, f"{conf:.2f}")

        p.end()
        return result

    def _collect_ocr_data(self) -> list[dict]:
        """Collect current OCR table data."""
        books: list[dict] = []
        for row in range(self._ocr_table.rowCount()):
            ocr_item = self._ocr_table.item(row, 2)
            matched_item = self._ocr_table.item(row, 3)
            if ocr_item is None:
                continue
            books.append({
                "book_name": ocr_item.text().strip(),
                "matched_name": matched_item.text().strip() if matched_item else "",
            })
        return books

    def _collect_match_data(self) -> dict | None:
        """Collect match table data; returns None if no match data exists."""
        if self._match_table.rowCount() == 0:
            return None
        log: list[dict] = []
        for row in range(self._match_table.rowCount()):
            cells = [self._match_table.item(row, c) for c in range(5)]
            log.append({
                "ocr_name": cells[0].text() if cells[0] else "",
                "matched_name": cells[1].text() if cells[1] else "",
                "score": cells[2].text() if cells[2] else "",
                "strategy": cells[3].text() if cells[3] else "",
                "needs_review": (cells[4].text() if cells[4] else "") == "⚠ 需确认",
            })
        summary_text = self._lbl_match_summary.text()
        return {
            "match_log": log,
            "summary": summary_text,
            "image": str(self._current_image) if self._current_image else None,
        }

    # ==================================================================
    # slot: export
    # ==================================================================

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "导出匹配结果", "inventory_result.csv",
            "CSV (*.csv);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog,
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["OCR书名", "匹配馆藏", "分数", "策略", "需确认"])
                for row in range(self._match_table.rowCount()):
                    writer.writerow([
                        self._match_table.item(row, col).text()
                        if self._match_table.item(row, col) else ""
                        for col in range(5)
                    ])
            self._status_bar.showMessage(f"已导出: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # ==================================================================
    # image navigation
    # ==================================================================

    def _build_image_list(self) -> None:
        if self._current_image is None:
            return
        parent = self._current_image.parent
        exts = {".jpg", ".jpeg", ".png"}
        images = sorted(
            [p for p in parent.iterdir() if p.suffix.lower() in exts],
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
        )
        self._image_list = images
        try:
            self._image_index = images.index(self._current_image)
        except ValueError:
            self._image_index = -1

    def _update_nav_buttons(self) -> None:
        self._btn_prev.setEnabled(self._image_index > 0)
        self._btn_next.setEnabled(self._image_index < len(self._image_list) - 1)

    def _navigate_to(self, path: Path) -> None:
        self._current_image = path
        self._preview.load_image(path)
        self._all_detections.clear()
        self._ocr_books.clear()
        self._crop_list.clear()
        self._crop_pixmaps_full.clear()
        self._crop_preview_label.setPixmap(QPixmap())
        self._crop_preview_label.setText("分割后在此显示书脊裁剪图")
        self._btn_save_crop_current.setEnabled(False)
        self._btn_save_crop_all.setEnabled(False)
        self._detect_table.setRowCount(0)
        self._lbl_detect_summary.setText("尚未检测")
        self._ocr_table.setRowCount(0)
        self._lbl_ocr_summary.setText("尚未 OCR 识别")
        self._match_table.setRowCount(0)
        self._lbl_match_summary.setText("OCR 完成后点击「📊 评估对比」查看详情")
        self._btn_segment.setEnabled(True)
        self._btn_ocr.setEnabled(False)
        self._btn_match.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._has_unsaved = False
        self._current_image_label.setText(f"📷 {path.name}")
        self._build_image_list()
        self._update_nav_buttons()
        self._status_bar.showMessage(f"已选择: {path.name}")

    def _on_prev_image(self) -> None:
        if self._image_index > 0:
            self._navigate_to(self._image_list[self._image_index - 1])

    def _on_next_image(self) -> None:
        if self._image_index < len(self._image_list) - 1:
            self._navigate_to(self._image_list[self._image_index + 1])

    # ==================================================================
    # drag & drop
    # ==================================================================

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self._navigate_to(path)
                break

    # ==================================================================
    # keyboard shortcuts
    # ==================================================================

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_open_image)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save)
        QShortcut(QKeySequence("Ctrl+E"), self, self._on_export)
        QShortcut(QKeySequence("Left"), self, self._on_prev_image)
        QShortcut(QKeySequence("Right"), self, self._on_next_image)
        QShortcut(QKeySequence("Escape"), self, self._on_close_dialog)

    def _on_close_dialog(self) -> None:
        self.close()

    # ==================================================================
    # helpers
    # ==================================================================

    def _set_busy(self, busy: bool, msg: str = "") -> None:
        for btn in [self._btn_open, self._btn_segment, self._btn_ocr, self._btn_match, self._btn_save,
                     self._btn_retry_fail, self._btn_retry_extra]:
            btn.setEnabled(not busy)
        if msg:
            self._status_bar.showMessage(msg)
        QApplication.processEvents()

    def _run_worker(self, fn, *args, **kwargs):
        """Run an API call in a background thread; wire up result/error."""
        self._worker = _ApiWorker(fn, *args, **kwargs)
        self._worker.finished.connect(self._on_worker_result)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_result(self, result):
        try:
            w = self._worker
            if w is None: return
            fn = w._fn  # noqa
            if fn is api_segment: self._on_segment_done(result)
            elif fn is api_inventory: self._on_match_done(result)
        except Exception: pass

    def _on_worker_error(self, err: str):
        try:
            self._set_busy(False)
            self._status_bar.showMessage(f"错误: {err}")
        except Exception: pass

    # ==================================================================
    # close
    # ==================================================================

    def closeEvent(self, event):
        if self._has_unsaved and self._all_detections:
            reply = QMessageBox.question(
                self, "确认关闭", "有未保存的结果，确定关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        self._backend.stop()
        event.accept()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    backend = BackendManager()
    window = MainWindow(backend)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
