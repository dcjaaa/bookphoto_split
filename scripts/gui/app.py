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
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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

from scripts.gui.backend import BackendManager
from scripts.gui.client import (
    segment as api_segment,
    ocr as api_ocr,
    inventory as api_inventory,
    health as api_health,
    ClientError,
)
from scripts.utils.paths import GUI_OUTPUT_DIR

# ---------------------------------------------------------------------------
# colour palette
# ---------------------------------------------------------------------------

COLOR_HIGH_CONF = QColor(0x27, 0xAE, 0x60, 220)     # green  — >= 0.8
COLOR_MED_CONF  = QColor(0xF3, 0x9C, 0x12, 220)     # orange — 0.5–0.8
COLOR_LOW_CONF  = QColor(0xE7, 0x4C, 0x3C, 220)     # red    — < 0.5
COLOR_NEEDS_REVIEW = QColor(0xFF, 0xF3, 0xCD)       # light yellow bg

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
        self.setMinimumSize(400, 300)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(0x2D, 0x2D, 0x2D))
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

    def _safe_paint(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        if self._scale <= 0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # draw image scaled+centered
        pw = int(self._pixmap.width() * self._scale)
        ph = int(self._pixmap.height() * self._scale)
        if pw <= 0 or ph <= 0:
            p.end()
            return
        scaled = self._pixmap.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
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
        self._all_detections: list[dict] = []
        self._ocr_books: list[dict] = []
        self._worker: _ApiWorker | None = None
        self._crop_pixmaps_full: list[QPixmap | None] = []  # full-size crop cache

        self.setWindowTitle("BookPhoto Split — 图书盘点")
        self.resize(1400, 850)
        self._build_ui()
        self._connect_signals()

        # auto-start backend
        self._on_start_backend()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- toolbar row ---
        toolbar = QHBoxLayout()
        self._btn_open = QPushButton("📂 选择图片")
        self._btn_open.setMinimumWidth(100)

        toolbar.addWidget(self._btn_open)
        toolbar.addSpacing(12)

        toolbar.addWidget(QLabel("置信度阈值:"))
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(10, 100)
        self._conf_slider.setValue(25)
        self._conf_slider.setFixedWidth(150)
        toolbar.addWidget(self._conf_slider)

        self._conf_label = QLabel("0.25")
        self._conf_label.setFixedWidth(35)
        toolbar.addWidget(self._conf_label)
        toolbar.addSpacing(24)

        self._backend_led = QLabel()
        self._backend_led.setFixedSize(14, 14)
        self._backend_led.setStyleSheet(
            "border-radius:7px;background-color:#888;"
        )
        toolbar.addWidget(self._backend_led)
        self._backend_status = QLabel("后端未连接")
        toolbar.addWidget(self._backend_status)
        toolbar.addStretch()

        root.addLayout(toolbar)

        # --- body: splitter (image | tabs) ---
        splitter = QSplitter(Qt.Horizontal)

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

        # tab 3: match
        self._tab_match = QWidget()
        self._build_match_tab()
        self._tabs.addTab(self._tab_match, "📚 匹配结果")

        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        root.addWidget(splitter)

        # --- bottom button bar ---
        btn_row = QHBoxLayout()
        self._btn_segment = QPushButton("🔍 分割检测")
        self._btn_segment.setMinimumHeight(36)
        btn_row.addWidget(self._btn_segment)

        self._btn_ocr = QPushButton("📖 OCR 识别")
        self._btn_ocr.setMinimumHeight(36)
        self._btn_ocr.setEnabled(False)
        btn_row.addWidget(self._btn_ocr)

        self._btn_match = QPushButton("📚 馆藏匹配")
        self._btn_match.setMinimumHeight(36)
        self._btn_match.setEnabled(False)
        btn_row.addWidget(self._btn_match)

        self._btn_save = QPushButton("💾 保存结果")
        self._btn_save.setMinimumHeight(36)
        self._btn_save.setEnabled(False)
        btn_row.addWidget(self._btn_save)

        self._btn_export = QPushButton("📤 导出 CSV")
        self._btn_export.setMinimumHeight(36)
        self._btn_export.setEnabled(False)
        btn_row.addWidget(self._btn_export)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # --- status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪 — 点击「选择图片」开始")

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

        hint = QLabel("双击单元格编辑书名和数量，编辑完成后点击「馆藏匹配」")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;")
        layout.addWidget(hint)

        self._ocr_table = QTableWidget(0, 2)
        self._ocr_table.setHorizontalHeaderLabels(["书名", "数量"])
        self._ocr_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._ocr_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self._ocr_table.setColumnWidth(1, 60)
        layout.addWidget(self._ocr_table)

    def _build_match_tab(self) -> None:
        layout = QVBoxLayout(self._tab_match)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._lbl_match_summary = QLabel("尚未匹配")
        self._lbl_match_summary.setWordWrap(True)
        layout.addWidget(self._lbl_match_summary)

        self._match_table = QTableWidget(0, 5)
        self._match_table.setHorizontalHeaderLabels([
            "OCR 书名", "匹配馆藏", "分数", "策略", "需确认",
        ])
        self._match_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        self._btn_match.clicked.connect(self._on_match)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_save.clicked.connect(self._on_save)
        self._crop_list.currentRowChanged.connect(self._on_crop_selected)
        self._btn_save_crop_current.clicked.connect(self._on_save_crop_current)
        self._btn_save_crop_all.clicked.connect(self._on_save_crop_all)

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
        path, _ = QFileDialog.getOpenFileName(
            self, "选择书架照片", "",
            "Images (*.jpg *.jpeg *.png);;All Files (*)",
        )
        if not path:
            return
        self._current_image = Path(path)
        self._preview.load_image(self._current_image)
        self._all_detections.clear()
        self._ocr_books.clear()

        # clear crop list
        self._crop_list.clear()
        self._crop_pixmaps_full.clear()
        self._crop_preview_label.setPixmap(QPixmap())
        self._crop_preview_label.setText("分割后在此显示书脊裁剪图")
        self._btn_save_crop_current.setEnabled(False)
        self._btn_save_crop_all.setEnabled(False)

        # reset tabs
        self._detect_table.setRowCount(0)
        self._lbl_detect_summary.setText("尚未检测")
        self._ocr_table.setRowCount(0)
        self._match_table.setRowCount(0)
        self._lbl_match_summary.setText("尚未匹配")

        # buttons
        self._btn_segment.setEnabled(True)
        self._btn_ocr.setEnabled(False)
        self._btn_match.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_export.setEnabled(False)

        self._status_bar.showMessage(f"已选择: {self._current_image.name}")

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
        """Generate full-size crops for all detections and populate crop list."""
        self._crop_list.clear()
        self._crop_pixmaps_full.clear()

        if self._preview._pixmap is None or self._preview._pixmap.isNull():
            return

        src_pix = self._preview._pixmap

        for idx, det in enumerate(detections):
            # full-size crop (for preview + save)
            full_pix = self._crop_spine_pixmap(src_pix, det, 1000)
            self._crop_pixmaps_full.append(full_pix)

            # small thumbnail for list icon
            thumb_pix = self._crop_spine_pixmap(src_pix, det, 40)
            conf = det["confidence"]

            # list item with icon + text
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
        """Show selected crop in large preview."""
        if row < 0 or row >= len(self._crop_pixmaps_full):
            self._crop_preview_label.setPixmap(QPixmap())
            self._crop_preview_label.setText("分割后在此显示书脊裁剪图")
            self._btn_save_crop_current.setEnabled(False)
            return

        pix = self._crop_pixmaps_full[row]
        if pix is None or pix.isNull():
            self._crop_preview_label.setText("无法生成裁剪图")
            self._btn_save_crop_current.setEnabled(False)
            return

        # fit to preview area (keep aspect ratio)
        avail_w = max(100, self._crop_preview_scroll.viewport().width() - 20)
        avail_h = max(100, self._crop_preview_scroll.viewport().height() - 20)
        scaled = pix.scaled(avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._crop_preview_label.setPixmap(scaled)
        self._btn_save_crop_current.setEnabled(True)

    def _on_save_crop_current(self) -> None:
        """Save the currently selected crop as a transparent PNG."""
        row = self._crop_list.currentRow()
        if row < 0 or row >= len(self._crop_pixmaps_full):
            return
        pix = self._crop_pixmaps_full[row]
        if pix is None or pix.isNull():
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "保存书脊裁剪图",
            f"spine_{row + 1:03d}.png",
            "PNG (*.png);;All Files (*)",
        )
        if not path:
            return
        pix.save(path, "PNG")
        self._status_bar.showMessage(f"已保存: {path}")

    def _on_save_crop_all(self) -> None:
        """Save all visible (conf >= threshold) crops to a directory."""
        conf_thresh = self._conf_slider.value() / 100.0
        visible_indices = [
            i for i, det in enumerate(self._all_detections)
            if det["confidence"] >= conf_thresh and i < len(self._crop_pixmaps_full)
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
            pix = self._crop_pixmaps_full[i]
            if pix is not None and not pix.isNull():
                pix.save(str(out_dir / f"spine_{i + 1:03d}.png"), "PNG")
                n += 1

        self._status_bar.showMessage(f"已保存 {n} 张裁剪图到 {out_dir}")
        QMessageBox.information(self, "保存完成", f"已保存 {n} 张裁剪图到:\n{out_dir}")

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
    # slot: OCR
    # ==================================================================

    def _on_ocr(self) -> None:
        if self._current_image is None:
            return
        self._set_busy(True, "正在 OCR 识别书名（可能需要 30-180 秒）…")
        self._run_worker(api_ocr, self._current_image)

    def _on_ocr_done(self, result: dict) -> None:
        self._set_busy(False)
        books = result.get("books", [])
        self._ocr_books = books
        cnt = len(books)

        self._ocr_table.setRowCount(cnt)
        for i, b in enumerate(books):
            name_item = QTableWidgetItem(b.get("book_name", ""))
            name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
            self._ocr_table.setItem(i, 0, name_item)

            cnt_item = QTableWidgetItem(str(b.get("count", 1)))
            cnt_item.setFlags(cnt_item.flags() | Qt.ItemIsEditable)
            cnt_item.setTextAlignment(Qt.AlignCenter)
            self._ocr_table.setItem(i, 1, cnt_item)

        self._btn_match.setEnabled(cnt > 0)
        self._tabs.setCurrentIndex(1)
        self._status_bar.showMessage(f"OCR 完成: {cnt} 本书名 — 可双击编辑后点击「馆藏匹配」")

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
        self._tabs.setCurrentIndex(2)
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
        if filtered_indices and self._crop_pixmaps_full:
            crops_dir = out_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            n_saved = 0
            for i in filtered_indices:
                if i < len(self._crop_pixmaps_full):
                    pix = self._crop_pixmaps_full[i]
                    if pix is not None and not pix.isNull():
                        pix.save(str(crops_dir / f"spine_{i + 1:03d}.png"), "PNG")
                        n_saved += 1
            if n_saved > 0:
                saved.append(f"裁剪书脊: {n_saved} 张 PNG")

        # 3. detection results JSON
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
        """Collect current OCR table data (post user edits)."""
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
    # helpers
    # ==================================================================

    def _set_busy(self, busy: bool, msg: str = "") -> None:
        for btn in [self._btn_open, self._btn_segment, self._btn_ocr, self._btn_match, self._btn_save]:
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
        """Dispatch to the correct handler based on what call finished last."""
        # Determine which handler to call by inspecting what we have queued
        w = self._worker
        if w is None:
            return
        fn = w._fn  # noqa: private access, intentional
        if fn is api_segment:
            self._on_segment_done(result)
        elif fn is api_ocr:
            self._on_ocr_done(result)
        elif fn is api_inventory:
            self._on_match_done(result)

    def _on_worker_error(self, err: str):
        self._set_busy(False)
        QMessageBox.warning(self, "API 错误", err)
        self._status_bar.showMessage(f"错误: {err}")

    # ==================================================================
    # close
    # ==================================================================

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        self._backend.stop()
        super().closeEvent(event)


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
