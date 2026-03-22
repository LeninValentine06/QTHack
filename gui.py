"""
gui.py  —  VNA Simulator  ·  Redesigned instrument-grade UI

Layout:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Menu bar                                                           │
  ├──────────┬──────────────────────────────┬──────────────────────────┤
  │  LEFT    │  CH1: S11 Log Mag            │  CH2: Mode-selectable    │
  │  sidebar │  (main plot, always S11)     │  (VSWR / phase / etc.)   │
  │          ├──────────────────────────────┴──────────────────────────┤
  │          │  Smith (square)  │  Impedance + BW + Sweep readout     │
  ├──────────┴────────────────────────────────────────────────────────┤
  │  Marker status bar (4 slots, inline, 1-line)                      │
  ├───────────────────────────────────────────────────────────────────┤
  │  Marker table (M1, M2, Δ row when 2 markers active)               │
  └───────────────────────────────────────────────────────────────────┘

Design language: industrial-precision instrument
  - Deep navy base (BG0=#0a0e1a) — darker than Catppuccin Mocha
  - Teal (#2dd4bf) primary accent (CH1, active, links)
  - Amber (#f59e0b) secondary accent (CH2, warnings, delta rows)
  - Left-border accent lines on group panels (no full box borders)
  - ALL-CAPS uppercase label identifiers, monospace value display
  - Glowing green START button
  - Zero decorative elements — every pixel is data or chrome
"""

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFormLayout, QLineEdit, QPushButton, QLabel, QComboBox,
    QGroupBox, QFileDialog, QSplitter, QMessageBox, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QCheckBox, QScrollArea,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QBrush

from rf_engine      import run_simulation
from antenna_models import PRESETS
from plot_s11       import VNACanvas, PLOT_MODES, _fmt_freq
from smith_chart    import SmithCanvas
from export_utils   import export_csv


# ─── Design tokens — strict 4-colour Keysight palette ────────────────────────
#  Blue → S11/CH1/primary  |  Green → Phase/CH2  |  Yellow → markers  |  Grey → refs

BG0   = "#0d1117"   # deepest background
BG1   = "#161b22"   # panel background
BG2   = "#21262d"   # raised surface
BG3   = "#30363d"   # input / hover
BORD  = "#21262d"   # subtle border
BORD2 = "#30363d"   # visible border

TEXT1 = "#e6edf3"   # primary text
TEXT2 = "#8b949e"   # label / secondary
TEXT3 = "#484f58"   # dimmed / disabled

C_BLUE  = "#58a6ff"   # CH1 / S11 accent
C_BLUE2 = "#1f6feb"   # blue dark
C_GREEN = "#3fb950"   # CH2 / Phase accent
C_GREEN2= "#238636"   # green dark
C_YELL  = "#d29922"   # markers / delta / warning
C_YELL2 = "#bb8009"   # yellow dark

# Semantic aliases for rest of code
TEAL   = C_BLUE;    TEAL2  = C_BLUE2
AMBER  = C_YELL;    AMBER2 = C_YELL2
GREEN  = C_GREEN
RED    = "#f85149"
MAUVE  = C_YELL;    PEACH  = C_BLUE;  SKY = C_GREEN;  LIME = "#484f58"

MONO = '"Consolas","Courier New",monospace'

DARK = f"""
QMainWindow, QWidget {{
    background: {BG1}; color: {TEXT1};
    font-family: {MONO}; font-size: 10px;
}}
QMenuBar {{
    background: {BG0}; color: {TEXT2};
    border-bottom: 1px solid {BORD2};
    padding: 1px 2px; font-size: 10px;
}}
QMenuBar::item {{ padding: 3px 8px; border-radius: 2px; }}
QMenuBar::item:selected {{ background: {BG3}; color: {TEXT1}; }}
QMenu {{ background: {BG0}; color: {TEXT1}; border: 1px solid {BORD2}; }}
QMenu::item:selected {{ background: {BG3}; }}

QGroupBox {{
    background: transparent;
    border: none; border-left: 2px solid {C_BLUE2};
    margin-top: 16px; padding: 2px 4px 4px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 6px;
    color: {TEXT2}; font-size: 8px; font-weight: bold;
    letter-spacing: 1px; text-transform: uppercase; background: transparent;
}}

QLineEdit {{
    background: {BG0}; border: 1px solid {BORD2};
    border-radius: 2px; padding: 1px 5px;
    color: {TEXT1}; font-family: {MONO}; font-size: 10px;
}}
QLineEdit:focus {{ border-color: {C_BLUE}; }}

QPushButton {{
    background: {BG2}; border: 1px solid {BORD2};
    border-radius: 3px; padding: 3px 8px;
    color: {TEXT2}; font-size: 10px; font-family: {MONO};
}}
QPushButton:hover {{ background: {BG3}; color: {TEXT1}; border-color: {C_BLUE}; }}
QPushButton:pressed {{ background: {C_BLUE2}; color: {BG0}; }}
QPushButton:disabled {{ color: {TEXT3}; border-color: {BORD}; }}
QPushButton#btn_run {{
    background: {C_GREEN2}; border: 1px solid {C_GREEN};
    color: {C_GREEN}; font-size: 11px; font-weight: bold; letter-spacing: 1px;
}}
QPushButton#btn_run:hover {{ background: #2ea043; color: #d2f3d2; }}
QPushButton#btn_run:disabled {{ background: {BG2}; border-color: {BORD2}; color: {TEXT3}; }}
QPushButton#btn_stop {{ background: {BG2}; border: 1px solid {RED}; color: {RED}; }}
QPushButton#btn_stop:hover {{ background: #3d0a0a; color: #ffa0a0; }}

QComboBox {{
    background: {BG0}; border: 1px solid {BORD2};
    border-radius: 2px; padding: 1px 5px;
    color: {TEXT1}; font-family: {MONO}; font-size: 10px;
}}
QComboBox:focus {{ border-color: {C_BLUE}; }}
QComboBox::drop-down {{ border: none; width: 14px; }}
QComboBox QAbstractItemView {{
    background: {BG0}; color: {TEXT1};
    border: 1px solid {BORD2}; selection-background-color: {BG3};
}}

QTableWidget {{
    background: {BG0}; gridline-color: {BORD2};
    color: {TEXT1}; font-size: 10px; border: none; font-family: {MONO};
}}
QTableWidget QHeaderView::section {{
    background: {BG1}; color: {TEXT2};
    padding: 3px 6px; border: none;
    border-bottom: 1px solid {BORD2}; border-right: 1px solid {BORD2};
    font-size: 9px; font-weight: bold;
    letter-spacing: 0.5px; font-family: {MONO};
}}
QTableWidget::item {{ padding: 3px 8px; border-bottom: 1px solid {BORD}; }}
QTableWidget::item:selected {{ background: {BG3}; }}

QSplitter::handle {{ background: {BORD2}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical   {{ height: 1px; }}

QCheckBox {{ color: {TEXT2}; spacing: 4px; }}
QCheckBox::indicator {{
    width: 12px; height: 12px;
    border: 1px solid {BORD2}; border-radius: 2px; background: {BG0};
}}
QCheckBox::indicator:checked {{ background: {C_BLUE}; border-color: {C_BLUE}; }}

QStatusBar {{
    background: {BG0}; color: {TEXT3};
    border-top: 1px solid {BORD2}; font-size: 9px; font-family: {MONO};
}}
QScrollBar:vertical {{ background: {BG0}; width: 4px; border: none; }}
QScrollBar::handle:vertical {{ background: {BG3}; border-radius: 2px; min-height: 20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _field(default="", tip="", w=None) -> QLineEdit:
    f = QLineEdit(default)
    f.setToolTip(tip)
    f.setFixedHeight(24)
    if w:
        f.setFixedWidth(w)
    return f


def _lbl(txt, color=TEXT2, size=9, bold=False) -> QLabel:
    lb = QLabel(txt)
    lb.setStyleSheet(
        f"color:{color};font-size:{size}px;"
        + ("font-weight:bold;" if bold else "")
        + f"font-family:{MONO};"
    )
    return lb


def _sep_h() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORD2};border:none;max-height:1px;")
    return f


def _make_rule() -> QFrame:
    """1px horizontal rule used between readout panel sections."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet(f"background:{BORD2};border:none;max-height:1px;margin:4px 0 4px 0;")
    return f


def _chan_badge(label: str, color: str) -> QLabel:
    lb = QLabel(label)
    lb.setStyleSheet(
        f"color:{BG0};background:{color};font-size:8px;font-weight:bold;"
        f"padding:1px 6px;border-radius:2px;letter-spacing:1.2px;"
        f"font-family:{MONO};"
    )
    return lb


# ─── Unit field ────────────────────────────────────────────────────────────────

class UnitField(QWidget):
    FREQ_UNITS = [("Hz", 1e0), ("kHz", 1e3), ("MHz", 1e6), ("GHz", 1e9)]
    IND_UNITS  = [("nH", 1e-9), ("µH", 1e-6), ("mH", 1e-3), ("H", 1e0)]
    CAP_UNITS  = [("pF", 1e-12), ("nF", 1e-9), ("µF", 1e-6), ("F", 1e0)]

    def __init__(self, units, default_value, default_unit, tip="", parent=None):
        super().__init__(parent)
        self._units = units
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._edit = QLineEdit(default_value)
        self._edit.setFixedHeight(24)
        self._edit.setToolTip(tip)
        self._combo = QComboBox()
        self._combo.setFixedHeight(24)
        self._combo.setFixedWidth(52)
        for lbl, _ in units:
            self._combo.addItem(lbl)
        self._combo.setCurrentIndex(
            next((i for i, (lb, _) in enumerate(units) if lb == default_unit), 0))
        lay.addWidget(self._edit)
        lay.addWidget(self._combo)

    def si_value(self):
        raw = float(self._edit.text())
        _, m = self._units[self._combo.currentIndex()]
        return raw * m

    def set_si_value(self, v):
        best = 0
        for i, (_, m) in enumerate(self._units):
            d = abs(v / m)
            if 0.1 <= d < 10000:
                best = i
                break
            if d >= 0.1:
                best = i
        _, m = self._units[best]
        self._combo.setCurrentIndex(best)
        self._edit.setText(f"{v/m:.6g}")

    def text(self):
        return self._edit.text()


# ─── Readout card ──────────────────────────────────────────────────────────────

class ReadoutCard(QWidget):
    """
    Flat two-column label/value section.
    No inner card widget — just a title bar + grid rows directly on the
    panel background.  accent_color used only for the title text and a
    1px top rule, so it reads cleanly on any dark background.
    """

    def __init__(self, title: str, rows: list, accent: str = C_BLUE, parent=None):
        super().__init__(parent)
        self._vals: dict[str, QLabel] = {}

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 6, 0, 6)
        v.setSpacing(3)

        # Section title — coloured text, no background box
        t = QLabel(title.upper())
        t.setStyleSheet(
            f"color:{accent};font-size:8px;font-weight:bold;"
            f"letter-spacing:1px;background:transparent;font-family:{MONO};"
            f"padding:0 0 2px 0;border-bottom:1px solid {accent};"
        )
        v.addWidget(t)

        # Two-column grid: label (left, dim) | value (right, bright)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setContentsMargins(0, 2, 0, 0)

        for i, (key, default) in enumerate(rows):
            lk = QLabel(key)
            lk.setStyleSheet(
                f"color:{TEXT3};font-size:9px;background:transparent;"
                f"font-family:{MONO};"
            )
            lv = QLabel(default)
            lv.setStyleSheet(
                f"color:{TEXT1};font-size:10px;font-weight:bold;"
                f"background:transparent;font-family:{MONO};"
            )
            lv.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(lk, i, 0)
            grid.addWidget(lv, i, 1)
            self._vals[key] = lv

        v.addLayout(grid)

    def set(self, key: str, text: str):
        if key in self._vals:
            self._vals[key].setText(text)


# ─── Background worker ─────────────────────────────────────────────────────────

class SimWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.finished.emit(run_simulation(**self.params))
        except Exception as e:
            self.error.emit(str(e))


# ─── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VNA Simulator  ·  RF Analysis Instrument")
        self.setMinimumSize(1440, 900)
        self._result = None
        self._worker = None
        self._sweep_running = False

        self._build_menu()
        self._build_ui()
        self.setStyleSheet(DARK)
        self._connect_signals()
        self.statusBar().showMessage(
            "READY  ·  configure sweep parameters and press ▶ START SWEEP")

    # ── Menu ───────────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        fm = mb.addMenu("File")
        a = QAction("Export CSV…", self); a.triggered.connect(self._on_export)
        fm.addAction(a)
        fm.addSeparator()
        a2 = QAction("Quit", self); a2.triggered.connect(self.close)
        fm.addAction(a2)

        sm = mb.addMenu("Sweep")
        ar = QAction("▶  Start Sweep", self); ar.triggered.connect(self._on_run)
        as_ = QAction("■  Stop",       self); as_.triggered.connect(self._on_stop)
        sm.addActions([ar, as_])

        mm = mb.addMenu("Measurement")
        for mode in PLOT_MODES:
            act = QAction(mode, self)
            act.triggered.connect(lambda checked, m=mode: self._set_secondary_mode(m))
            mm.addAction(act)

        tm = mb.addMenu("Tools")
        ac = QAction("Clear All Markers", self)
        ac.triggered.connect(self._clear_markers)
        tm.addAction(ac)

        hm = mb.addMenu("Help")
        ab = QAction("About", self)
        ab.triggered.connect(lambda: QMessageBox.information(
            self, "About",
            "VNA Simulator  ·  RF Analysis Instrument\n\n"
            "CH1 always shows S11 (dB).\n"
            "CH2 mode is selectable via the header dropdown.\n"
            "Click either plot or the Smith chart to place a marker.\n"
            "Two active markers → Δ row appears automatically."))
        hm.addAction(ab)

    # ── Root layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rv = QVBoxLayout(root)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        # Horizontal root: sidebar | content
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(2)
        h_split.addWidget(self._build_sidebar())
        h_split.addWidget(self._build_content())
        h_split.setSizes([200, 1240])
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)

        rv.addWidget(h_split, stretch=1)
        rv.addWidget(self._build_marker_status_bar())
        rv.addWidget(self._build_marker_table_widget())

    # ── Sidebar ────────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        w = QWidget()
        w.setFixedWidth(210)
        w.setStyleSheet(f"background:{BG0}; border-right:1px solid {BORD2};")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Compact title strip — single line, no wasted space
        title_w = QWidget()
        title_w.setFixedHeight(26)
        title_w.setStyleSheet(
            f"background:{BG0};border-bottom:1px solid {BORD2};"
        )
        tl = QHBoxLayout(title_w)
        tl.setContentsMargins(10, 0, 10, 0)
        t1 = QLabel("VNA SIM")
        t1.setStyleSheet(
            f"color:{C_BLUE};font-size:10px;font-weight:bold;"
            f"letter-spacing:2px;background:transparent;font-family:{MONO};"
        )
        t2 = QLabel("RF Analysis Instrument")
        t2.setStyleSheet(
            f"color:{TEXT3};font-size:8px;background:transparent;"
            f"font-family:{MONO};"
        )
        tl.addWidget(t1)
        tl.addSpacing(8)
        tl.addWidget(t2)
        tl.addStretch()
        v.addWidget(title_w)

        # Scrollable parameter groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background:{BG0};border:none;")

        content = QWidget()
        content.setStyleSheet(f"background:{BG0};")
        cv = QVBoxLayout(content)
        cv.setContentsMargins(8, 8, 8, 8)
        cv.setSpacing(10)
        cv.addWidget(self._build_sweep_group())
        cv.addWidget(self._build_rlc_group())
        cv.addWidget(self._build_preset_group())
        cv.addStretch()
        scroll.setWidget(content)

        v.addWidget(scroll, stretch=1)
        v.addWidget(self._build_buttons())
        return w

    def _build_sweep_group(self):
        gb = QGroupBox("Sweep Control")
        fl = QFormLayout(gb)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        fl.setVerticalSpacing(5)
        fl.setContentsMargins(4, 6, 4, 6)

        self.inp_f_start = UnitField(UnitField.FREQ_UNITS, "50", "MHz")
        self.inp_f_stop  = UnitField(UnitField.FREQ_UNITS, "1",  "GHz")
        self.inp_n_pts   = _field("500", "Sweep points (10–2000)")

        self.inp_if_bw = QComboBox()
        for bw in ["10 Hz","100 Hz","1 kHz","10 kHz","100 kHz","1 MHz"]:
            self.inp_if_bw.addItem(bw)
        self.inp_if_bw.setCurrentIndex(2)

        self.inp_pwr = QComboBox()
        for p in ["-30 dBm","-20 dBm","-10 dBm","0 dBm","+10 dBm"]:
            self.inp_pwr.addItem(p)
        self.inp_pwr.setCurrentIndex(2)

        self.inp_sweep_type = QComboBox()
        self.inp_sweep_type.addItems(["Logarithmic", "Linear"])

        def rl(t):
            lb = QLabel(t)
            lb.setStyleSheet(
                f"color:{TEXT3};font-size:9px;background:transparent;font-family:{MONO};")
            return lb

        fl.addRow(rl("Start Freq"),  self.inp_f_start)
        fl.addRow(rl("Stop Freq"),   self.inp_f_stop)
        fl.addRow(rl("Points"),      self.inp_n_pts)
        fl.addRow(rl("IF BW"),       self.inp_if_bw)
        fl.addRow(rl("Power"),       self.inp_pwr)
        fl.addRow(rl("Sweep Type"),  self.inp_sweep_type)
        return gb

    def _build_rlc_group(self):
        gb = QGroupBox("Antenna Model")
        gb.setStyleSheet(
            f"QGroupBox {{ border-left:2px solid {AMBER2}; }}"
            f"QGroupBox::title {{ color:{AMBER}; }}"
        )
        fl = QFormLayout(gb)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        fl.setVerticalSpacing(5)
        fl.setContentsMargins(4, 6, 4, 6)

        self.inp_R  = _field("73",  "Resistance Ω")
        self.inp_L  = UnitField(UnitField.IND_UNITS, "35", "nH")
        self.inp_C  = UnitField(UnitField.CAP_UNITS, "7",  "pF")
        self.inp_Z0 = _field("50",  "Reference impedance Ω")

        def rl(t):
            lb = QLabel(t)
            lb.setStyleSheet(
                f"color:{TEXT3};font-size:9px;background:transparent;font-family:{MONO};")
            return lb

        fl.addRow(rl("R (Ω)"),   self.inp_R)
        fl.addRow(rl("L"),       self.inp_L)
        fl.addRow(rl("C"),       self.inp_C)
        fl.addRow(rl("Z₀ (Ω)"), self.inp_Z0)
        return gb

    def _build_preset_group(self):
        gb = QGroupBox("Preset")
        lay = QVBoxLayout(gb)
        lay.setContentsMargins(4, 6, 4, 6)
        lay.setSpacing(4)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.lbl_preset_note = QLabel("")
        self.lbl_preset_note.setStyleSheet(
            f"color:{TEXT3};font-size:8px;background:transparent;"
            "font-style:italic;")
        self.lbl_preset_note.setWordWrap(True)
        lay.addWidget(self.preset_combo)
        lay.addWidget(self.lbl_preset_note)
        return gb

    def _build_buttons(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG0};border-top:1px solid {BORD2};")
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        self.btn_run = QPushButton("▶  START SWEEP")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(28)
        self.btn_stop.setEnabled(False)

        self.btn_clr_mk = QPushButton("✕  Markers")
        self.btn_clr_mk.setFixedHeight(28)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_clr_mk)

        self.btn_export = QPushButton("↓  Export CSV")
        self.btn_export.setFixedHeight(28)
        self.btn_export.setEnabled(False)

        self.chk_marker_mode = QCheckBox("Click → place marker")
        self.chk_marker_mode.setChecked(True)

        v.addWidget(self.btn_run)
        v.addLayout(row)
        v.addWidget(self.btn_export)
        v.addWidget(self.chk_marker_mode)
        return w

    # ── Main content ───────────────────────────────────────────────────────────

    def _build_content(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Top row: two plots
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setHandleWidth(2)
        top.addWidget(self._build_ch1_panel())
        top.addWidget(self._build_ch2_panel())
        top.setSizes([760, 480])

        # Bottom row: Smith + readout
        bot = QSplitter(Qt.Orientation.Horizontal)
        bot.setHandleWidth(2)
        bot.addWidget(self._build_smith_panel())
        bot.addWidget(self._build_readout_panel())
        bot.setSizes([540, 690])

        # Vertical split
        vs = QSplitter(Qt.Orientation.Vertical)
        vs.setHandleWidth(2)
        vs.addWidget(top)
        vs.addWidget(bot)
        vs.setSizes([490, 300])

        v.addWidget(vs)
        return w

    def _build_ch1_panel(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG1};")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        hdr = self._make_plot_header(
            badge_text="CH1", badge_color=TEAL,
            title="S11  Return Loss  (dB)"
        )
        v.addWidget(hdr)
        self.main_canvas = VNACanvas(dpi=95)
        v.addWidget(self.main_canvas, stretch=1)
        return w

    def _build_ch2_panel(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG1};")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(22)
        hdr.setStyleSheet(
            f"background:{BG0};"
            f"border-bottom:1px solid {BORD2};"
            f"border-left:3px solid {C_GREEN};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 0, 10, 0)
        hl.setSpacing(8)

        ch2_lbl = QLabel("CH2")
        ch2_lbl.setStyleSheet(
            f"color:{C_GREEN};font-size:8px;font-weight:bold;"
            f"letter-spacing:1px;background:transparent;font-family:{MONO};"
        )
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(PLOT_MODES[1:])   # Log Mag is always CH1
        self.mode_combo.setCurrentText("VSWR")
        self.mode_combo.setFixedWidth(160)
        self.mode_combo.setStyleSheet(
            f"background:{BG2};border:1px solid {BORD2};"
            f"color:{C_GREEN};font-size:9px;padding:1px 5px;"
        )
        hl.addWidget(ch2_lbl)
        hl.addWidget(self.mode_combo)
        hl.addStretch()
        v.addWidget(hdr)

        self.secondary_canvas = VNACanvas(dpi=95)
        # Sync canvas mode with combo initial value so CH2 never
        # defaults to Log Mag / S11-blue on first sweep.
        self.secondary_canvas.set_mode(self.mode_combo.currentText())
        v.addWidget(self.secondary_canvas, stretch=1)
        return w

    def _build_smith_panel(self):
        w = QWidget()
        w.setStyleSheet(f"background:{BG1};")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Thin 2px colour stripe instead of a tall badge header
        stripe = QWidget()
        stripe.setFixedHeight(2)
        stripe.setStyleSheet(f"background:{C_GREEN};border:none;")
        v.addWidget(stripe)

        self.smith_canvas = SmithCanvas(dpi=95)
        v.addWidget(self.smith_canvas, stretch=1)
        return w

    def _make_plot_header(self, badge_text, badge_color, title):
        """
        Compact 22px plot header: coloured left-border stripe + label.
        Keysight-style: no pill badge, just a 3px left accent.
        """
        hdr = QWidget()
        hdr.setFixedHeight(22)
        hdr.setStyleSheet(
            f"background:{BG0};"
            f"border-bottom:1px solid {BORD2};"
            f"border-left:3px solid {badge_color};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 0, 10, 0)
        hl.setSpacing(8)

        # Channel label
        ch = QLabel(badge_text)
        ch.setStyleSheet(
            f"color:{badge_color};font-size:8px;font-weight:bold;"
            f"letter-spacing:1px;background:transparent;font-family:{MONO};"
        )
        t = QLabel(title)
        t.setStyleSheet(
            f"color:{TEXT2};font-size:9px;"
            f"background:transparent;font-family:{MONO};"
        )
        hl.addWidget(ch)
        hl.addWidget(t)
        hl.addStretch()
        return hdr

    # ── Readout panel ──────────────────────────────────────────────────────────

    def _build_readout_panel(self):
        """
        Right-hand info panel — Keysight style: no card boxes, just flat
        sections separated by 1px rules.  All sections share BG0 background.
        Dense two-column layout: label left (TEXT3), value right (TEXT1 bold).
        """
        w = QWidget()
        w.setStyleSheet(f"background:{BG0};border-left:1px solid {BORD2};")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background:{BG0};border:none;")

        inner = QWidget()
        inner.setStyleSheet(f"background:{BG0};")
        v = QVBoxLayout(inner)
        v.setContentsMargins(12, 8, 12, 8)
        v.setSpacing(0)

        # ── Active Marker ────────────────────────────────────────────────────
        self._card_marker = ReadoutCard(
            "Active Marker",
            [("Freq","—"),("S11","—"),("VSWR","—"),
             ("Γ","—"),("Z","—"),("Re(Z)","—"),("Im(Z)","—")],
            accent=C_YELL,
        )
        v.addWidget(self._card_marker)
        v.addWidget(_make_rule())

        # ── Resonance ────────────────────────────────────────────────────────
        self._card_res = ReadoutCard(
            "Resonance",
            [("f0","—"),("S11 min","—"),("VSWR","—"),
             ("Ret. Loss","—"),("Z","—")],
            accent=C_BLUE,
        )
        v.addWidget(self._card_res)
        v.addWidget(_make_rule())

        # ── Bandwidth ────────────────────────────────────────────────────────
        self._card_bw = ReadoutCard(
            "−10 dB Bandwidth",
            [("BW","—"),("f_low","—"),("f_high","—"),("Q","—")],
            accent=C_GREEN,
        )
        v.addWidget(self._card_bw)
        v.addWidget(_make_rule())

        # ── Sweep info ───────────────────────────────────────────────────────
        self._card_sweep = ReadoutCard(
            "Sweep",
            [("Type","—"),("Points","—"),("IF BW","—"),("Power","—")],
            accent=TEXT3,
        )
        v.addWidget(self._card_sweep)

        self.lbl_res_warn = QLabel("")
        self.lbl_res_warn.setWordWrap(True)
        self.lbl_res_warn.setStyleSheet(
            f"color:{C_YELL};font-size:8px;font-family:{MONO};"
            f"padding-top:4px;")
        v.addWidget(self.lbl_res_warn)
        v.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        return w

    # ── Marker status bar (1-line, below main content) ─────────────────────────

    def _build_marker_status_bar(self):
        """
        Single-line instrument status bar between plots and marker table.
        Left: M1 / M2 inline readouts (yellow/blue).
        Right: resonance quick-read (blue).
        Styled exactly like the bottom status strip on a Keysight PNA.
        """
        w = QWidget()
        w.setFixedHeight(24)
        w.setStyleSheet(
            f"background:{BG0};"
            f"border-top:1px solid {BORD2};"
            f"border-bottom:1px solid {BORD2};"
        )
        h = QHBoxLayout(w)
        h.setContentsMargins(10, 0, 10, 0)
        h.setSpacing(0)

        # M1 / M2 cells — yellow then blue (strict 2-colour markers)
        MARKER_STYLE = [
            f"color:{C_YELL};font-size:9px;font-family:{MONO};padding-right:24px;",
            f"color:{C_BLUE};font-size:9px;font-family:{MONO};padding-right:24px;",
            f"color:{TEXT3};font-size:9px;font-family:{MONO};padding-right:24px;",
            f"color:{TEXT3};font-size:9px;font-family:{MONO};padding-right:24px;",
        ]
        self.status_cells = []
        for i in range(4):
            lb = QLabel("—")
            lb.setStyleSheet(MARKER_STYLE[i])
            h.addWidget(lb)
            # pair: (label, colour_string) — colour used when active
            c = [C_YELL, C_BLUE, TEXT2, TEXT2][i]
            self.status_cells.append((lb, c))

        h.addStretch()

        self.lbl_status_res = QLabel("")
        self.lbl_status_res.setStyleSheet(
            f"color:{C_BLUE};font-size:9px;font-family:{MONO};"
        )
        h.addWidget(self.lbl_status_res)
        return w

    # ── Marker table ───────────────────────────────────────────────────────────

    def _build_marker_table_widget(self):
        """
        5-column marker table: #, Frequency, S11, VSWR, Phase.
        Reduced from 9 cols — keeps only what you read at a glance.
        The Δ row appears automatically when exactly 2 markers are active.
        """
        w = QWidget()
        w.setFixedHeight(112)   # header(26) + 3 rows×22 + border(2)
        w.setStyleSheet(f"background:{BG0};")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.marker_table = QTableWidget(0, 5)
        self.marker_table.setHorizontalHeaderLabels([
            "MARKER", "FREQUENCY", "S11 (dB)", "VSWR", "PHASE (°)",
        ])
        hdr = self.marker_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setMinimumSectionSize(80)
        self.marker_table.verticalHeader().setVisible(False)
        self.marker_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.marker_table.setFixedHeight(108)
        self.marker_table.setStyleSheet(
            f"QTableWidget {{ background:{BG0}; border:none; }}"
        )
        v.addWidget(self.marker_table)
        return w

    # ── Signal wiring ──────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_clr_mk.clicked.connect(self._clear_markers)
        self.preset_combo.currentTextChanged.connect(self._on_preset)
        self.mode_combo.currentTextChanged.connect(self._set_secondary_mode)
        self.main_canvas.mpl_connect("button_press_event",  self._on_main_click)
        self.smith_canvas.mpl_connect("button_press_event", self._on_smith_click)

    # ── Handlers ───────────────────────────────────────────────────────────────

    def _on_preset(self, name):
        p = PRESETS.get(name)
        if p is None:
            self.lbl_preset_note.setText("")
            return
        self.inp_R.setText(str(p["R"]))
        self.inp_L.set_si_value(p["L"])
        self.inp_C.set_si_value(p["C"])
        self.inp_f_start.set_si_value(p["f_start"])
        self.inp_f_stop.set_si_value(p["f_stop"])
        self.lbl_preset_note.setText(p.get("note", ""))

    def _set_secondary_mode(self, mode):
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(mode)
        self.mode_combo.blockSignals(False)
        self.secondary_canvas.set_mode(mode)

    def _on_run(self):
        try:
            params = self._parse_inputs()
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))
            return
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_run.setText("⏳  SWEEPING…")
        self._sweep_running = True
        self.statusBar().showMessage("Sweep running…")
        self._worker = SimWorker(params)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
        self._sweep_running = False
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  START SWEEP")
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Sweep stopped.")

    def _on_done(self, result):
        self._result = result
        self._sweep_running = False
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  START SWEEP")
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)

        # Feed both canvases and Smith; secondary uses its own mode
        self.main_canvas.plot_result(result)
        self.secondary_canvas.plot_result(result)
        self.smith_canvas.plot_result(result)

        self._update_readout(result)
        self._update_marker_table()
        self._update_status_bar()

        bw = result["bandwidth"]
        rc = result.get("resonance_check", {})
        bw_str = _fmt_freq(bw["bandwidth"]) if bw["valid"] else "—"
        self.lbl_status_res.setText(
            f"f_res={_fmt_freq(bw['f_res'])}  "
            f"S11={bw['s11_min']:.2f}dB  BW={bw_str}"
        )
        if rc.get("warning"):
            self.lbl_res_warn.setText(rc["message"])
            self.statusBar().showMessage(f"Done  ·  {rc['message']}")
        else:
            self.lbl_res_warn.setText("")
            self.statusBar().showMessage(
                f"Done  ·  f_res={_fmt_freq(bw['f_res'])}  "
                f"S11={bw['s11_min']:.2f} dB  BW={bw_str}")

    def _on_error(self, msg):
        self._sweep_running = False
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  START SWEEP")
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Simulation Error", msg)
        self.statusBar().showMessage("Error: " + msg)

    def _on_main_click(self, event):
        if self._result is None or event.xdata is None:
            return
        if not self.chk_marker_mode.isChecked():
            return
        m = self.main_canvas.place_marker(event.xdata, event.ydata)
        if m:
            self._update_marker_table()
            self._update_status_bar()
            self._update_active_marker_card(m)

    def _on_smith_click(self, event):
        if self._result is None or event.xdata is None:
            return
        m = self.smith_canvas.place_marker(event.xdata, event.ydata)
        if m:
            self._update_marker_table()
            self._update_status_bar()
            self._update_active_marker_card(m)

    def _on_export(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "vna_sweep.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            export_csv(path, self._result)
            self.statusBar().showMessage(f"Exported → {path}")
        except OSError as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _clear_markers(self):
        self.main_canvas.clear_markers()
        self.smith_canvas.clear_markers()
        self._update_marker_table()
        self._update_status_bar()
        for lb, _ in self.status_cells:
            lb.setText("—")
            lb.setStyleSheet(
                f"color:{TEXT3};font-size:9px;font-family:{MONO};")
        for key in ["Freq","S11","VSWR","Γ","Z","Re(Z)","Im(Z)"]:
            self._card_marker.set(key, "—")

    # ── Readout updates ────────────────────────────────────────────────────────

    def _update_active_marker_card(self, m: dict):
        """
        Update the Active Marker readout card.

        Accepts dicts from EITHER source:
          plot_s11 VNACanvas — has: f, s11, vswr, gmag, phase, gd_ns, z_re, z_im
          SmithCanvas        — has: f, s11, vswr, gmag, gamma (complex), Z_L (complex)

        Missing fields are derived so neither source raises a KeyError.
        """
        # phase (degrees) — derive from gamma complex if not directly present
        if "phase" in m:
            phase_deg = float(m["phase"])
        elif "gamma" in m:
            phase_deg = float(np.degrees(np.angle(m["gamma"])))
        else:
            phase_deg = 0.0

        # Re(Z) / Im(Z) — derive from Z_L complex if not directly present
        if "z_re" in m and "z_im" in m:
            z_re = float(m["z_re"])
            z_im = float(m["z_im"])
        elif "Z_L" in m:
            z_re = float(m["Z_L"].real)
            z_im = float(m["Z_L"].imag)
        else:
            z_re, z_im = 0.0, 0.0

        c = self._card_marker
        c.set("Freq",        _fmt_freq(m["f"]))
        c.set("S11",         f"{m['s11']:.3f} dB")
        c.set("VSWR",        self._fmt_vswr(m["vswr"]))
        c.set("Γ",           f"{m['gmag']:.4f} ∠{phase_deg:+.1f}°")
        s = "+" if z_im >= 0 else ""
        c.set("Z",           f"{z_re:.2f}{s}{z_im:.2f}j Ω")
        c.set("Re(Z)",       f"{z_re:.2f} Ω")
        c.set("Im(Z)",       f"{z_im:.2f} Ω")

    def _update_readout(self, result):
        bw  = result["bandwidth"]
        rc  = result.get("resonance_check", {})
        idx = int(np.argmin(result["s11_db"]))
        Z   = result["Z_L"][idx]
        rl  = result["return_loss"][idx]

        f_str = _fmt_freq(bw["f_res"])
        if rc:
            f_str += f"  Δ={rc['deviation_pct']:.3f}%"
        self._card_res.set("f0",        f_str)
        self._card_res.set("S11 min",   f"{bw['s11_min']:.3f} dB")
        self._card_res.set("VSWR",      self._fmt_vswr(result["vswr"][idx]))
        self._card_res.set("Ret. Loss", f"{rl:.3f} dB")
        s = "+" if Z.imag >= 0 else ""
        self._card_res.set("Z",         f"{Z.real:.2f}{s}{Z.imag:.2f}j Ω")

        if bw["valid"]:
            q = bw["f_res"] / bw["bandwidth"] if bw["bandwidth"] > 0 else float("inf")
            self._card_bw.set("BW",       _fmt_freq(bw["bandwidth"]))
            self._card_bw.set("f_low",    _fmt_freq(bw["f_low"]))
            self._card_bw.set("f_high",   _fmt_freq(bw["f_high"]))
            self._card_bw.set("Q",        f"{q:.2f}")
        else:
            self._card_bw.set("BW",       "< −10 dB")
            self._card_bw.set("f_low",    "—")
            self._card_bw.set("f_high",   "—")
            self._card_bw.set("Q",        "—")

        self._card_sweep.set("Type",   result.get("sweep_type","log").capitalize())
        self._card_sweep.set("Points", str(len(result["frequencies"])))
        ifbw = result.get("if_bw_hz", 1000)
        self._card_sweep.set("IF BW",  _fmt_freq(ifbw))
        pwr = result.get("output_power_dbm", -10)
        self._card_sweep.set("Power",  f"{pwr:.0f} dBm")

    def _update_status_bar(self):
        markers = self.main_canvas.get_marker_data()
        for i, (lb, col) in enumerate(self.status_cells):
            if i < len(markers):
                m = markers[i]
                num = m.get("num", i + 1)
                lb.setText(
                    f"M{num}: {_fmt_freq(m['f'])}  "
                    f"S11={m['s11']:.1f}dB  VSWR={m['vswr']:.2f}"
                )
                lb.setStyleSheet(
                    f"color:{col};font-size:9px;font-family:{MONO};")
            else:
                lb.setText("—")
                lb.setStyleSheet(
                    f"color:{TEXT3};font-size:9px;font-family:{MONO};")

    # ── Marker table helpers ───────────────────────────────────────────────────

    @staticmethod
    def _fmt_vswr(v: float) -> str:
        v = float(v)
        return ">9999" if v >= 9999 else f"{v:.3f}"

    @staticmethod
    def _compute_delta(m1, m2):
        phi1 = np.deg2rad(m1["phase"])
        phi2 = np.deg2rad(m2["phase"])
        dphi = float(np.degrees(np.angle(np.exp(1j * (phi2 - phi1)))))
        return dict(
            delta_f     = m2["f"]    - m1["f"],
            delta_s11   = m2["s11"]  - m1["s11"],
            delta_vswr  = m2["vswr"] - m1["vswr"],
            delta_gmag  = m2["gmag"] - m1["gmag"],
            delta_phase = dphi,
        )

    def _update_marker_table(self):
        """
        5-column table: MARKER | FREQUENCY | S11 | VSWR | PHASE
        M1/M2 rows use yellow/blue foreground on their label.
        Δ row uses amber text on dark background.
        """
        # M1=yellow, M2=blue — strictly 2 colours for markers
        MARKER_FG = [QColor(C_YELL), QColor(C_BLUE)]
        DELTA_BG  = QColor("#1a1400")
        DELTA_FG  = QColor(C_YELL)
        FG_NORM   = QColor(TEXT1)
        ROW_BGS   = [QColor(BG0), QColor(BG1)]

        markers = self.main_canvas.get_marker_data()
        two_mode = len(markers) == 2
        self.marker_table.setRowCount(len(markers) + (1 if two_mode else 0))

        def _set(r, c, txt, bg=None, fg=None, align=Qt.AlignmentFlag.AlignCenter):
            it = QTableWidgetItem(txt)
            it.setTextAlignment(align)
            if bg: it.setBackground(QBrush(bg))
            if fg: it.setForeground(QBrush(fg))
            self.marker_table.setItem(r, c, it)

        for row, m in enumerate(markers):
            num  = m.get("num", row + 1)
            rbg  = ROW_BGS[row % 2]
            mfg  = MARKER_FG[(num - 1) % len(MARKER_FG)]
            vals = [
                f"M{num}",
                _fmt_freq(m["f"]),
                f"{m['s11']:.2f} dB",
                self._fmt_vswr(m["vswr"]),
                f"{m['phase']:.1f}°",
            ]
            for c, v in enumerate(vals):
                _set(row, c, v, bg=rbg, fg=(mfg if c == 0 else FG_NORM))

        if two_mode:
            d   = self._compute_delta(markers[0], markers[1])
            df  = d["delta_f"]
            dv  = d["delta_vswr"]
            vswr_str = (f"{'>' if dv > 0 else '<'}9999"
                        if abs(dv) >= 9999 else f"{dv:+.3f}")
            df_str   = ("−" if df < 0 else "+") + _fmt_freq(abs(df))
            dvals = [
                "Δ(2−1)", df_str,
                f"{d['delta_s11']:+.2f} dB",
                vswr_str,
                f"{d['delta_phase']:+.1f}°",
            ]
            for c, v in enumerate(dvals):
                _set(2, c, v, bg=DELTA_BG, fg=DELTA_FG)

    # ── Input parsing ──────────────────────────────────────────────────────────

    def _parse_inputs(self):
        def pf(field, name, positive=True):
            try:
                v = (field.si_value() if isinstance(field, UnitField)
                     else float(field.text()))
            except ValueError:
                raise ValueError(f"'{name}' must be a valid number.")
            if positive and v <= 0:
                raise ValueError(f"'{name}' must be positive.")
            return v

        f_start = pf(self.inp_f_start, "Start Frequency")
        f_stop  = pf(self.inp_f_stop,  "Stop Frequency")
        try:
            n_pts = int(self.inp_n_pts.text())
        except ValueError:
            raise ValueError("'Sweep Points' must be a whole number.")
        R  = pf(self.inp_R,  "Resistance", positive=False)
        L  = pf(self.inp_L,  "Inductance")
        C  = pf(self.inp_C,  "Capacitance")
        Z0 = pf(self.inp_Z0, "Z0")

        if f_stop <= f_start:
            raise ValueError("Stop frequency must be > Start frequency.")
        if not (10 <= n_pts <= 2000):
            raise ValueError("Sweep Points must be 10–2000.")

        if_bw_map = {
            "10 Hz": 10, "100 Hz": 100, "1 kHz": 1e3,
            "10 kHz": 10e3, "100 kHz": 100e3, "1 MHz": 1e6,
        }
        if_bw = if_bw_map.get(self.inp_if_bw.currentText(), 1e3)

        pwr_map = {
            "-30 dBm": -30, "-20 dBm": -20, "-10 dBm": -10,
            "0 dBm": 0, "+10 dBm": 10,
        }
        pwr = pwr_map.get(self.inp_pwr.currentText(), -10)

        sweep = "log" if self.inp_sweep_type.currentIndex() == 0 else "linear"

        return dict(
            f_start=f_start, f_stop=f_stop, n_points=n_pts,
            R=R, L=L, C=C, Z0=Z0,
            sweep_type=sweep, if_bw=if_bw, output_power_dbm=pwr,
        )