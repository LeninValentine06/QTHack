"""
matching_widget.py  —  Impedance Matching Tab  (L and Pi networks only)

Layout
------
┌─────────────────────────────────────────────────────────────────────┐
│  LEFT (280px)                │  RIGHT                               │
│  Z_S / Z_L inputs            │  Smith chart (unmatched → matched)   │
│  ⬇ Load from VNA             │  S11 before/after comparison plot    │
│  f₀ / Z₀                    │  Element values table                │
│  Topology (L / Pi)           │  Design notes                        │
│  Q target                    │                                      │
│  ▶ CALCULATE                 │                                      │
│  ≡ COMPARE ALL               │                                      │
│  → APPLY TO NETWORK EDITOR   │                                      │
│  Comparison table            │                                      │
└─────────────────────────────────────────────────────────────────────┘

"Apply to Network Editor":
  Inserts the computed matching elements BEFORE the existing RLC load
  in the schematic.  Series elements go as series mode, shunt elements
  as shunt mode.  The original RLC antenna remains untouched as the
  last (terminal) component.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QScrollArea, QFrame, QSizePolicy, QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from matching_engine import (
    solve_matching, compare_l_and_pi, sweep_matched_network,
    MatchResult, _si_L, _si_C, _si_f, _si_Z,
)

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False

# ── Palette (matches gui.py) ─────────────────────────────────────────────────
BG0   = "#0d1117";  BG1 = "#161b22";  BG2 = "#21262d";  BG3 = "#30363d"
TEXT1 = "#e6edf3";  TEXT2 = "#8b949e";  TEXT3 = "#484f58"
C_BLUE = "#58a6ff"; C_GREEN = "#3fb950"; C_YELL = "#d29922"; RED = "#f85149"
C_TEAL = "#94e2d5"; C_MAUVE = "#cba6f7"; C_PEACH = "#fab387"
BORD2 = "#30363d"
MONO  = '"Consolas","Courier New",monospace'

# ── Topology list: (display name, engine key) ─────────────────────────────────
TOPOLOGIES = [
    ("L-Network (Auto)",     "L_auto"),
    ("L-Network Low-Pass",   "L_lo_pass"),
    ("L-Network High-Pass",  "L_hi_pass"),
    ("Pi-Network Low-Pass",  "pi_lo_pass"),
    ("Pi-Network High-Pass", "pi_hi_pass"),
]

TOPO_DIAGRAMS = {
    "L_auto":       ("Auto-select: low-pass or high-pass L-network\n"
                     "depending on which gives lower Q.\n\n"
                     "Low-pass:   Z_S ──┬──[X_s]── Z_L\n"
                     "                 [B_p]\n"
                     "                  GND"),
    "L_lo_pass":    ("Low-pass L-network\n\n"
                     "Z_S ──┬──[L]── Z_L     (R_S > R_L)\n"
                     "     [C]\n"
                     "      GND\n\n"
                     "Z_S ──[L]──┬── Z_L     (R_L > R_S)\n"
                     "          [C]\n"
                     "           GND"),
    "L_hi_pass":    ("High-pass L-network\n\n"
                     "Z_S ──┬──[C]── Z_L     (R_S > R_L)\n"
                     "     [L]\n"
                     "      GND"),
    "pi_lo_pass":   ("Low-pass Pi-network\n\n"
                     "Z_S ──┬──[L]──┬── Z_L\n"
                     "     [C1]    [C2]\n"
                     "      GND     GND"),
    "pi_hi_pass":   ("High-pass Pi-network\n\n"
                     "Z_S ──┬──[C]──┬── Z_L\n"
                     "     [L1]    [L2]\n"
                     "      GND     GND"),
}


# =============================================================================
# Small helpers
# =============================================================================

def _field(default="", w=120) -> QLineEdit:
    f = QLineEdit(default)
    f.setFixedHeight(24)
    f.setFixedWidth(w)
    f.setStyleSheet(
        f"background:{BG0};border:1px solid {BORD2};"
        f"color:{TEXT1};font-size:10px;padding:0 4px;"
        f"font-family:{MONO};border-radius:2px;")
    return f


def _lbl(txt, color=TEXT2, size=9, bold=False) -> QLabel:
    lb = QLabel(txt)
    lb.setStyleSheet(
        f"color:{color};font-size:{size}px;"
        + ("font-weight:bold;" if bold else "")
        + f"font-family:{MONO};background:transparent;")
    return lb


def _combo(items) -> QComboBox:
    cb = QComboBox()
    for item in items:
        cb.addItem(item)
    cb.setStyleSheet(
        f"background:{BG0};border:1px solid {BORD2};"
        f"color:{TEXT1};font-size:10px;padding:1px 4px;"
        f"font-family:{MONO};border-radius:2px;")
    cb.setFixedHeight(24)
    return cb


def _section(title, accent=C_BLUE) -> QGroupBox:
    gb = QGroupBox(title)
    gb.setStyleSheet(
        f"QGroupBox{{background:transparent;border:none;"
        f"border-left:2px solid {accent};"
        f"margin-top:16px;padding:2px 4px 4px 8px;}}"
        f"QGroupBox::title{{subcontrol-origin:margin;left:6px;"
        f"color:{TEXT2};font-size:8px;font-weight:bold;"
        f"letter-spacing:1px;text-transform:uppercase;"
        f"background:transparent;}}")
    return gb


def _btn(text, color=C_BLUE, bg=BG2) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(30)
    b.setStyleSheet(
        f"QPushButton{{background:{bg};border:1px solid {color};"
        f"color:{color};font-size:10px;font-family:{MONO};"
        f"border-radius:3px;padding:0 10px;}}"
        f"QPushButton:hover{{background:{color};color:{BG0};}}"
        f"QPushButton:disabled{{color:{TEXT3};border-color:{BORD2};}}")
    return b


def _parse_complex(text: str) -> complex:
    t = text.strip().replace(" ", "")
    if not t:
        raise ValueError("Empty input")
    try:
        return complex(float(t))
    except ValueError:
        pass
    try:
        return complex(t)
    except ValueError:
        raise ValueError(f"Cannot parse '{text}' as impedance")


def _freq_fmt_tick(x, _pos):
    if x <= 0:    return ""
    if x >= 1e9:  return f"{x/1e9:.3g} GHz"
    if x >= 1e6:  return f"{x/1e6:.3g} MHz"
    if x >= 1e3:  return f"{x/1e3:.3g} kHz"
    return f"{x:.3g} Hz"


# =============================================================================
# Mini plot canvases
# =============================================================================

class _S11Canvas(FigureCanvas):
    """S11 before/after comparison plot."""
    def __init__(self, parent=None, dpi=90):
        self.fig = Figure(figsize=(6, 3.0), dpi=dpi, facecolor=BG1)
        self.ax  = self.fig.add_subplot(111)
        self._style()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def _style(self):
        ax = self.ax
        ax.set_facecolor(BG0)
        ax.tick_params(colors=TEXT1, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORD2)
        ax.grid(True, which="both", ls="--", lw=0.4, color=BG2)
        ax.title.set_color(TEXT1)
        ax.xaxis.label.set_color(TEXT2)
        ax.yaxis.label.set_color(TEXT2)

    def plot(self, res_before, res_after, f0):
        self.ax.cla(); self._style()
        ax = self.ax

        def _draw(res, color, label):
            if res is None: return
            freqs = res["frequencies"]
            s11   = res["s11_db"]
            if res.get("sweep_type", "log") == "log":
                ax.set_xscale("log")
            ax.plot(freqs, s11, color=color, lw=1.8, label=label, zorder=3)

        _draw(res_before, C_BLUE,  "Unmatched")
        _draw(res_after,  C_GREEN, "Matched")

        ax.axhline(-10, color=C_YELL, ls="--", lw=0.7, alpha=0.7, label="−10 dB")
        ax.axvline(f0,  color=RED,    ls=":",  lw=0.9, alpha=0.7,
                   label=f"f₀ = {_si_f(f0)}")

        ax.set_xlabel("Frequency", color=TEXT2, fontsize=8)
        ax.set_ylabel("S11 (dB)",  color=TEXT2, fontsize=8)
        ax.set_title("S11: Unmatched vs Matched", color=TEXT1, fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_freq_fmt_tick))
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
        ax.legend(fontsize=6.5, facecolor=BG2, labelcolor=TEXT1,
                  edgecolor=BORD2, loc="upper right")
        self.fig.tight_layout(pad=1.2)
        self.draw()


class _SmithCanvas(FigureCanvas):
    """Mini Smith chart: unmatched load + matched trace."""
    def __init__(self, parent=None, dpi=90):
        self.fig = Figure(figsize=(3.8, 3.8), dpi=dpi, facecolor=BG1)
        self.ax  = self.fig.add_subplot(111)
        self._draw_grid()
        super().__init__(self.fig)
        self.setParent(parent)

    def _draw_grid(self):
        ax = self.ax
        ax.set_facecolor(BG0)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORD2)
        ax.tick_params(colors=TEXT1, labelsize=6)
        ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
        ax.set_aspect("equal")
        ax.set_title("Smith Chart", color=TEXT1, fontsize=8, pad=3)
        ax.set_xlabel("Re(Γ)", color=TEXT2, fontsize=7)
        ax.set_ylabel("Im(Γ)", color=TEXT2, fontsize=7)

        if HAS_SKRF:
            try:
                skrf.plotting.smith(ax=ax, draw_labels=True,
                                    ref_imm=1.0, chart_type="z")
                for ln in ax.lines:
                    ln.set_color(BG3); ln.set_linewidth(0.6); ln.set_zorder(1)
                for txt in ax.texts:
                    txt.set_color(TEXT3); txt.set_fontsize(4.5)
                return
            except Exception:
                pass
        import math
        th = np.linspace(0, 2*math.pi, 300)
        ax.plot(np.cos(th), np.sin(th), color=BORD2, lw=1.0)
        ax.axhline(0, color=BG3, lw=0.5)
        self.fig.tight_layout(pad=1.2)

    def plot_points(self, Z_S, Z_L, Z0, match: MatchResult = None,
                    freqs_override: np.ndarray = None):
        self.ax.cla(); self._draw_grid()
        ax = self.ax

        def _g(Z):
            d = Z + Z0
            return (Z - Z0) / d if abs(d) > 1e-30 else 1+0j

        ax.plot(_g(Z_L).real, _g(Z_L).imag, "o", color=RED, ms=9, zorder=8,
                label=f"Z_L = {_si_Z(Z_L)}")
        ax.plot(_g(Z_S).real, _g(Z_S).imag, "s", color=C_YELL, ms=7, zorder=8,
                label=f"Z_S = {_si_Z(Z_S)}")
        ax.plot(0, 0, "+", color=C_GREEN, ms=10, lw=1.5, zorder=8,
                label=f"Z₀ = {Z0:.4g} Ω")

        if match is not None and match.valid:
            try:
                f0    = match.f0
                if freqs_override is not None:
                    res = sweep_matched_network(
                        match,
                        float(freqs_override[0]), float(freqs_override[-1]),
                        freqs_override=freqs_override)
                else:
                    f_lo  = max(f0 * 0.15, 1e6)
                    f_hi  = f0 * 8.0
                    res   = sweep_matched_network(match, f_lo, f_hi, n=201)
                g     = res["gamma"]
                ax.plot(g.real, g.imag, "-", color=C_TEAL, lw=1.6,
                        alpha=0.85, zorder=6, label="Matched trace")
                idx = int(np.argmin(res["s11_db"]))
                ax.plot(g[idx].real, g[idx].imag, "*", color=C_GREEN,
                        ms=11, zorder=9,
                        label=f"Best match @ {_si_f(res['frequencies'][idx])}")
            except Exception:
                pass

        ax.legend(fontsize=5.5, facecolor=BG2, labelcolor=TEXT1,
                  edgecolor=BORD2, loc="lower left", framealpha=0.9)
        self.fig.tight_layout(pad=1.2)
        self.draw()


# =============================================================================
# Main widget
# =============================================================================

class MatchingWidget(QWidget):
    """
    Impedance matching tab — L and Pi networks.

    Signal
    ------
    network_updated(list) — emitted when user clicks "Apply to Network Editor".
        Payload: list of component dicts the schematic should insert BEFORE
        the existing RLC load:
            {"type": "L"|"C", "value": float, "_shunt": bool}
    """

    network_updated = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_match: MatchResult = None
        self._vna_result = None
        self._vna_rlc_R: float = None
        self._vna_rlc_L: float = None
        self._vna_rlc_C: float = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet(
            f"QWidget{{background:{BG0};color:{TEXT1};"
            f"font-family:{MONO};font-size:10px;}}")
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(self._build_left())
        sp.addWidget(self._build_right())
        sp.setSizes([285, 1100])
        sp.setHandleWidth(2)
        root.addWidget(sp)

    def _build_left(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(285)
        w.setStyleSheet(f"background:{BG1};border-right:1px solid {BORD2};")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"background:{BG1};border:none;")

        inner = QWidget()
        inner.setStyleSheet(f"background:{BG1};")
        v = QVBoxLayout(inner)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)

        # Impedances
        gb_z = _section("Impedances", C_BLUE)
        fl = QFormLayout(gb_z)
        fl.setVerticalSpacing(5)
        fl.setContentsMargins(4, 6, 4, 6)
        self.inp_ZS = _field("50")
        self.inp_ZL = _field("73+0j")
        fl.addRow(_lbl("Z_S (Ω)"), self.inp_ZS)
        fl.addRow(_lbl("Z_L (Ω)"), self.inp_ZL)
        self.btn_from_vna = _btn("⬇  Load Z_L from VNA", color=C_TEAL)
        self.btn_from_vna.setEnabled(False)
        self.btn_from_vna.clicked.connect(self._on_load_from_vna)
        fl.addRow(self.btn_from_vna)
        v.addWidget(gb_z)

        # Frequency & Z0
        gb_f = _section("Frequency & Reference", C_YELL)
        fl2 = QFormLayout(gb_f)
        fl2.setVerticalSpacing(5)
        fl2.setContentsMargins(4, 6, 4, 6)
        self.inp_f0 = _field("321e6")
        self.inp_Z0 = _field("50")
        fl2.addRow(_lbl("f₀ (Hz)"), self.inp_f0)
        fl2.addRow(_lbl("Z₀ (Ω)"),  self.inp_Z0)
        v.addWidget(gb_f)

        # Topology
        gb_t = _section("Topology", C_MAUVE)
        vt = QVBoxLayout(gb_t)
        vt.setContentsMargins(4, 6, 4, 6)
        vt.setSpacing(4)
        self.topo_combo = _combo([t[0] for t in TOPOLOGIES])
        self.topo_combo.currentIndexChanged.connect(self._on_topo_changed)
        vt.addWidget(self.topo_combo)
        self.lbl_diagram = QLabel()
        self.lbl_diagram.setStyleSheet(
            f"color:{TEXT3};font-size:8px;font-family:{MONO};"
            f"background:{BG0};padding:4px;border-radius:2px;")
        self.lbl_diagram.setWordWrap(True)
        vt.addWidget(self.lbl_diagram)
        qrow = QHBoxLayout()
        qrow.addWidget(_lbl("Q target (Pi)"))
        self.inp_Q = _field("5", w=55)
        qrow.addWidget(self.inp_Q)
        qrow.addStretch()
        vt.addLayout(qrow)
        v.addWidget(gb_t)

        # Buttons
        self.btn_calc = _btn("▶  CALCULATE", color=C_GREEN, bg="#1a3a1a")
        self.btn_calc.setFixedHeight(38)
        self.btn_calc.clicked.connect(self._on_calculate)
        v.addWidget(self.btn_calc)

        self.btn_compare = _btn("≡  COMPARE ALL (L + Pi)", color=C_BLUE)
        self.btn_compare.clicked.connect(self._on_compare_all)
        v.addWidget(self.btn_compare)

        self.btn_apply = _btn("→  APPLY TO NETWORK EDITOR", color=C_PEACH)
        self.btn_apply.setEnabled(False)
        self.btn_apply.setToolTip(
            "Insert matching elements BEFORE the RLC load in the schematic")
        self.btn_apply.clicked.connect(self._on_apply_to_network)
        v.addWidget(self.btn_apply)

        # Comparison results table
        gb_r = _section("All Results", C_YELL)
        vr = QVBoxLayout(gb_r)
        vr.setContentsMargins(0, 6, 0, 0)
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Topology", "Q", "BW", "✓"])
        hdr = self.results_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, 4):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setFixedHeight(160)
        self.results_table.setStyleSheet(
            f"QTableWidget{{background:{BG0};border:none;"
            f"gridline-color:{BORD2};color:{TEXT1};font-size:9px;}}"
            f"QTableWidget QHeaderView::section{{background:{BG1};"
            f"color:{TEXT2};padding:2px 4px;"
            f"border-bottom:1px solid {BORD2};}}"
            f"QTableWidget::item{{padding:2px 6px;}}"
            f"QTableWidget::item:selected{{background:{BG2};}}")
        self.results_table.cellClicked.connect(self._on_table_row_clicked)
        vr.addWidget(self.results_table)
        v.addWidget(gb_r)

        v.addStretch()

        self.lbl_status = QLabel("Ready — edit schematic or click Load Z_L from VNA")
        self.lbl_status.setStyleSheet(
            f"color:{TEXT3};font-size:8px;font-family:{MONO};"
            f"padding:4px 8px;border-top:1px solid {BORD2};")
        self.lbl_status.setWordWrap(True)

        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)
        outer.addWidget(self.lbl_status)

        self._on_topo_changed(0)
        return w

    def _build_right(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{BG0};")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.setHandleWidth(2)

        self.smith_canvas = _SmithCanvas(dpi=90)
        top.addWidget(self.smith_canvas)

        ev_w = QWidget()
        ev_w.setStyleSheet(f"background:{BG1};")
        ev_v = QVBoxLayout(ev_w)
        ev_v.setContentsMargins(10, 10, 10, 10)
        ev_v.setSpacing(8)

        ev_v.addWidget(_lbl("ELEMENT VALUES", color=C_BLUE, size=9, bold=True))
        self.ev_table = QTableWidget(0, 2)
        self.ev_table.setHorizontalHeaderLabels(["Element", "Value"])
        self.ev_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.ev_table.verticalHeader().setVisible(False)
        self.ev_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ev_table.setStyleSheet(
            f"QTableWidget{{background:{BG0};border:none;"
            f"gridline-color:{BORD2};color:{TEXT1};font-size:10px;}}"
            f"QTableWidget QHeaderView::section{{background:{BG2};"
            f"color:{TEXT2};padding:3px 6px;"
            f"border-bottom:1px solid {BORD2};}}"
            f"QTableWidget::item{{padding:3px 8px;}}")
        ev_v.addWidget(self.ev_table)

        ev_v.addWidget(_lbl("DESIGN NOTES", color=C_YELL, size=9, bold=True))
        self.notes_box = QTextEdit()
        self.notes_box.setReadOnly(True)
        self.notes_box.setFixedHeight(150)
        self.notes_box.setStyleSheet(
            f"background:{BG0};color:{TEXT2};border:1px solid {BORD2};"
            f"font-size:9px;font-family:{MONO};border-radius:2px;")
        ev_v.addWidget(self.notes_box)
        ev_v.addStretch()

        top.addWidget(ev_w)
        top.setSizes([400, 380])

        self.s11_canvas = _S11Canvas(dpi=90)
        v.addWidget(top, stretch=3)
        v.addWidget(self.s11_canvas, stretch=2)
        return w

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _on_topo_changed(self, idx):
        key = TOPOLOGIES[idx][1]
        self.lbl_diagram.setText(TOPO_DIAGRAMS.get(key, ""))
        self.inp_Q.setEnabled("pi" in key)

    def _parse_inputs(self):
        Z_S = _parse_complex(self.inp_ZS.text())
        Z_L = _parse_complex(self.inp_ZL.text())
        f0  = float(self.inp_f0.text())
        Z0  = float(self.inp_Z0.text())
        Q_t = float(self.inp_Q.text())
        return Z_S, Z_L, f0, Z0, Q_t

    def _on_calculate(self):
        try:
            Z_S, Z_L, f0, Z0, Q_t = self._parse_inputs()
        except Exception as e:
            self._set_status(f"Input error: {e}", error=True); return

        topo = TOPOLOGIES[self.topo_combo.currentIndex()][1]
        try:
            match = solve_matching(Z_S, Z_L, f0, topology=topo,
                                   Z0=Z0, Q_target=Q_t,
                                   rlc_R=self._vna_rlc_R,
                                   rlc_L=self._vna_rlc_L,
                                   rlc_C=self._vna_rlc_C)
        except Exception as e:
            self._set_status(f"Solver error: {e}", error=True); return

        self._last_match = match
        self._display_match(match, Z_S, Z_L, Z0, f0)
        self.btn_apply.setEnabled(match.valid)

    def _on_compare_all(self):
        try:
            Z_S, Z_L, f0, Z0, Q_t = self._parse_inputs()
        except Exception as e:
            self._set_status(f"Input error: {e}", error=True); return

        results = compare_l_and_pi(Z_S, Z_L, f0, Z0=Z0, Q_target=Q_t,
                                   rlc_R=self._vna_rlc_R,
                                   rlc_L=self._vna_rlc_L,
                                   rlc_C=self._vna_rlc_C)
        self._populate_table(results)
        if results:
            self._last_match = results[0]
            self._display_match(results[0], Z_S, Z_L, Z0, f0)
            self.btn_apply.setEnabled(True)
            self._set_status(
                f"Best: {results[0].topology}  Q = {results[0].Q:.3f}")

    def _on_table_row_clicked(self, row, col):
        item = self.results_table.item(row, 0)
        if item is None: return
        topo = item.data(Qt.ItemDataRole.UserRole)
        if topo is None: return
        try:
            Z_S, Z_L, f0, Z0, Q_t = self._parse_inputs()
        except Exception: return
        match = solve_matching(Z_S, Z_L, f0, topology=topo,
                               Z0=Z0, Q_target=Q_t,
                               rlc_R=self._vna_rlc_R,
                               rlc_L=self._vna_rlc_L,
                               rlc_C=self._vna_rlc_C)
        self._last_match = match
        self._display_match(match, Z_S, Z_L, Z0, f0)
        self.btn_apply.setEnabled(match.valid)

    def _on_load_from_vna(self):
        if self._vna_result is not None:
            self.load_from_vna(self._vna_result)

    def _on_apply_to_network(self):
        """
        Emit network_updated with the matching component list.

        The list contains only the matching network elements (L/C),
        NOT the RLC load.  gui.py's handler inserts them BEFORE
        the existing RLC terminal in the schematic.
        """
        if self._last_match is None or not self._last_match.valid:
            return
        # Strip internal keys — only pass type, value, _shunt to gui
        clean = []
        for c in self._last_match.components:
            clean.append({
                "type":   c["type"],
                "value":  c["value"],
                "_shunt": c.get("_shunt", False),
            })
        self.network_updated.emit(clean)

    # ── Display ───────────────────────────────────────────────────────────────

    def _display_match(self, match: MatchResult, Z_S, Z_L, Z0, f0):
        # Element values table
        self.ev_table.setRowCount(len(match.element_values))
        for row, (k, val) in enumerate(match.element_values.items()):
            ki = QTableWidgetItem(k)
            vi = QTableWidgetItem(val)
            ki.setForeground(QBrush(QColor(TEXT2)))
            vi.setForeground(QBrush(QColor(C_BLUE)))
            vi.setFont(QFont("Consolas", 10))
            self.ev_table.setItem(row, 0, ki)
            self.ev_table.setItem(row, 1, vi)

        # Notes
        self.notes_box.setPlainText(
            f"Topology : {match.topology}\n"
            f"Q        : {match.Q:.4f}\n"
            f"BW ≈     : {_si_f(match.bandwidth_hz)}\n"
            f"Z_S      : {_si_Z(Z_S)}\n"
            f"Z_L      : {_si_Z(Z_L)}\n"
            f"Z₀       : {Z0:.4g} Ω\n"
            f"f₀       : {_si_f(f0)}\n"
            f"────────────────────\n"
            f"{match.notes}"
            + (f"\n⚠ {match.error}" if match.error else "")
        )

        # Smith chart
        self.smith_canvas.plot_points(Z_S, Z_L, Z0, match,
                                      freqs_override=(
                                          self._vna_result["frequencies"]
                                          if self._vna_result is not None
                                          else None))

        # S11 comparison — use the VNA's actual sweep range and type so this
        # plot is directly comparable to what the VNA Plots tab will show after
        # a re-sweep.  Fall back to a sensible auto-range only when no VNA
        # result is available yet (e.g. manual Z_L entry before any sweep).
        try:
            if self._vna_result is not None:
                freqs_ref  = self._vna_result["frequencies"]
                f_lo       = float(freqs_ref[0])
                f_hi       = float(freqs_ref[-1])
                n          = len(freqs_ref)
                sweep_type = self._vna_result.get("sweep_type", "log")
            else:
                f_lo       = max(f0 * 0.2, 1e4)
                f_hi       = f0 * 6.0
                n          = 401
                sweep_type = "log"

            if sweep_type == "log":
                freqs = np.logspace(np.log10(max(f_lo, 1.0)),
                                    np.log10(f_hi), n)
            else:
                freqs = np.linspace(f_lo, f_hi, n)

            # Unmatched S11: use frequency-varying Z_L(f) when RLC params are
            # available — same model as sweep_matched_network uses for the load,
            # so the "before" curve is also physically consistent.
            rlc_R = self._vna_rlc_R
            rlc_L = self._vna_rlc_L
            rlc_C = self._vna_rlc_C
            if rlc_R is not None and rlc_L is not None and rlc_C is not None:
                omega_f = 2.0 * np.pi * freqs
                omega_f = np.where(np.abs(omega_f) < 1e-30, 1e-30, omega_f)
                ZL_arr = (rlc_R
                          + 1j * (omega_f * rlc_L - 1.0 / (omega_f * rlc_C)))
            else:
                ZL_arr = np.full(n, Z_L, dtype=complex)

            dL     = ZL_arr + Z0
            dL     = np.where(np.abs(dL) < 1e-30, 1e-30+0j, dL)
            g_un   = (ZL_arr - Z0) / dL
            s11_un = 20 * np.log10(np.clip(np.abs(g_un), 1e-12, None))
            res_before = {"frequencies": freqs, "s11_db": s11_un,
                          "sweep_type": sweep_type}

            res_after = sweep_matched_network(match, f_lo, f_hi, n=n,
                                              sweep_type=sweep_type,
                                              freqs_override=freqs)
            self.s11_canvas.plot(res_before, res_after, f0)
        except Exception as e:
            self._set_status(f"Sweep error: {e}", error=True)

        self._set_status(
            f"✓  {match.topology}   Q = {match.Q:.3f}   "
            f"BW ≈ {_si_f(match.bandwidth_hz)}")

    def _populate_table(self, results):
        self.results_table.setRowCount(len(results))
        for row, mr in enumerate(results):
            t_it = QTableWidgetItem(mr.topology)
            t_it.setData(Qt.ItemDataRole.UserRole, mr.topology)
            q_it = QTableWidgetItem(f"{mr.Q:.3f}")
            b_it = QTableWidgetItem(_si_f(mr.bandwidth_hz))
            v_it = QTableWidgetItem("✓" if mr.valid else "✗")
            t_it.setForeground(QBrush(QColor(C_BLUE)))
            q_it.setForeground(QBrush(QColor(C_YELL)))
            b_it.setForeground(QBrush(QColor(C_GREEN)))
            v_it.setForeground(QBrush(QColor(C_GREEN if mr.valid else RED)))
            for c, it in enumerate([t_it, q_it, b_it, v_it]):
                self.results_table.setItem(row, c, it)

    def _set_status(self, msg, error=False):
        color = RED if error else TEXT3
        self.lbl_status.setStyleSheet(
            f"color:{color};font-size:8px;font-family:{MONO};"
            f"padding:4px 8px;border-top:1px solid {BORD2};")
        self.lbl_status.setText(msg)

    # ── Public API (called by gui.py) ─────────────────────────────────────────

    def sync_rlc(self, Z_L: complex, f0: float, Z0: float,
                 R: float = None, L: float = None, C: float = None):
        """Live sync from schematic edits — no sweep needed."""
        self.inp_ZL.setText(
            f"{Z_L.real:.4g}{'+' if Z_L.imag >= 0 else ''}{Z_L.imag:.4g}j")
        self.inp_ZS.setText(f"{Z0:.4g}")
        self.inp_Z0.setText(f"{Z0:.4g}")
        self.inp_f0.setText(f"{f0:.6g}")

        parts = []
        if R is not None: parts.append(f"R={R:.4g} Ω")
        if L is not None:
            parts.append(f"L={L*1e9:.4g} nH" if L < 1e-6 else f"L={L*1e6:.4g} µH")
        if C is not None:
            parts.append(f"C={C*1e12:.4g} pF" if C < 1e-9 else f"C={C*1e9:.4g} nF")
        self._set_status(
            f"⟳ {' '.join(parts)}  →  "
            f"Z_L={Z_L.real:.4g}{'+' if Z_L.imag>=0 else ''}{Z_L.imag:.4g}j Ω  "
            f"f₀={_si_f(f0)}")

    def load_from_vna(self, result: dict,
                      rlc_R: float = None,
                      rlc_L: float = None,
                      rlc_C: float = None):
        """Called by MainWindow._on_done after every sweep.

        rlc_R/L/C: sidebar antenna values passed directly from gui.py so we
        have them even when the network_config path doesn't yield an RLC.
        """
        self._vna_result = result
        self.btn_from_vna.setEnabled(True)

        bw  = result.get("bandwidth", {})
        f0  = bw.get("f_res")
        idx = int(np.argmin(result["s11_db"]))

        # Z_load_bare = terminal RLC only (not the full cascade input Z)
        Z_arr = result.get("Z_load_bare", result["Z_L"])
        Z     = Z_arr[idx]
        Z0    = result.get("Z0", 50.0)

        # Extract RLC params — priority: network_config RLC > direct sidebar values.
        # These are stored and later threaded into solve_matching so that
        # sweep_matched_network computes Z_L(f) across the full frequency range,
        # giving physically realistic S11 depth (−20 to −35 dB) instead of
        # an ideal mathematical null (−∞ dB).
        self._vna_rlc_R = None
        self._vna_rlc_L = None
        self._vna_rlc_C = None
        network_config = result.get("network_config", [])
        for comp in reversed(network_config):
            if str(comp.get("type", "")).upper() == "RLC":
                self._vna_rlc_R = float(comp.get("R", 0.0))
                self._vna_rlc_L = float(comp.get("L", 1e-30))
                self._vna_rlc_C = float(comp.get("C", 1e-30))
                break
        # Fallback to sidebar values when config has no RLC element
        if self._vna_rlc_R is None and rlc_R is not None:
            self._vna_rlc_R = rlc_R
            self._vna_rlc_L = rlc_L
            self._vna_rlc_C = rlc_C

        if f0: self.inp_f0.setText(f"{f0:.6g}")
        self.inp_ZL.setText(
            f"{Z.real:.4g}{'+' if Z.imag >= 0 else ''}{Z.imag:.4g}j")
        self.inp_ZS.setText(f"{Z0:.4g}")
        self.inp_Z0.setText(f"{Z0:.4g}")

        src = ("Network Editor RLC"
               if result.get("n_components", 1) > 1 else "Antenna Model")
        rlc_note = ""
        if self._vna_rlc_R is not None:
            rlc_note = (f"  RLC={self._vna_rlc_R:.4g}\u03a9/"
                        f"{self._vna_rlc_L*1e9:.4g}nH/"
                        f"{self._vna_rlc_C*1e12:.4g}pF")
        self._set_status(
            f"\u2713 {src}  Z_L={Z.real:.4g}{'+' if Z.imag>=0 else ''}"
            f"{Z.imag:.4g}j \u03a9  f\u2080={_si_f(f0) if f0 else '?'}"
            + rlc_note)