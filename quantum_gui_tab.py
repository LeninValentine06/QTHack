"""
quantum_gui_tab.py  —  ⚛ Quantum Systems tab for VNA Simulator v2
===================================================================

A self-contained QWidget subclass that plugs into the existing QTabWidget
in gui.py as a third tab.

Insertion in gui.py (after the Impedance Matching tab):
-------------------------------------------------------
    from quantum_gui_tab import QuantumTab
    self.quantum_tab = QuantumTab()
    self._main_tabs.addTab(self.quantum_tab, "⚛ Quantum")
    self.quantum_tab.sweep_completed.connect(self.smith_canvas.plot_result)

Features
--------
- System selector (QUBIT_SYSTEM_CONFIGS)
- f_start / f_stop / n_points / Z0 controls (same style as existing sweep)
- Coupling strength g (MHz) and anharmonicity α (MHz) inputs
- Cryo chain checkboxes (300K / 4K / 900mK / 100mK)
- ▶ RUN QUANTUM SWEEP button → calls quantum_network.run_quantum_sweep()
- Readout panel: f₀, Q_loaded, Q_int, T₁, χ, dispersive regime badge
- Embedded matplotlib canvas: CH1 = S11 (dB), CH2 = loaded Q vs frequency
  on a twin-y axis
- Emits sweep_completed(dict) signal for Smith chart integration

Imports only: quantum_models, quantum_network, cryo_cable, rf_engine
No circular imports.
"""

from __future__ import annotations
import sys
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QComboBox, QLineEdit, QPushButton,
    QGroupBox, QCheckBox, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QColor

# ── Optional scikit-rf (same pattern as network_engine.py) ───────────────────
try:
    import skrf          # noqa: F401
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False

# ── Matplotlib embedded canvas ────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Quantum modules ───────────────────────────────────────────────────────────
from quantum_models  import QUBIT_SYSTEM_CONFIGS, QUBIT_SYSTEM_FREQ_HINTS
from quantum_network import run_quantum_sweep, compute_q_loaded
from cryo_cable      import build_cryo_chain, total_attenuation_db, CRYO_STAGES
from rf_engine       import compute_q_factor, compute_t1_estimate

_EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens (mirrors gui.py dark theme)
# ─────────────────────────────────────────────────────────────────────────────

BG0   = "#0d1117"
BG1   = "#161b22"
BG2   = "#21262d"
BG3   = "#30363d"
BORD2 = "#30363d"

TEXT1 = "#e6edf3"
TEXT2 = "#8b949e"
TEXT3 = "#484f58"

C_BLUE  = "#58a6ff"
C_GREEN = "#3fb950"
C_YELL  = "#d29922"
C_TEAL  = "#2dd4bf"
C_RED   = "#f85149"
C_MAUVE = "#cba6f7"

MONO = '"Consolas","Courier New",monospace'

_PANEL_STYLE = f"""
QGroupBox {{
    background: transparent;
    border: none; border-left: 2px solid {C_BLUE};
    margin-top: 14px; padding: 4px 6px;
    color: {TEXT2}; font-size: 9px; font-family: {MONO};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px; top: 2px;
    color: {C_BLUE}; font-size: 9px; font-weight: bold;
}}
QLabel {{ color: {TEXT2}; font-size: 10px; font-family: {MONO}; }}
QLineEdit {{
    background: {BG2}; color: {TEXT1}; border: 1px solid {BORD2};
    border-radius: 2px; padding: 2px 4px; font-size: 10px;
    font-family: {MONO};
}}
QComboBox {{
    background: {BG2}; color: {TEXT1}; border: 1px solid {BORD2};
    border-radius: 2px; padding: 2px 4px; font-size: 10px;
    font-family: {MONO};
}}
QComboBox QAbstractItemView {{
    background: {BG0}; color: {TEXT1}; border: 1px solid {BORD2};
    selection-background-color: {BG3};
}}
QCheckBox {{ color: {TEXT2}; font-size: 10px; font-family: {MONO}; }}
QCheckBox::indicator {{ width: 12px; height: 12px;
    border: 1px solid {BORD2}; background: {BG2}; }}
QCheckBox::indicator:checked {{ background: {C_TEAL}; border-color: {C_TEAL}; }}
"""

_BTN_RUN = f"""
QPushButton {{
    background: #1a4a1a; color: #4ade80; border: 1px solid #22c55e;
    border-radius: 3px; font-weight: bold; font-size: 11px;
    font-family: {MONO}; padding: 6px 12px;
}}
QPushButton:hover  {{ background: #1f5c1f; color: #86efac; }}
QPushButton:pressed{{ background: #145214; }}
QPushButton:disabled{{ background: {BG2}; color: {TEXT3};
    border-color: {TEXT3}; }}
"""

_READOUT_LABEL_STYLE = (
    f"color:{TEXT2}; font-size:9px; font-family:{MONO};"
)
_READOUT_VALUE_STYLE = (
    f"color:{TEXT1}; font-size:12px; font-weight:bold; font-family:{MONO};"
)


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread (keeps GUI responsive during sweep)
# ─────────────────────────────────────────────────────────────────────────────

class _SweepWorker(QThread):
    """Run run_quantum_sweep() in a background thread."""
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self._kwargs = kwargs

    def run(self):
        try:
            result = run_quantum_sweep(**self._kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Embedded matplotlib canvas
# ─────────────────────────────────────────────────────────────────────────────

class _QuantumCanvas(FigureCanvas):
    """
    Dual-trace canvas:
      CH1 (left axis, blue)  — S11 (dB)
      CH2 (right axis, teal) — Q_loaded vs frequency (rolling window)
    """

    def __init__(self, parent=None):
        fig = Figure(figsize=(7, 3.8), dpi=95, facecolor=BG1)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._fig  = fig
        self._ax1  = None
        self._ax2  = None
        self._result = None
        self._draw_empty()

    def _draw_empty(self):
        self._fig.clf()
        ax = self._fig.add_subplot(111, facecolor=BG0)
        ax.set_title("No data — run a quantum sweep",
                     color=TEXT2, fontsize=9, fontfamily="monospace")
        ax.tick_params(colors=TEXT3)
        for sp in ax.spines.values():
            sp.set_color(BORD2)
        self.draw()

    def plot_result(self, result: dict):
        """Render S11 (CH1) and a rolling-window Q trace (CH2)."""
        self._result = result
        self._fig.clf()

        ax1 = self._fig.add_subplot(111, facecolor=BG0)
        ax2 = ax1.twinx()
        self._ax1 = ax1
        self._ax2 = ax2

        freqs   = result["frequencies"]
        s11_db  = result["s11_db"]
        f_ghz   = freqs / 1e9

        # CH1 — S11 (dB)
        ax1.plot(f_ghz, s11_db,
                 color=C_BLUE, linewidth=1.8, label="CH1: S11 (dB)", zorder=5)
        ax1.set_xlabel("Frequency (GHz)", color=TEXT2,
                       fontsize=8, fontfamily="monospace")
        ax1.set_ylabel("S11 (dB)", color=C_BLUE,
                       fontsize=8, fontfamily="monospace")
        ax1.tick_params(axis="y", colors=C_BLUE, labelsize=7)
        ax1.tick_params(axis="x", colors=TEXT2, labelsize=7)
        for sp in ax1.spines.values():
            sp.set_color(BORD2)
        ax1.yaxis.label.set_color(C_BLUE)

        # Resonance marker
        res_idx = int(np.argmin(s11_db))
        ax1.axvline(f_ghz[res_idx], color=C_YELL,
                    linewidth=0.8, linestyle="--", alpha=0.7, zorder=3)
        ax1.annotate(
            f"f₀={f_ghz[res_idx]:.4f} GHz",
            xy=(f_ghz[res_idx], s11_db[res_idx]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=6, color=C_YELL, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc=BG0, ec=C_YELL, alpha=0.85),
        )

        # CH2 — rolling-window loaded Q
        window = max(len(freqs) // 30, 5)
        q_trace = _rolling_q(freqs, s11_db, window)
        ax2.plot(f_ghz, q_trace,
                 color=C_TEAL, linewidth=1.3, linestyle="--",
                 label="CH2: Q_loaded (rolling)", zorder=4, alpha=0.85)
        ax2.set_ylabel("Loaded Q (rolling)", color=C_TEAL,
                       fontsize=8, fontfamily="monospace")
        ax2.tick_params(axis="y", colors=C_TEAL, labelsize=7)
        ax2.yaxis.label.set_color(C_TEAL)
        ax2.set_yscale("log")
        for sp in ax2.spines.values():
            sp.set_color(BORD2)

        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=6, facecolor=BG2,
                   labelcolor=TEXT1, edgecolor=BORD2,
                   loc="lower left", framealpha=0.9)

        ax1.set_title("Quantum Sweep — S11 & Loaded Q", color=TEXT1,
                      fontsize=9, fontfamily="monospace")
        self._fig.tight_layout(pad=1.2)
        self.draw()


def _rolling_q(freqs: np.ndarray, s11_db: np.ndarray,
               window: int) -> np.ndarray:
    """
    Compute a rolling-window loaded Q trace (Q = f_centre / BW_3dB in window).
    Returns an array of the same length as freqs; edges use edge values.
    """
    n   = len(freqs)
    out = np.zeros(n)
    hw  = window // 2
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        if hi - lo < 4:
            out[i] = out[i-1] if i > 0 else 1.0
            continue
        q = compute_q_loaded(freqs[lo:hi], s11_db[lo:hi], threshold_db=-3.0)
        out[i] = max(q, 1.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main widget
# ─────────────────────────────────────────────────────────────────────────────

class QuantumTab(QWidget):
    """
    Self-contained Quantum Systems tab widget.

    Signals
    -------
    sweep_completed(dict)
        Emitted after a successful sweep with the full result dict,
        so gui.py can forward it to the Smith chart tab.
    """
    sweep_completed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{BG1}; color:{TEXT1}; "
                           f"font-family:{MONO}; font-size:10px;")
        self._result  = None
        self._worker  = None

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # Left panel (controls)
        left = self._build_left_panel()
        left.setFixedWidth(230)
        root.addWidget(left)

        # Right area (canvas + readout)
        right = QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(self._build_readout_panel())
        right.addWidget(self._build_canvas(), stretch=1)
        root_right = QWidget()
        root_right.setLayout(right)
        root.addWidget(root_right, stretch=1)

    # ── Left panel construction ───────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(_PANEL_STYLE)
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        # System selector
        sys_box = QGroupBox("QUANTUM SYSTEM")
        sys_lay = QVBoxLayout(sys_box)
        self._sys_combo = QComboBox()
        for name in QUBIT_SYSTEM_CONFIGS:
            self._sys_combo.addItem(name)
        self._sys_combo.currentTextChanged.connect(self._on_system_changed)
        sys_lay.addWidget(self._sys_combo)
        v.addWidget(sys_box)

        # Sweep parameters
        sw_box = QGroupBox("SWEEP PARAMETERS")
        sw_lay = QGridLayout(sw_box)
        sw_lay.setHorizontalSpacing(6)
        sw_lay.setVerticalSpacing(4)

        def _row(label, default, row):
            sw_lay.addWidget(QLabel(label), row, 0)
            ed = QLineEdit(default)
            ed.setFixedHeight(22)
            sw_lay.addWidget(ed, row, 1)
            return ed

        self._f_start_ed  = _row("f_start (GHz)", "4.0",  0)
        self._f_stop_ed   = _row("f_stop  (GHz)", "8.0",  1)
        self._n_points_ed = _row("Points",        "400",  2)
        self._z0_ed       = _row("Z₀  (Ω)",       "50.0", 3)
        v.addWidget(sw_box)

        # Quantum parameters
        qp_box = QGroupBox("QUANTUM PARAMETERS")
        qp_lay = QGridLayout(qp_box)
        qp_lay.setHorizontalSpacing(6)
        qp_lay.setVerticalSpacing(4)

        def _qrow(label, default, row):
            qp_lay.addWidget(QLabel(label), row, 0)
            ed = QLineEdit(default)
            ed.setFixedHeight(22)
            qp_lay.addWidget(ed, row, 1)
            return ed

        self._g_ed     = _qrow("g  (MHz)",    "100.0",  0)
        self._alpha_ed = _qrow("α  (MHz)",    "-200.0", 1)
        v.addWidget(qp_box)

        # Cryo chain
        cryo_box = QGroupBox("CRYO ATTENUATION CHAIN")
        cryo_lay = QVBoxLayout(cryo_box)
        self._cryo_checks: dict[str, QCheckBox] = {}
        att_labels = {"300K": "300 K  (0 dB)",
                      "4K":   "4 K    (−20 dB)",
                      "900mK":"900 mK (−10 dB)",
                      "100mK":"100 mK (−20 dB)"}
        for stage in ["300K", "4K", "900mK", "100mK"]:
            cb = QCheckBox(att_labels[stage])
            self._cryo_checks[stage] = cb
            cryo_lay.addWidget(cb)
        v.addWidget(cryo_box)

        # Run button
        self._run_btn = QPushButton("▶  RUN QUANTUM SWEEP")
        self._run_btn.setStyleSheet(_BTN_RUN)
        self._run_btn.setFixedHeight(34)
        self._run_btn.clicked.connect(self._on_run)
        v.addWidget(self._run_btn)

        v.addStretch(1)
        return w

    # ── Readout panel ─────────────────────────────────────────────────────────

    def _build_readout_panel(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background:{BG0}; border-bottom:1px solid {BORD2};")
        w.setFixedHeight(80)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(20)

        def _cell(label: str):
            cell = QWidget()
            cv   = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(1)
            lbl = QLabel(label)
            lbl.setStyleSheet(_READOUT_LABEL_STYLE)
            val = QLabel("—")
            val.setStyleSheet(_READOUT_VALUE_STYLE)
            cv.addWidget(lbl)
            cv.addWidget(val)
            lay.addWidget(cell)
            return val

        self._ro_f0      = _cell("f₀ (GHz)")
        self._ro_ql      = _cell("Loaded Q")
        self._ro_qi      = _cell("Int. Q est.")
        self._ro_t1      = _cell("T₁ (µs)")
        self._ro_chi     = _cell("χ (MHz)")
        self._ro_disp    = _cell("Dispersive")

        lay.addStretch(1)
        return w

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _build_canvas(self) -> QWidget:
        self._canvas = _QuantumCanvas(self)
        return self._canvas

    # ── Slots ─────────────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_system_changed(self, name: str):
        """Auto-populate f_start / f_stop from hint table."""
        hints = QUBIT_SYSTEM_FREQ_HINTS.get(name)
        if hints:
            self._f_start_ed.setText(f"{hints[0]/1e9:.2f}")
            self._f_stop_ed.setText(f"{hints[1]/1e9:.2f}")

    @pyqtSlot()
    def _on_run(self):
        if self._worker is not None and self._worker.isRunning():
            return

        try:
            f_start  = float(self._f_start_ed.text())  * 1e9
            f_stop   = float(self._f_stop_ed.text())   * 1e9
            n_points = int(self._n_points_ed.text())
            Z0       = float(self._z0_ed.text())
            g_hz     = float(self._g_ed.text())   * 1e6
            alpha_hz = float(self._alpha_ed.text()) * 1e6
        except ValueError as exc:
            self._show_error(f"Invalid parameter: {exc}")
            return

        # Cryo chain prefix
        selected_stages = [s for s, cb in self._cryo_checks.items()
                           if cb.isChecked()]
        prefix = build_cryo_chain(selected_stages) if selected_stages else None

        config_name = self._sys_combo.currentText()

        kwargs = dict(
            config_name    = config_name,
            f_start        = f_start,
            f_stop         = f_stop,
            n_points       = n_points,
            Z0             = Z0,
            network_prefix = prefix,
            g_hz           = g_hz,
            alpha_hz       = alpha_hz,
        )

        self._run_btn.setEnabled(False)
        self._run_btn.setText("⏳  RUNNING…")

        self._worker = _SweepWorker(kwargs, parent=self)
        self._worker.finished.connect(self._on_sweep_done)
        self._worker.error.connect(self._on_sweep_error)
        self._worker.start()

    @pyqtSlot(dict)
    def _on_sweep_done(self, result: dict):
        self._result = result
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  RUN QUANTUM SWEEP")

        # Update canvas
        self._canvas.plot_result(result)

        # Update readout panel
        freqs   = result["frequencies"]
        s11_db  = result["s11_db"]
        res_idx = int(np.argmin(s11_db))
        f0_ghz  = float(freqs[res_idx]) / 1e9

        ql   = result.get("q_loaded", 0.0)
        qi   = result.get("q_internal_est", 0.0)
        t1   = result.get("t1_us", 0.0)
        chi  = result.get("chi_mhz", 0.0)
        disp = result.get("dispersive_regime", False)

        self._ro_f0.setText(f"{f0_ghz:.4f}")
        self._ro_ql.setText(f"{ql:.1f}" if ql > 0 else "—")
        self._ro_qi.setText(f"{qi:.1f}" if qi > 0 else "—")
        self._ro_t1.setText(f"{t1:.3f}" if t1 > 0 else "—")
        self._ro_chi.setText(f"{chi:.3f}")

        if disp:
            self._ro_disp.setText("✔ YES")
            self._ro_disp.setStyleSheet(
                _READOUT_VALUE_STYLE + f"color:{C_GREEN};")
        else:
            self._ro_disp.setText("✘ NO")
            self._ro_disp.setStyleSheet(
                _READOUT_VALUE_STYLE + f"color:{C_RED};")

        # Emit for Smith chart
        self.sweep_completed.emit(result)

    @pyqtSlot(str)
    def _on_sweep_error(self, msg: str):
        self._run_btn.setEnabled(True)
        self._run_btn.setText("▶  RUN QUANTUM SWEEP")
        self._show_error(msg)

    def _show_error(self, msg: str):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Quantum Sweep Error", msg)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — headless smoke-test (no GUI)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(_sys.argv)
    tab = QuantumTab()
    tab.resize(1200, 700)
    tab.setWindowTitle("QuantumTab — standalone test")
    tab.show()
    _sys.exit(app.exec())
