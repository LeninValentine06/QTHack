"""
gui.py  —  PyQt6 main window for the VNA Simulator v2
Four-panel layout: Left inputs | Centre S11 | Right Smith | Bottom results
"""

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFormLayout, QLineEdit, QPushButton, QLabel, QComboBox,
    QGroupBox, QStatusBar, QFileDialog, QSplitter, QMessageBox,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from rf_engine      import run_simulation
from antenna_models import PRESETS
from plot_s11       import S11Canvas, PLOT_MODES, _fmt_freq
from smith_chart    import SmithCanvas
from export_utils   import export_csv


# ─── Unit-aware input widget ──────────────────────────────────────────────────

class UnitField(QWidget):
    """A QLineEdit + QComboBox pair that returns a value in SI base units."""

    FREQ_UNITS = [("Hz", 1e0), ("kHz", 1e3), ("MHz", 1e6), ("GHz", 1e9)]
    IND_UNITS  = [("nH", 1e-9), ("µH", 1e-6), ("mH", 1e-3), ("H", 1e0)]
    CAP_UNITS  = [("pF", 1e-12), ("nF", 1e-9), ("µF", 1e-6), ("F", 1e0)]

    def __init__(self, units: list, default_value: str, default_unit: str,
                 tip: str = "", parent=None):
        super().__init__(parent)
        self._units = units
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._edit = QLineEdit(default_value)
        self._edit.setToolTip(tip)
        self._edit.setFixedHeight(26)

        self._combo = QComboBox()
        self._combo.setFixedHeight(26)
        self._combo.setFixedWidth(54)
        for label, _ in units:
            self._combo.addItem(label)
        idx = next((i for i, (lbl, _) in enumerate(units) if lbl == default_unit), 0)
        self._combo.setCurrentIndex(idx)

        layout.addWidget(self._edit)
        layout.addWidget(self._combo)

    def si_value(self) -> float:
        """Return the numeric value converted to SI base units."""
        raw = float(self._edit.text())
        _, mult = self._units[self._combo.currentIndex()]
        return raw * mult

    def set_si_value(self, si_val: float):
        """Set the field from an SI value, auto-selecting the most readable unit."""
        best_idx = 0
        for i, (_, mult) in enumerate(self._units):
            disp = abs(si_val / mult)
            if 0.1 <= disp < 10000:
                best_idx = i
                break
            if disp >= 0.1:
                best_idx = i
        _, mult = self._units[best_idx]
        self._combo.setCurrentIndex(best_idx)
        self._edit.setText(f"{si_val / mult:.6g}")

    def text(self) -> str:
        return self._edit.text()


# ─── Background worker ────────────────────────────────────────────────────────

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


# ─── Styled widgets ───────────────────────────────────────────────────────────

def _field(default="", tip="") -> QLineEdit:
    f = QLineEdit(default)
    f.setToolTip(tip)
    f.setFixedHeight(26)
    return f

def _label(text: str, bold=False, color="#cdd6f4", size=9) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{color}; font-size:{size}px;"
                      + (" font-weight:bold;" if bold else ""))
    return lbl

def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#45475a;")
    return line


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    DARK_STYLE = """
        QMainWindow,QWidget          { background:#1e1e2e; color:#cdd6f4;
                                       font-family:"Segoe UI",sans-serif; font-size:10px; }
        QGroupBox                    { border:1px solid #45475a; border-radius:6px;
                                       margin-top:10px; padding:6px 4px 4px 4px; }
        QGroupBox::title             { subcontrol-origin:margin; left:8px;
                                       padding:0 4px; color:#89b4fa; font-size:9px; }
        QLineEdit                    { background:#313244; border:1px solid #45475a;
                                       border-radius:4px; padding:2px 6px; color:#cdd6f4; }
        QLineEdit:focus              { border:1px solid #89b4fa; }
        QPushButton                  { background:#313244; border:1px solid #45475a;
                                       border-radius:6px; padding:4px 10px; color:#cdd6f4; }
        QPushButton:hover            { background:#45475a; border-color:#89b4fa; }
        QPushButton:pressed          { background:#89b4fa; color:#1e1e2e; }
        QPushButton#run_btn          { background:#1e3a5f; border-color:#89b4fa;
                                       color:#89b4fa; font-weight:bold; font-size:11px; }
        QPushButton#run_btn:hover    { background:#89b4fa; color:#1e1e2e; }
        QPushButton:disabled         { color:#585b70; border-color:#313244; }
        QComboBox                    { background:#313244; border:1px solid #45475a;
                                       border-radius:4px; padding:2px 6px; color:#cdd6f4; }
        QComboBox::drop-down         { border:none; }
        QComboBox QAbstractItemView  { background:#313244; color:#cdd6f4;
                                       selection-background-color:#45475a; }
        QStatusBar                   { background:#181825; color:#a6adc8; font-size:9px; }
        QSplitter::handle            { background:#45475a; width:2px; }
        QLabel                       { color:#cdd6f4; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VNA Simulator v2  —  Antenna Reflection Analyser")
        self.setMinimumSize(1280, 760)
        self._result = None
        self._worker = None
        self._build_ui()
        self.setStyleSheet(self.DARK_STYLE)
        self._connect_signals()
        self.statusBar().showMessage(
            "Ready  ·  Select a preset or enter RLC values, then press ▶ Run Simulation")

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root   = QWidget()
        vroot  = QVBoxLayout(root)
        vroot.setContentsMargins(6, 6, 6, 4)
        vroot.setSpacing(4)
        self.setCentralWidget(root)

        # ── Top bar ──
        top = QHBoxLayout()
        title = QLabel("VNA Simulator")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color:#89b4fa;")
        sub   = QLabel("Vector Network Analyser  ·  Antenna Reflection Analysis")
        sub.setStyleSheet("color:#a6adc8; font-size:9px; padding-left:8px;")
        top.addWidget(title)
        top.addWidget(sub)
        top.addStretch()
        vroot.addLayout(top)
        vroot.addWidget(_separator())

        # ── Main area (splitter) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_centre_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([230, 620, 420])
        vroot.addWidget(splitter, stretch=3)

        vroot.addWidget(_separator())
        vroot.addWidget(self._build_bottom_panel(), stretch=0)

    # ── Left panel (inputs) ───────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(230)
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(6)

        # Preset
        gb_pre = QGroupBox("Antenna Preset")
        lay_pre = QVBoxLayout(gb_pre)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        lay_pre.addWidget(self.preset_combo)
        self.lbl_preset_note = QLabel("")
        self.lbl_preset_note.setStyleSheet("color:#a6adc8; font-size:8px;")
        self.lbl_preset_note.setWordWrap(True)
        lay_pre.addWidget(self.lbl_preset_note)
        v.addWidget(gb_pre)

        # Frequency sweep
        gb_sw = QGroupBox("Frequency Sweep")
        form_sw = QFormLayout(gb_sw)
        form_sw.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.inp_f_start  = UnitField(UnitField.FREQ_UNITS, "1", "MHz",
                                      "Start frequency")
        self.inp_f_stop   = UnitField(UnitField.FREQ_UNITS, "1", "GHz",
                                      "Stop frequency")
        self.inp_n_points = _field("500", "Number of sweep points (10–2000)")
        form_sw.addRow("Start:",  self.inp_f_start)
        form_sw.addRow("Stop:",   self.inp_f_stop)
        form_sw.addRow("Points:", self.inp_n_points)
        v.addWidget(gb_sw)

        # RLC
        gb_rlc = QGroupBox("Antenna Impedance (RLC Model)")
        form_rlc = QFormLayout(gb_rlc)
        form_rlc.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.inp_R  = _field("73", "Resistance Ω")
        self.inp_L  = UnitField(UnitField.IND_UNITS, "35", "nH", "Inductance")
        self.inp_C  = UnitField(UnitField.CAP_UNITS, "7",  "pF", "Capacitance")
        self.inp_Z0 = _field("50", "Transmission line impedance Ω")
        form_rlc.addRow("R (Ω):",  self.inp_R)
        form_rlc.addRow("L:",      self.inp_L)
        form_rlc.addRow("C:",      self.inp_C)
        form_rlc.addRow("Z₀ (Ω):", self.inp_Z0)
        v.addWidget(gb_rlc)

        v.addStretch()

        # Run button
        self.btn_run = QPushButton("▶  Run Simulation")
        self.btn_run.setObjectName("run_btn")
        self.btn_run.setFixedHeight(40)
        v.addWidget(self.btn_run)

        # Export button
        self.btn_export = QPushButton("💾  Export CSV")
        self.btn_export.setFixedHeight(28)
        self.btn_export.setEnabled(False)
        v.addWidget(self.btn_export)

        return w

    # ── Centre panel (S11 plot) ───────────────────────────────────────────────

    def _build_centre_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 0, 4, 0)

        # Mode selector
        mode_bar = QHBoxLayout()
        mode_bar.addWidget(QLabel("Display:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(PLOT_MODES)
        self.mode_combo.setFixedWidth(180)
        mode_bar.addWidget(self.mode_combo)
        mode_bar.addStretch()
        v.addLayout(mode_bar)

        self.s11_canvas = S11Canvas(dpi=95)
        v.addWidget(self.s11_canvas)
        return w

    # ── Right panel (Smith chart) ─────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 0, 4, 0)
        v.addWidget(QLabel("Smith Chart"))
        self.smith_canvas = SmithCanvas(dpi=95)
        v.addWidget(self.smith_canvas)
        return w

    # ── Bottom panel (results) ────────────────────────────────────────────────

    def _build_bottom_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(130)
        h = QHBoxLayout(w)
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(16)

        # Column helper
        def col(title: str) -> tuple:
            gb  = QGroupBox(title)
            vl  = QVBoxLayout(gb)
            vl.setSpacing(2)
            vl.setContentsMargins(8, 6, 8, 6)
            h.addWidget(gb)
            return vl

        MONO = "font-family:Consolas,monospace; font-size:9px;"

        # Resonance column
        v1 = col("Resonance")
        self.lbl_f_res  = QLabel("Frequency:    —"); self.lbl_f_res.setStyleSheet(MONO)
        self.lbl_s11m   = QLabel("S11 min:      —"); self.lbl_s11m.setStyleSheet(MONO)
        self.lbl_vswr_r = QLabel("VSWR:         —"); self.lbl_vswr_r.setStyleSheet(MONO)
        self.lbl_z_res  = QLabel("Z at res:     —"); self.lbl_z_res.setStyleSheet(MONO)
        for lb in (self.lbl_f_res, self.lbl_s11m, self.lbl_vswr_r, self.lbl_z_res):
            v1.addWidget(lb)

        # Bandwidth column
        v2 = col("−10 dB Bandwidth")
        self.lbl_bw     = QLabel("Bandwidth:    —"); self.lbl_bw.setStyleSheet(MONO)
        self.lbl_bw_lo  = QLabel("f_low:        —"); self.lbl_bw_lo.setStyleSheet(MONO)
        self.lbl_bw_hi  = QLabel("f_high:       —"); self.lbl_bw_hi.setStyleSheet(MONO)
        self.lbl_rl     = QLabel("Return Loss:  —"); self.lbl_rl.setStyleSheet(MONO)
        for lb in (self.lbl_bw, self.lbl_bw_lo, self.lbl_bw_hi, self.lbl_rl):
            v2.addWidget(lb)

        # Marker column
        v3 = col("Active Marker  (click S11 or Smith)")
        self.lbl_mk_f    = QLabel("Frequency:    —"); self.lbl_mk_f.setStyleSheet(MONO)
        self.lbl_mk_s11  = QLabel("S11:          —"); self.lbl_mk_s11.setStyleSheet(MONO)
        self.lbl_mk_vswr = QLabel("VSWR:         —"); self.lbl_mk_vswr.setStyleSheet(MONO)
        self.lbl_mk_z    = QLabel("Z:            —"); self.lbl_mk_z.setStyleSheet(MONO)
        self.lbl_mk_gam  = QLabel("|Γ|:          —"); self.lbl_mk_gam.setStyleSheet(MONO)
        for lb in (self.lbl_mk_f, self.lbl_mk_s11, self.lbl_mk_vswr,
                   self.lbl_mk_z, self.lbl_mk_gam):
            v3.addWidget(lb)

        return w

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.btn_run.clicked.connect(self._on_run)
        self.btn_export.clicked.connect(self._on_export)
        self.preset_combo.currentTextChanged.connect(self._on_preset)
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)
        self.s11_canvas.mpl_connect("button_press_event", self._on_s11_click)
        self.smith_canvas.mpl_connect("button_press_event", self._on_smith_click)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_preset(self, name: str):
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

    def _on_mode_change(self, mode: str):
        self.s11_canvas.set_mode(mode)

    def _on_run(self):
        try:
            params = self._parse_inputs()
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳  Simulating…")
        self.statusBar().showMessage("Running simulation…")

        self._worker = SimWorker(params)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: dict):
        self._result = result
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Simulation")
        self.btn_export.setEnabled(True)

        self.s11_canvas.plot_result(result)
        self.smith_canvas.plot_result(result)
        self._update_results_panel(result)

        bw = result["bandwidth"]
        bw_str = _fmt_freq(bw["bandwidth"]) if bw["valid"] else "—"
        self.statusBar().showMessage(
            f"Done  ·  Resonance {_fmt_freq(bw['f_res'])}  "
            f"·  S11 {bw['s11_min']:.2f} dB  "
            f"·  −10 dB BW {bw_str}"
        )

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Simulation")
        QMessageBox.critical(self, "Simulation Error", msg)
        self.statusBar().showMessage("Error: " + msg)

    def _on_s11_click(self, event):
        if self._result is None or event.xdata is None:
            return
        m = self.s11_canvas.place_marker(event.xdata)
        if m:
            self._update_marker_panel(m)

    def _on_smith_click(self, event):
        if self._result is None or event.xdata is None:
            return
        m = self.smith_canvas.place_marker(event.xdata, event.ydata)
        if m:
            result = self._result
            freqs  = result["frequencies"]
            idx    = int(np.argmin(np.abs(freqs - m["f"])))
            self._update_marker_panel(dict(
                f    = m["f"],
                s11  = result["s11_db"][idx],
                vswr = result["vswr"][idx],
                gmag = abs(m["gamma"]),
                Z_L  = m["Z_L"],
            ))

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

    # ── Results panel update ──────────────────────────────────────────────────

    def _update_results_panel(self, result: dict):
        bw    = result["bandwidth"]
        idx   = int(np.argmin(result["s11_db"]))
        freqs = result["frequencies"]
        Z_res = result["Z_L"][idx]
        rl    = result["return_loss"][idx]

        self.lbl_f_res.setText(f"Frequency:    {_fmt_freq(bw['f_res'])}")
        self.lbl_s11m.setText( f"S11 min:      {bw['s11_min']:.3f} dB")
        self.lbl_vswr_r.setText(f"VSWR:         {result['vswr'][idx]:.4f}")
        self.lbl_z_res.setText(f"Z at res:     {Z_res.real:.2f}{Z_res.imag:+.2f}j Ω")

        if bw["valid"]:
            self.lbl_bw.setText(   f"Bandwidth:    {_fmt_freq(bw['bandwidth'])}")
            self.lbl_bw_lo.setText(f"f_low:        {_fmt_freq(bw['f_low'])}")
            self.lbl_bw_hi.setText(f"f_high:       {_fmt_freq(bw['f_high'])}")
        else:
            self.lbl_bw.setText(   "Bandwidth:    < −10 dB not reached")
            self.lbl_bw_lo.setText("f_low:        —")
            self.lbl_bw_hi.setText("f_high:       —")
        self.lbl_rl.setText(f"Return Loss:  {rl:.3f} dB")

    def _update_marker_panel(self, m: dict):
        Z_L = m["Z_L"]
        Z0  = self._result["Z0"] if self._result else 50.0
        self.lbl_mk_f.setText(   f"Frequency:    {_fmt_freq(m['f'])}")
        self.lbl_mk_s11.setText( f"S11:          {m['s11']:.3f} dB")
        self.lbl_mk_vswr.setText(f"VSWR:         {m['vswr']:.4f}")
        self.lbl_mk_z.setText(   f"Z:            {Z_L.real:.2f}{Z_L.imag:+.2f}j Ω  "
                                 f"(z={Z_L.real/Z0:.3f}{Z_L.imag/Z0:+.3f}j)")
        self.lbl_mk_gam.setText( f"|Γ|:          {m['gmag']:.5f}")

    # ── Input parsing ─────────────────────────────────────────────────────────

    def _parse_inputs(self) -> dict:
        def pf(field, name, positive=True):
            try:
                # UnitField exposes si_value(); plain QLineEdit uses float(text)
                v = field.si_value() if isinstance(field, UnitField) else float(field.text())
            except ValueError:
                raise ValueError(f"'{name}' must be a valid number.")
            if positive and v <= 0:
                raise ValueError(f"'{name}' must be positive.")
            return v

        f_start  = pf(self.inp_f_start,  "Start Frequency")
        f_stop   = pf(self.inp_f_stop,   "Stop Frequency")
        n_points = int(pf(self.inp_n_points, "Sweep Points"))
        R        = pf(self.inp_R,  "Resistance", positive=False)
        L        = pf(self.inp_L,  "Inductance")
        C        = pf(self.inp_C,  "Capacitance")
        Z0       = pf(self.inp_Z0, "Z0")

        if f_stop <= f_start:
            raise ValueError("Stop Frequency must be greater than Start Frequency.")
        if not (10 <= n_points <= 2000):
            raise ValueError("Sweep Points must be between 10 and 2000.")

        return dict(f_start=f_start, f_stop=f_stop, n_points=n_points,
                    R=R, L=L, C=C, Z0=Z0)