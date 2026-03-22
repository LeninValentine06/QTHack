"""
gui.py  —  VNA Simulator  ·  Professional layout matching reference UI

Layout:
  ┌────────────────────────────────────────────────────────┐
  │  Menu bar  (File | Sweep | Measurement | Tools | Help) │
  ├──────────┬─────────────────────────────────────────────┤
  │  LEFT    │  TOP-CENTRE: main plot (tab-selectable)     │
  │  Sweep   │                                             │
  │  Control ├──────────────────────┬──────────────────────┤
  │  RLC     │  Smith Chart         │  Impedance / Marker  │
  │  Params  │                      │  readout panel        │
  ├──────────┴──────────────────────┴──────────────────────┤
  │  Marker table  (bottom bar — all active markers)       │
  └────────────────────────────────────────────────────────┘
"""

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFormLayout, QLineEdit, QPushButton, QLabel, QComboBox,
    QGroupBox, QFileDialog, QSplitter, QMessageBox, QFrame,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QMenuBar, QMenu, QSpinBox, QDoubleSpinBox,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QAction

from rf_engine      import run_simulation
from antenna_models import PRESETS
from plot_s11       import VNACanvas, PLOT_MODES, _fmt_freq
from smith_chart    import SmithCanvas
from export_utils   import export_csv


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _field(default="", tip="", w=None) -> QLineEdit:
    f = QLineEdit(default)
    f.setToolTip(tip)
    f.setFixedHeight(24)
    if w: f.setFixedWidth(w)
    return f

def _lbl(txt, bold=False, color="#cdd6f4", size=9) -> QLabel:
    lb = QLabel(txt)
    lb.setStyleSheet(f"color:{color};font-size:{size}px;" +
                     ("font-weight:bold;" if bold else ""))
    return lb

def _sep() -> QFrame:
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color:#45475a;"); return f

def _mono(lbl: QLabel):
    lbl.setStyleSheet("font-family:Consolas,monospace;font-size:9px;color:#cdd6f4;")
    return lbl


# ─── Unit-aware field ─────────────────────────────────────────────────────────

class UnitField(QWidget):
    FREQ_UNITS = [("Hz",1e0),("kHz",1e3),("MHz",1e6),("GHz",1e9)]
    IND_UNITS  = [("nH",1e-9),("µH",1e-6),("mH",1e-3),("H",1e0)]
    CAP_UNITS  = [("pF",1e-12),("nF",1e-9),("µF",1e-6),("F",1e0)]

    def __init__(self, units, default_value, default_unit, tip="", parent=None):
        super().__init__(parent)
        self._units = units
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(2)
        self._edit  = QLineEdit(default_value); self._edit.setFixedHeight(24)
        self._edit.setToolTip(tip)
        self._combo = QComboBox(); self._combo.setFixedHeight(24); self._combo.setFixedWidth(50)
        for lbl,_ in units: self._combo.addItem(lbl)
        self._combo.setCurrentIndex(
            next((i for i,(lb,_) in enumerate(units) if lb==default_unit), 0))
        lay.addWidget(self._edit); lay.addWidget(self._combo)

    def si_value(self):
        raw = float(self._edit.text())
        _, m = self._units[self._combo.currentIndex()]
        return raw * m

    def set_si_value(self, v):
        best = 0
        for i,(_,m) in enumerate(self._units):
            d = abs(v/m)
            if 0.1 <= d < 10000: best = i; break
            if d >= 0.1: best = i
        _, m = self._units[best]
        self._combo.setCurrentIndex(best)
        self._edit.setText(f"{v/m:.6g}")

    def text(self): return self._edit.text()


# ─── Background simulation worker ─────────────────────────────────────────────

class SimWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, params): super().__init__(); self.params = params

    def run(self):
        try:    self.finished.emit(run_simulation(**self.params))
        except Exception as e: self.error.emit(str(e))


# ─── Main Window ──────────────────────────────────────────────────────────────

DARK = """
QMainWindow,QWidget      { background:#1e1e2e; color:#cdd6f4;
                           font-family:"Segoe UI",sans-serif; font-size:10px; }
QMenuBar                 { background:#181825; color:#cdd6f4; }
QMenuBar::item:selected  { background:#313244; }
QMenu                    { background:#181825; color:#cdd6f4;
                           border:1px solid #45475a; }
QMenu::item:selected     { background:#313244; }
QGroupBox                { border:1px solid #45475a; border-radius:5px;
                           margin-top:10px; padding:5px 3px 3px 3px; }
QGroupBox::title         { subcontrol-origin:margin; left:7px;
                           padding:0 3px; color:#89b4fa; font-size:9px; }
QLineEdit                { background:#313244; border:1px solid #45475a;
                           border-radius:3px; padding:1px 5px; color:#cdd6f4; }
QLineEdit:focus          { border:1px solid #89b4fa; }
QPushButton              { background:#313244; border:1px solid #45475a;
                           border-radius:5px; padding:3px 9px; color:#cdd6f4; }
QPushButton:hover        { background:#45475a; border-color:#89b4fa; }
QPushButton:pressed      { background:#89b4fa; color:#1e1e2e; }
QPushButton#run_btn      { background:#1e3a5f; border-color:#89b4fa;
                           color:#89b4fa; font-weight:bold; font-size:11px; }
QPushButton#run_btn:hover{ background:#89b4fa; color:#1e1e2e; }
QPushButton:disabled     { color:#585b70; border-color:#313244; }
QComboBox                { background:#313244; border:1px solid #45475a;
                           border-radius:3px; padding:1px 5px; color:#cdd6f4; }
QComboBox::drop-down     { border:none; }
QComboBox QAbstractItemView{ background:#313244; color:#cdd6f4;
                             selection-background-color:#45475a; }
QTabWidget::pane         { border:1px solid #45475a; }
QTabBar::tab             { background:#181825; color:#a6adc8;
                           padding:4px 12px; border:1px solid #45475a;
                           border-bottom:none; border-radius:3px 3px 0 0; }
QTabBar::tab:selected    { background:#1e1e2e; color:#89b4fa; }
QTableWidget             { background:#181825; gridline-color:#313244;
                           color:#cdd6f4; font-size:9px; }
QTableWidget QHeaderView::section { background:#313244; color:#a6adc8;
                           padding:3px; border:none; font-size:9px; }
QStatusBar               { background:#181825; color:#a6adc8; font-size:9px; }
QSplitter::handle        { background:#45475a; }
QSpinBox,QDoubleSpinBox  { background:#313244; border:1px solid #45475a;
                           border-radius:3px; padding:1px 4px; color:#cdd6f4; }
QCheckBox                { color:#cdd6f4; }
QCheckBox::indicator     { width:13px; height:13px; border:1px solid #45475a;
                           border-radius:2px; background:#313244; }
QCheckBox::indicator:checked { background:#89b4fa; }
"""


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VNA Simulator  ·  Professional RF Analysis")
        self.setMinimumSize(1400, 840)
        self._result = None
        self._worker = None
        self._sweep_running = False

        self._build_menu()
        self._build_ui()
        self.setStyleSheet(DARK)
        self._connect_signals()
        self.statusBar().showMessage(
            "Ready  ·  Configure sweep parameters and press ▶ Start Sweep")

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu("File")
        a_exp = QAction("Export CSV…", self); a_exp.triggered.connect(self._on_export)
        fm.addAction(a_exp)
        fm.addSeparator()
        a_quit = QAction("Quit", self); a_quit.triggered.connect(self.close)
        fm.addAction(a_quit)

        # Sweep
        sm = mb.addMenu("Sweep")
        a_run  = QAction("▶  Start Sweep", self); a_run.triggered.connect(self._on_run)
        a_stop = QAction("⏹  Stop",        self); a_stop.triggered.connect(self._on_stop)
        sm.addActions([a_run, a_stop])

        # Measurement
        mm = mb.addMenu("Measurement")
        for mode in PLOT_MODES:
            act = QAction(mode, self)
            act.triggered.connect(lambda checked, m=mode: self._set_plot_mode(m))
            mm.addAction(act)

        # Tools
        tm = mb.addMenu("Tools")
        a_clrm = QAction("Clear All Markers", self)
        a_clrm.triggered.connect(self._clear_markers)
        tm.addAction(a_clrm)

        # Help
        hm = mb.addMenu("Help")
        a_about = QAction("About", self)
        a_about.triggered.connect(lambda: QMessageBox.information(
            self, "About", "VNA Simulator v3\nProfessional RF Analysis Tool"))
        hm.addAction(a_about)

    # ── Main UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root  = QWidget(); self.setCentralWidget(root)
        vroot = QVBoxLayout(root)
        vroot.setContentsMargins(4, 4, 4, 2)
        vroot.setSpacing(3)

        # Horizontal splitter: left panel | centre+right
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.addWidget(self._build_left_panel())

        # Right of left: vertical splitter: main plot top | smith+readout bottom
        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.addWidget(self._build_main_plot_panel())
        v_split.addWidget(self._build_lower_panel())
        v_split.setSizes([520, 320])

        h_split.addWidget(v_split)
        h_split.setSizes([240, 1160])
        vroot.addWidget(h_split, stretch=1)

        # Bottom: marker table
        vroot.addWidget(_sep())
        vroot.addWidget(self._build_marker_table())

    # ── LEFT panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self):
        w = QWidget(); w.setFixedWidth(238)
        v = QVBoxLayout(w); v.setContentsMargins(3,3,3,3); v.setSpacing(5)

        # ── Sweep Control ──
        gb_sw = QGroupBox("Sweep Control")
        fs = QFormLayout(gb_sw); fs.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        fs.setVerticalSpacing(4)
        self.inp_f_start = UnitField(UnitField.FREQ_UNITS, "50", "MHz", "Start frequency")
        self.inp_f_stop  = UnitField(UnitField.FREQ_UNITS, "1",  "GHz", "Stop frequency")
        self.inp_n_pts   = _field("500", "Sweep points (10–2000)", w=60)

        # IF BW
        self.inp_if_bw = QComboBox()
        for bw in ["10 Hz","100 Hz","1 kHz","10 kHz","100 kHz","1 MHz"]:
            self.inp_if_bw.addItem(bw)
        self.inp_if_bw.setCurrentIndex(2)   # 1 kHz default

        # Output power
        self.inp_pwr = QComboBox()
        for p in ["-30 dBm","-20 dBm","-10 dBm","0 dBm","+10 dBm"]:
            self.inp_pwr.addItem(p)
        self.inp_pwr.setCurrentIndex(2)   # -10 dBm

        # Sweep type
        self.inp_sweep_type = QComboBox()
        self.inp_sweep_type.addItems(["Logarithmic", "Linear"])

        fs.addRow("Start:",       self.inp_f_start)
        fs.addRow("Stop:",        self.inp_f_stop)
        fs.addRow("Points:",      self.inp_n_pts)
        fs.addRow("IF BW:",       self.inp_if_bw)
        fs.addRow("Output Pwr:",  self.inp_pwr)
        fs.addRow("Sweep Type:",  self.inp_sweep_type)
        v.addWidget(gb_sw)

        # ── Antenna Preset ──
        gb_pre = QGroupBox("Antenna Preset")
        lp = QVBoxLayout(gb_pre); lp.setSpacing(3)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.lbl_preset_note = QLabel("")
        self.lbl_preset_note.setStyleSheet("color:#a6adc8;font-size:8px;")
        self.lbl_preset_note.setWordWrap(True)
        lp.addWidget(self.preset_combo)
        lp.addWidget(self.lbl_preset_note)
        v.addWidget(gb_pre)

        # ── RLC Model ──
        gb_rlc = QGroupBox("Antenna RLC Model")
        fr = QFormLayout(gb_rlc); fr.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        fr.setVerticalSpacing(4)
        self.inp_R  = _field("73",  "Resistance Ω")
        self.inp_L  = UnitField(UnitField.IND_UNITS, "35",  "nH", "Inductance")
        self.inp_C  = UnitField(UnitField.CAP_UNITS, "7",   "pF", "Capacitance")
        self.inp_Z0 = _field("50",  "Reference impedance Ω")
        fr.addRow("R (Ω):",   self.inp_R)
        fr.addRow("L:",        self.inp_L)
        fr.addRow("C:",        self.inp_C)
        fr.addRow("Z₀ (Ω):",  self.inp_Z0)
        v.addWidget(gb_rlc)

        v.addStretch()

        # ── Buttons ──
        self.btn_run  = QPushButton("▶  Start Sweep"); self.btn_run.setObjectName("run_btn")
        self.btn_run.setFixedHeight(38)
        self.btn_stop = QPushButton("⏹  Stop"); self.btn_stop.setFixedHeight(28)
        self.btn_stop.setEnabled(False)
        self.btn_export = QPushButton("💾  Export CSV"); self.btn_export.setFixedHeight(28)
        self.btn_export.setEnabled(False)
        self.btn_clr_mk = QPushButton("✕  Clear Markers"); self.btn_clr_mk.setFixedHeight(28)
        v.addWidget(self.btn_run)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_clr_mk)
        v.addLayout(btn_row)
        v.addWidget(self.btn_export)

        return w

    # ── Main plot (top centre) ─────────────────────────────────────────────────

    def _build_main_plot_panel(self):
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(2,2,2,0); v.setSpacing(2)

        # Mode selector toolbar
        bar = QHBoxLayout(); bar.setSpacing(6)
        bar.addWidget(_lbl("Display:", color="#a6adc8"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(PLOT_MODES)
        self.mode_combo.setFixedWidth(200)
        bar.addWidget(self.mode_combo)
        bar.addWidget(_sep() if False else QFrame())   # spacer trick

        # Add marker button
        self.chk_marker_mode = QCheckBox("Click = Add Marker")
        self.chk_marker_mode.setChecked(True)
        bar.addWidget(self.chk_marker_mode)
        bar.addStretch()
        v.addLayout(bar)

        self.main_canvas = VNACanvas(dpi=95)
        v.addWidget(self.main_canvas, stretch=1)
        return w

    # ── Lower panel: Smith + Readout ──────────────────────────────────────────

    def _build_lower_panel(self):
        w = QWidget()
        h = QHBoxLayout(w); h.setContentsMargins(2,2,2,2); h.setSpacing(4)

        # Smith chart
        smith_wrap = QGroupBox("Smith Chart")
        sv = QVBoxLayout(smith_wrap); sv.setContentsMargins(2,4,2,2)
        self.smith_canvas = SmithCanvas(dpi=90)
        sv.addWidget(self.smith_canvas)
        h.addWidget(smith_wrap, stretch=2)

        # Impedance / measurement readout
        h.addWidget(self._build_readout_panel(), stretch=1)
        return w

    # ── Readout panel ─────────────────────────────────────────────────────────

    def _build_readout_panel(self):
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(2,2,2,2); v.setSpacing(4)

        # Resonance
        gb_res = QGroupBox("Resonance")
        gr = QGridLayout(gb_res); gr.setVerticalSpacing(3)
        self.lbl_f_res   = _mono(QLabel("—"))
        self.lbl_s11_min = _mono(QLabel("—"))
        self.lbl_vswr_r  = _mono(QLabel("—"))
        self.lbl_rl_r    = _mono(QLabel("—"))
        self.lbl_z_res   = _mono(QLabel("—"))
        rows = [("Frequency:", self.lbl_f_res),
                ("S11 min:",   self.lbl_s11_min),
                ("VSWR:",      self.lbl_vswr_r),
                ("Ret. Loss:", self.lbl_rl_r),
                ("Z:",         self.lbl_z_res)]
        for i,(lbl,val) in enumerate(rows):
            gr.addWidget(_lbl(lbl, color="#a6adc8", size=9), i, 0)
            gr.addWidget(val, i, 1)
        v.addWidget(gb_res)

        # Bandwidth
        gb_bw = QGroupBox("−10 dB Bandwidth")
        gb = QGridLayout(gb_bw); gb.setVerticalSpacing(3)
        self.lbl_bw     = _mono(QLabel("—"))
        self.lbl_bw_lo  = _mono(QLabel("—"))
        self.lbl_bw_hi  = _mono(QLabel("—"))
        self.lbl_q      = _mono(QLabel("—"))
        brows = [("BW:", self.lbl_bw),
                 ("f_low:", self.lbl_bw_lo),
                 ("f_high:", self.lbl_bw_hi),
                 ("Q factor:", self.lbl_q)]
        for i,(lb,val) in enumerate(brows):
            gb.addWidget(_lbl(lb, color="#a6adc8", size=9), i, 0)
            gb.addWidget(val, i, 1)
        v.addWidget(gb_bw)

        # Sweep info
        gb_sw = QGroupBox("Sweep Info")
        gs = QGridLayout(gb_sw); gs.setVerticalSpacing(3)
        self.lbl_sw_type = _mono(QLabel("—"))
        self.lbl_sw_pts  = _mono(QLabel("—"))
        self.lbl_sw_ifbw = _mono(QLabel("—"))
        self.lbl_sw_pwr  = _mono(QLabel("—"))
        srows = [("Type:",    self.lbl_sw_type),
                 ("Points:",  self.lbl_sw_pts),
                 ("IF BW:",   self.lbl_sw_ifbw),
                 ("Pwr:",     self.lbl_sw_pwr)]
        for i,(lb,val) in enumerate(srows):
            gs.addWidget(_lbl(lb, color="#a6adc8", size=9), i, 0)
            gs.addWidget(val, i, 1)
        v.addWidget(gb_sw)

        # Resonance check warning
        self.lbl_res_warn = QLabel("")
        self.lbl_res_warn.setWordWrap(True)
        self.lbl_res_warn.setStyleSheet("color:#f9e2af;font-size:8px;")
        v.addWidget(self.lbl_res_warn)

        v.addStretch()
        return w

    # ── Marker table (bottom bar) ──────────────────────────────────────────────

    def _build_marker_table(self):
        w = QWidget(); w.setFixedHeight(120)
        v = QVBoxLayout(w); v.setContentsMargins(4,2,4,2); v.setSpacing(2)
        v.addWidget(_lbl("Markers", bold=True, color="#89b4fa"))

        self.marker_table = QTableWidget(0, 9)
        self.marker_table.setHorizontalHeaderLabels([
            "#", "Frequency", "S11 (dB)", "VSWR", "|Γ|",
            "Phase (°)", "GD (ns)", "Re(Z) Ω", "Im(Z) Ω"
        ])
        self.marker_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.marker_table.verticalHeader().setVisible(False)
        self.marker_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.marker_table.setFixedHeight(95)
        v.addWidget(self.marker_table)
        return w

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_export.clicked.connect(self._on_export)
        self.btn_clr_mk.clicked.connect(self._clear_markers)
        self.preset_combo.currentTextChanged.connect(self._on_preset)
        self.mode_combo.currentTextChanged.connect(self._set_plot_mode)

        self.main_canvas.mpl_connect("button_press_event",  self._on_main_click)
        self.smith_canvas.mpl_connect("button_press_event", self._on_smith_click)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_preset(self, name):
        p = PRESETS.get(name)
        if p is None:
            self.lbl_preset_note.setText(""); return
        self.inp_R.setText(str(p["R"]))
        self.inp_L.set_si_value(p["L"])
        self.inp_C.set_si_value(p["C"])
        self.inp_f_start.set_si_value(p["f_start"])
        self.inp_f_stop.set_si_value(p["f_stop"])
        self.lbl_preset_note.setText(p.get("note", ""))

    def _set_plot_mode(self, mode):
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(mode)
        self.mode_combo.blockSignals(False)
        self.main_canvas.set_mode(mode)

    def _on_run(self):
        try:
            params = self._parse_inputs()
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e)); return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_run.setText("⏳  Sweeping…")
        self.statusBar().showMessage("Running simulation…")
        self._sweep_running = True

        self._worker = SimWorker(params)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
        self._sweep_running = False
        self.btn_run.setEnabled(True); self.btn_run.setText("▶  Start Sweep")
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Sweep stopped.")

    def _on_done(self, result):
        self._result = result
        self._sweep_running = False
        self.btn_run.setEnabled(True); self.btn_run.setText("▶  Start Sweep")
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)

        self.main_canvas.plot_result(result)
        self.smith_canvas.plot_result(result)
        self._update_readout(result)
        self._update_marker_table()

        bw  = result["bandwidth"]
        rc  = result.get("resonance_check", {})
        bw_str = _fmt_freq(bw["bandwidth"]) if bw["valid"] else "—"
        status = (f"Done  ·  f_res={_fmt_freq(bw['f_res'])}  "
                  f"·  S11={bw['s11_min']:.2f} dB  "
                  f"·  BW={bw_str}")
        if rc.get("warning"):
            status += f"  {rc['message']}"
            self.statusBar().setStyleSheet("QStatusBar{color:#f9e2af;}")
        else:
            self.statusBar().setStyleSheet("")
        self.statusBar().showMessage(status)

    def _on_error(self, msg):
        self._sweep_running = False
        self.btn_run.setEnabled(True); self.btn_run.setText("▶  Start Sweep")
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Simulation Error", msg)
        self.statusBar().showMessage("Error: " + msg)

    def _on_main_click(self, event):
        if self._result is None or event.xdata is None: return
        if not self.chk_marker_mode.isChecked(): return
        m = self.main_canvas.place_marker(event.xdata, event.ydata)
        if m:
            self._update_marker_table()

    def _on_smith_click(self, event):
        if self._result is None or event.xdata is None: return
        m = self.smith_canvas.place_marker(event.xdata, event.ydata)
        if m:
            self._update_marker_table()

    def _on_export(self):
        if self._result is None: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "vna_sweep.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            export_csv(path, self._result)
            self.statusBar().showMessage(f"Exported → {path}")
        except OSError as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _clear_markers(self):
        self.main_canvas.clear_markers()
        self.smith_canvas.clear_markers()
        self._update_marker_table()

    # ── Readout update ────────────────────────────────────────────────────────

    def _update_readout(self, result):
        bw  = result["bandwidth"]
        rc  = result.get("resonance_check", {})
        idx = int(np.argmin(result["s11_db"]))
        Z   = result["Z_L"][idx]
        rl  = result["return_loss"][idx]

        f_str = _fmt_freq(bw["f_res"])
        if rc:
            f_str += f"  (Δ={rc['deviation_pct']:.3f}%)"
        self.lbl_f_res.setText(f_str)
        self.lbl_s11_min.setText(f"{bw['s11_min']:.3f} dB")
        self.lbl_vswr_r.setText(f"{result['vswr'][idx]:.4f}")
        self.lbl_rl_r.setText(f"{rl:.3f} dB")
        self.lbl_z_res.setText(f"{Z.real:.2f}{Z.imag:+.2f}j Ω")

        if bw["valid"]:
            self.lbl_bw.setText(_fmt_freq(bw["bandwidth"]))
            self.lbl_bw_lo.setText(_fmt_freq(bw["f_low"]))
            self.lbl_bw_hi.setText(_fmt_freq(bw["f_high"]))
            q = bw["f_res"] / bw["bandwidth"] if bw["bandwidth"] > 0 else float("inf")
            self.lbl_q.setText(f"{q:.2f}")
        else:
            self.lbl_bw.setText("< −10 dB not reached")
            self.lbl_bw_lo.setText("—"); self.lbl_bw_hi.setText("—")
            self.lbl_q.setText("—")

        self.lbl_sw_type.setText(result.get("sweep_type", "log").capitalize())
        self.lbl_sw_pts.setText(str(len(result["frequencies"])))
        ifbw = result.get("if_bw_hz", 1000)
        self.lbl_sw_ifbw.setText(_fmt_freq(ifbw).replace("Hz","Hz"))
        pwr  = result.get("output_power_dbm", -10)
        self.lbl_sw_pwr.setText(f"{pwr:.0f} dBm")

        warn = rc.get("message", "") if rc.get("warning") else ""
        self.lbl_res_warn.setText(warn)

    # ── Marker table ──────────────────────────────────────────────────────────

    def _update_marker_table(self):
        markers = self.main_canvas.get_marker_data()
        self.marker_table.setRowCount(len(markers))
        for row, m in enumerate(markers):
            vals = [
                str(m["num"]) if "num" in m else str(row+1),
                _fmt_freq(m["f"]),
                f"{m['s11']:.2f} dB",
                f"{m['vswr']:.3f}",
                f"{m['gmag']:.4f}",
                f"{m['phase']:.1f}°",
                f"{m['gd_ns']:.3f} ns",
                f"{m['z_re']:.2f}",
                f"{m['z_im']:.2f}",
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.marker_table.setItem(row, col, item)

    # ── Input parsing ─────────────────────────────────────────────────────────

    def _parse_inputs(self):
        def pf(field, name, positive=True):
            try:
                v = field.si_value() if isinstance(field, UnitField) else float(field.text())
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
        R  = pf(self.inp_R, "Resistance", positive=False)
        L  = pf(self.inp_L, "Inductance")
        C  = pf(self.inp_C, "Capacitance")
        Z0 = pf(self.inp_Z0, "Z0")

        if f_stop <= f_start:
            raise ValueError("Stop frequency must be > Start frequency.")
        if not (10 <= n_pts <= 2000):
            raise ValueError("Sweep Points must be 10–2000.")

        # IF BW parse
        if_bw_map = {"10 Hz":10,"100 Hz":100,"1 kHz":1e3,
                     "10 kHz":10e3,"100 kHz":100e3,"1 MHz":1e6}
        if_bw = if_bw_map.get(self.inp_if_bw.currentText(), 1e3)

        # Output power parse
        pwr_map = {"-30 dBm":-30,"-20 dBm":-20,"-10 dBm":-10,
                   "0 dBm":0,"+10 dBm":10}
        pwr = pwr_map.get(self.inp_pwr.currentText(), -10)

        sweep = "log" if self.inp_sweep_type.currentIndex() == 0 else "linear"

        return dict(f_start=f_start, f_stop=f_stop, n_points=n_pts,
                    R=R, L=L, C=C, Z0=Z0,
                    sweep_type=sweep, if_bw=if_bw, output_power_dbm=pwr)