"""
plot_s11.py  —  Multi-mode VNA plot canvas
Supports: Log Mag, Linear Mag, VSWR, Phase, Unwrapped Phase,
          Group Delay, Real Z, Imaginary Z, Polar
Click anywhere on the plot to drop a numbered marker.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
import matplotlib.ticker as ticker

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(
    # Plot backgrounds matched to gui.py navy palette
    bg      = "#161b22",   # BG1 — outer figure
    axes_bg = "#0d1117",   # BG0 — inner axes
    grid    = "#21262d",   # BG2 — grid lines (low opacity)
    border  = "#30363d",   # BORD2 — spine / tick marks
    text    = "#e6edf3",   # TEXT1
    subtext = "#8b949e",   # TEXT2
    # Trace colours aligned with 4-colour palette (same as _TRACE_COLORS)
    blue    = "#58a6ff",   # S11 / Log Mag — CH1
    green   = "#3fb950",   # Phase — CH2 green
    yellow  = "#d29922",   # VSWR / markers
    red     = "#f85149",   # resonance vline / stop marker
    # Kept for backward compat (reference lines, Smith chart, etc.)
    peach   = "#d29922",   # remapped to yellow
    teal    = "#3fb950",   # remapped to green
    mauve   = "#d29922",   # remapped to yellow
    lavender= "#8b949e",   # grey
    sky     = "#58a6ff",   # remapped to blue
    sapphire= "#58a6ff",   # remapped to blue
)

# All available plot modes
PLOT_MODES = [
    "Log Mag (dB)",
    "Linear Mag",
    "VSWR",
    "Phase (deg)",
    "Unwrapped Phase",
    "Group Delay (ns)",
    "Real Z (Ω)",
    "Imaginary Z (Ω)",
    "Polar",
]

# Fixed y-axis limits for consistent VNA display
Y_LIMITS = {
    "Log Mag (dB)":      (-40.0,  0.0),
    "Linear Mag":        (0.0,    1.0),
    "VSWR":              (1.0,   50.0),
    "Phase (deg)":       (-180.0, 180.0),
    "Unwrapped Phase":   None,   # auto — can span many cycles
    "Group Delay (ns)":  None,   # auto
    "Real Z (Ω)":        None,   # auto
    "Imaginary Z (Ω)":   None,   # auto
    "Polar":             None,   # handled specially
}

# Reference lines per mode: (value, color, label)
REF_LINES = {
    "Log Mag (dB)": [
        (-6,  "#484f58", "−6 dB"),     # grey — least critical
        (-10, "#d29922", "−10 dB"),    # yellow — match marker/warning colour
        (-20, "#3fb950", "−20 dB"),    # green — good match indicator
    ],
    "Linear Mag": [
        (0.316, "#d29922", "0.316 (−10 dB)"),
        (0.100, "#3fb950", "0.100 (−20 dB)"),
    ],
    "VSWR": [
        (2.0, "#3fb950", "VSWR = 2.0"),
        (3.0, "#d29922", "VSWR = 3.0"),
    ],
    "Phase (deg)":      [],
    "Unwrapped Phase":  [],
    "Group Delay (ns)": [],
    "Real Z (Ω)":       [],
    "Imaginary Z (Ω)":  [],
    "Polar":            [],
}

# M1=yellow, M2=blue — strictly 2 primary marker colours.
# Cycles for M3-M8 but those are rare in practice.
MARKER_COLORS = ["#d29922", "#58a6ff", "#3fb950", "#8b949e",
                 "#d29922", "#58a6ff", "#3fb950", "#8b949e"]
MAX_MARKERS = 8


def _fmt_freq(f: float) -> str:
    if f >= 1e9:  return f"{f/1e9:.4f} GHz"
    if f >= 1e6:  return f"{f/1e6:.4f} MHz"
    if f >= 1e3:  return f"{f/1e3:.4f} kHz"
    return f"{f:.2f} Hz"


def _freq_fmt_tick(x, _pos):
    """
    Human-readable frequency tick label — always returns a clean unit string.
    Handles both log and linear sweeps across any frequency range.
    Examples: 100 MHz, 1 GHz, 2.4 GHz, 500 kHz.
    """
    if x <= 0:
        return ""
    if x >= 1e9:
        v = x / 1e9
        # Show 1 decimal only when fractional part is significant
        return f"{v:.3g} GHz"
    if x >= 1e6:
        return f"{x/1e6:.3g} MHz"
    if x >= 1e3:
        return f"{x/1e3:.3g} kHz"
    return f"{x:.3g} Hz"


def _configure_freq_axis(ax, freqs, sweep):
    """
    Set up a clean frequency X axis for both log and linear sweeps.

    For log sweeps:  FormatStrFormatter is overridden with our _freq_fmt_tick
                     so ticks always show '1 GHz', '2 GHz' etc., never '1e9'.
    For linear sweeps: same formatter, with MaxNLocator for even spacing.
    """
    import matplotlib.ticker as _ticker

    formatter = _ticker.FuncFormatter(_freq_fmt_tick)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(_ticker.NullFormatter())

    if sweep == "log":
        ax.set_xscale("log")
        # Replace matplotlib's default LogLocator (which may emit 1e9 labels)
        # with one that hits decade multiples: 1, 2, 5 × 10^n
        ax.xaxis.set_major_locator(
            _ticker.LogLocator(base=10.0, numticks=8))
        ax.xaxis.set_minor_locator(
            _ticker.LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9],
                               numticks=50))
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(_ticker.MaxNLocator(nbins=8, prune="both"))

    ax.tick_params(axis="x", labelrotation=25, labelsize=7)


class VNACanvas(FigureCanvas):
    """
    Multi-mode VNA plot canvas with up to 8 numbered markers.
    Polar mode uses a separate polar subplot.
    """

    def __init__(self, parent=None, dpi=95):
        self.fig = Figure(figsize=(7, 4), dpi=dpi, facecolor=C["bg"])
        self._ax_cart  = self.fig.add_subplot(111)           # Cartesian
        self._ax_polar = None                                # created on demand
        self._result   = None
        self._mode     = "Log Mag (dB)"
        self._markers  = []   # list of dict: {idx, artists, color, num}
        self._is_polar = False

        self._style_ax(self._ax_cart)
        super().__init__(self.fig)
        self.setParent(parent)

    # ── Axes styling ──────────────────────────────────────────────────────────

    def _style_ax(self, ax, polar=False):
        ax.set_facecolor(C["axes_bg"])
        if not polar:
            ax.tick_params(colors=C["text"], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
            ax.grid(True, which="both", linestyle="--",
                    linewidth=0.4, color=C["grid"])
        else:
            ax.tick_params(colors=C["text"], labelsize=7)
            ax.grid(True, linestyle="--", linewidth=0.4, color=C["grid"])
        ax.title.set_color(C["text"])
        ax.xaxis.label.set_color(C["subtext"])
        ax.yaxis.label.set_color(C["subtext"])

    # ── Public API ────────────────────────────────────────────────────────────

    def set_mode(self, mode: str):
        self._mode = mode
        self._markers.clear()
        if self._result is not None:
            self._redraw()

    def plot_result(self, result: dict):
        self._result = result
        self._markers.clear()
        self._redraw()

    def clear_markers(self):
        self._markers.clear()
        if self._result is not None:
            self._redraw()

    # ── Data selector ─────────────────────────────────────────────────────────

    # Strict 4-colour palette matching gui.py:
    #   Blue  (#58a6ff) → S11 / Log Mag (CH1)
    #   Green (#3fb950) → Phase / Unwrapped Phase
    #   Yellow(#d29922) → VSWR / Group Delay / linear quantities
    #   Grey  (#8b949e) → Z plots (least used)
    _TRACE_COLORS = {
        "Log Mag (dB)":    "#58a6ff",   # blue  — primary S11
        "Linear Mag":      "#58a6ff",   # blue  — same channel
        "VSWR":            "#d29922",   # yellow — mismatch metric
        "Phase (deg)":     "#3fb950",   # green — phase
        "Unwrapped Phase": "#3fb950",   # green
        "Group Delay (ns)":"#d29922",   # yellow
        "Real Z (Ω)":      "#8b949e",   # grey
        "Imaginary Z (Ω)": "#8b949e",   # grey
    }

    def _get_y(self, result, mode):
        col = self._TRACE_COLORS.get(mode, "#58a6ff")
        if mode == "Log Mag (dB)":       return result["s11_db"],          "S11 (dB)",        col
        if mode == "Linear Mag":          return np.abs(result["gamma"]),   "|Γ|",             col
        if mode == "VSWR":                return np.minimum(result["vswr"], 50.0), "VSWR",            col
        if mode == "Phase (deg)":
            # Insert NaN at wrap boundaries so matplotlib draws a gap instead of
            # a vertical line jumping ±360° across the wrap point.
            ph = result["phase_deg"].copy().astype(float)
            ph[np.abs(np.diff(ph, prepend=ph[0])) > 180.0] = np.nan
            return ph, "Phase (°)", col
        if mode == "Unwrapped Phase":     return result["phase_unwrapped"],  "Phase (°)",       col
        if mode == "Group Delay (ns)":    return result["group_delay_ns"],   "Group Delay (ns)",col
        if mode == "Real Z (Ω)":          return result["z_real"],           "Re(Z) (Ω)",       col
        if mode == "Imaginary Z (Ω)":     return result["z_imag"],           "Im(Z) (Ω)",       col
        return None, "", "#58a6ff"

    # ── Core drawing ─────────────────────────────────────────────────────────

    def _redraw(self):
        result = self._result
        mode   = self._mode
        freqs  = result["frequencies"]
        sweep  = result.get("sweep_type", "log")

        self._is_polar = (mode == "Polar")

        # ── Switch between polar and Cartesian subplots ───────────────────────
        if self._is_polar:
            self.fig.clear()
            self._ax_polar = self.fig.add_subplot(111, projection="polar")
            self._ax_cart  = None
            self._draw_polar(result)
        else:
            self.fig.clear()
            self._ax_cart  = self.fig.add_subplot(111)
            self._ax_polar = None
            self._draw_cartesian(result, mode, freqs, sweep)

        self.fig.tight_layout(pad=1.6)
        self.draw()

    def _draw_cartesian(self, result, mode, freqs, sweep):
        ax = self._ax_cart
        self._style_ax(ax)

        y, ylabel, ycolor = self._get_y(result, mode)

        # X axis — use centralised helper for consistent units across both canvases
        _configure_freq_axis(ax, freqs, sweep)
        ax.set_xlabel("Frequency (log scale)" if sweep == "log" else "Frequency")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Frequency")

        ax.plot(freqs, y, color=ycolor, linewidth=1.8, zorder=3)

        # Y limits
        ylim = Y_LIMITS.get(mode)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ymin, ymax = float(y.min()), float(y.max())
            span = ymax - ymin
            if span < 1e-6:
                # Flat signal (e.g. Re(Z)=73 Ω constant) — pad by 20% of the
                # value magnitude so the axis isn't a ±1 Ω window that makes
                # sub-pixel rendering noise look like huge oscillations.
                pad = max(abs(ymax) * 0.20, 1.0)
            else:
                pad = span * 0.12
            ax.set_ylim(ymin - pad, ymax + pad)

        # Reference lines
        y_lo, y_hi = ax.get_ylim()
        for ref, col, lbl in REF_LINES.get(mode, []):
            if y_lo <= ref <= y_hi:
                ax.axhline(ref, color=col, linestyle="--",
                           linewidth=0.7, alpha=0.65, label=lbl)

        # Resonance vline (all Cartesian modes)
        bw    = result["bandwidth"]
        f_res = bw["f_res"]
        ax.axvline(f_res, color=C["red"], linestyle=":",
                   linewidth=1.0, alpha=0.75, zorder=2,
                   label=f"Res {_fmt_freq(f_res)}")

        # -10 dB BW shading (log mag only)
        if mode == "Log Mag (dB)" and bw["valid"]:
            ax.axvspan(bw["f_low"], bw["f_high"],
                       alpha=0.10, color=C["blue"], zorder=1)
            ax.axvline(bw["f_low"],  color=C["blue"],
                       linestyle="--", linewidth=0.8, alpha=0.45)
            ax.axvline(bw["f_high"], color=C["blue"],
                       linestyle="--", linewidth=0.8, alpha=0.45)

        ax.legend(fontsize=6.5, facecolor=C["grid"], labelcolor=C["text"],
                  edgecolor=C["border"], loc="upper right")

        # Re-draw persistent markers
        self._redraw_markers_cartesian(result, mode, freqs, y)

    def _draw_polar(self, result):
        ax     = self._ax_polar
        gamma  = result["gamma"]
        mag    = np.abs(gamma)
        ang    = np.angle(gamma)

        self._style_ax(ax, polar=True)
        ax.set_title("Polar  (Γ plane)", color=C["text"], pad=12)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_tick_params(labelsize=6, colors=C["subtext"])
        ax.xaxis.set_tick_params(labelsize=7, colors=C["text"])

        # Unit circle
        theta_c = np.linspace(0, 2*np.pi, 400)
        ax.plot(theta_c, np.ones(400), color=C["border"],
                linewidth=0.8, linestyle="--")

        # Trace with colour gradient low→high freq
        n = len(gamma)
        for i in range(n - 1):
            t   = i / max(n - 2, 1)
            col = (t * 0.9, 0.25, 1.0 - t * 0.9, 0.85)
            ax.plot(ang[i:i+2], mag[i:i+2], color=col, linewidth=1.8)

        # Key markers
        res_idx = int(np.argmin(result["s11_db"]))
        ax.plot(ang[0],       mag[0],       "o", color=C["green"],
                markersize=7, zorder=6, label=f"Start {_fmt_freq(result['frequencies'][0])}")
        ax.plot(ang[res_idx], mag[res_idx], "*", color=C["yellow"],
                markersize=10, zorder=7,
                label=f"Res {_fmt_freq(result['frequencies'][res_idx])}")
        ax.plot(ang[-1],      mag[-1],      "s", color=C["red"],
                markersize=6, zorder=6,
                label=f"Stop {_fmt_freq(result['frequencies'][-1])}")

        ax.legend(fontsize=6, facecolor=C["grid"], labelcolor=C["text"],
                  edgecolor=C["border"], loc="lower left",
                  bbox_to_anchor=(-0.12, -0.05))

    # ── Marker system ─────────────────────────────────────────────────────────

    def place_marker(self, x_data: float, y_data: float = None):
        """
        Add a numbered marker at the clicked frequency (Cartesian) or
        nearest Γ point (Polar).  Returns a dict with measurement values.
        """
        if self._result is None:
            return None
        if len(self._markers) >= MAX_MARKERS:
            # Evict oldest
            self._markers.pop(0)

        result = self._result
        freqs  = result["frequencies"]

        if self._is_polar:
            gamma = result["gamma"]
            # Find nearest point in polar Γ-plane
            dists = (gamma.real - np.cos(x_data)*y_data)**2 + \
                    (gamma.imag - np.sin(x_data)*y_data)**2 \
                    if y_data is not None else \
                    (np.abs(gamma) - y_data)**2
            # x_data=theta, y_data=r in polar axes
            dists = (np.angle(gamma) - x_data)**2 + \
                    (np.abs(gamma)   - (y_data or 0))**2
            idx = int(np.argmin(dists))
        else:
            # Log-domain nearest for log sweep
            log_f = np.log(np.maximum(freqs, 1e-30))
            log_x = np.log(max(x_data, 1e-30))
            idx   = int(np.argmin(np.abs(log_f - log_x)))

        num   = len(self._markers) + 1
        color = MARKER_COLORS[(num - 1) % len(MARKER_COLORS)]

        marker_data = self._build_marker_data(result, idx)
        self._markers.append(dict(idx=idx, num=num, color=color,
                                  data=marker_data))
        self._redraw()
        return marker_data

    def _build_marker_data(self, result, idx):
        f    = result["frequencies"][idx]
        Z_L  = result["Z_L"][idx]
        Z0   = result["Z0"]
        return dict(
            idx   = idx,
            f     = f,
            s11   = result["s11_db"][idx],
            vswr  = result["vswr"][idx],
            gmag  = float(np.abs(result["gamma"][idx])),
            phase = result["phase_deg"][idx],
            gd_ns = result["group_delay_ns"][idx],
            z_re  = Z_L.real,
            z_im  = Z_L.imag,
            Z_L   = Z_L,
            Z0    = Z0,
        )

    def _redraw_markers_cartesian(self, result, mode, freqs, y):
        ax = self._ax_cart
        if ax is None:
            return
        for mk in self._markers:
            idx   = mk["idx"]
            color = mk["color"]
            num   = mk["num"]
            f     = freqs[idx]

            # Y value for this mode
            if mode == "Log Mag (dB)":        yv = result["s11_db"][idx]
            elif mode == "Linear Mag":         yv = float(np.abs(result["gamma"][idx]))
            elif mode == "VSWR":               yv = float(np.clip(result["vswr"][idx], 1, 100))
            elif mode == "Phase (deg)":        yv = result["phase_deg"][idx]
            elif mode == "Unwrapped Phase":    yv = result["phase_unwrapped"][idx]
            elif mode == "Group Delay (ns)":   yv = result["group_delay_ns"][idx]
            elif mode == "Real Z (Ω)":         yv = result["z_real"][idx]
            elif mode == "Imaginary Z (Ω)":    yv = result["z_imag"][idx]
            else:                              yv = y[idx]

            # Clamp to visible range
            ylim = Y_LIMITS.get(mode)
            if ylim:
                yv = float(np.clip(yv, ylim[0], ylim[1]))

            ax.axvline(f, color=color, linestyle="-.", linewidth=0.9,
                       alpha=0.8, zorder=6)
            ax.plot(f, yv, "D", color=color, markersize=7, zorder=8)

            # Numbered badge
            ax.annotate(
                str(num),
                xy=(f, yv), xytext=(6, 6),
                textcoords="offset points",
                fontsize=7, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"],
                          edgecolor=color, alpha=0.9),
                zorder=9,
            )

    def get_marker_data(self):
        """Return list of all current marker measurement dicts, with marker number."""
        if self._result is None:
            return []
        return [dict(**mk["data"], num=mk["num"]) for mk in self._markers]