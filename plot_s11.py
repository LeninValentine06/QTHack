"""
plot_s11.py  —  Multi-mode VNA plot canvas, skrf-first.

All RF quantities (S11 dB, VSWR, phase, group delay, return loss, Z)
are derived from a skrf.Network object wherever skrf is available.
Raw numpy fallback is kept for the skrf-absent case.

Supported CH2 modes
-------------------
  Log Mag (dB)       — skrf net.s_db[:, 0, 0]
  Linear Mag         — skrf net.s_mag[:, 0, 0]  (= |Γ|)
  VSWR               — skrf net.s_vswr[:, 0, 0]
  Phase (deg)        — skrf net.s_deg[:, 0, 0]
  Unwrapped Phase    — np.unwrap on skrf phase
  Group Delay (ns)   — skrf net.group_delay (via s-param derivative)
  Real Z (Ω)         — skrf net.z[:, 0, 0].real
  Imaginary Z (Ω)    — skrf net.z[:, 0, 0].imag
  Polar              — matplotlib polar axes, |Γ| vs ∠Γ
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False

# ── Colour palette (Github-dark instrument style) ─────────────────────────────
C = dict(
    bg      = "#161b22",
    axes_bg = "#0d1117",
    grid    = "#21262d",
    border  = "#30363d",
    text    = "#e6edf3",
    subtext = "#8b949e",
    blue    = "#58a6ff",
    green   = "#3fb950",
    yellow  = "#d29922",
    red     = "#f85149",
    peach   = "#d29922",
    teal    = "#3fb950",
    mauve   = "#d29922",
    lavender= "#8b949e",
    sky     = "#58a6ff",
    sapphire= "#58a6ff",
)

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

Y_LIMITS = {
    "Log Mag (dB)":    None,     # adaptive
    "Linear Mag":      (0.0,  1.0),
    "VSWR":            (1.0, 50.0),
    "Phase (deg)":     (-180.0, 180.0),
    "Unwrapped Phase": None,
    "Group Delay (ns)":None,
    "Real Z (Ω)":      None,
    "Imaginary Z (Ω)": None,
    "Polar":           None,
}

REF_LINES = {
    "Log Mag (dB)": [
        (-6,  "#484f58", "−6 dB"),
        (-10, "#d29922", "−10 dB"),
        (-20, "#3fb950", "−20 dB"),
    ],
    "Linear Mag": [
        (0.316, "#d29922", "0.316 (−10 dB)"),
        (0.100, "#3fb950", "0.100 (−20 dB)"),
    ],
    "VSWR": [
        (2.0, "#3fb950", "VSWR = 2"),
        (3.0, "#d29922", "VSWR = 3"),
    ],
    "Phase (deg)":      [],
    "Unwrapped Phase":  [],
    "Group Delay (ns)": [],
    "Real Z (Ω)":       [],
    "Imaginary Z (Ω)":  [],
    "Polar":            [],
}

MARKER_COLORS = ["#d29922", "#58a6ff", "#3fb950", "#8b949e",
                 "#d29922", "#58a6ff", "#3fb950", "#8b949e"]
MAX_MARKERS = 8

_EPS = 1e-12


# =============================================================================
# Helpers
# =============================================================================

def _fmt_freq(f: float) -> str:
    if f >= 1e9:  return f"{f/1e9:.4f} GHz"
    if f >= 1e6:  return f"{f/1e6:.4f} MHz"
    if f >= 1e3:  return f"{f/1e3:.4f} kHz"
    return f"{f:.2f} Hz"


def _freq_fmt_tick(x, _pos):
    if x <= 0:    return ""
    if x >= 1e9:  return f"{x/1e9:.3g} GHz"
    if x >= 1e6:  return f"{x/1e6:.3g} MHz"
    if x >= 1e3:  return f"{x/1e3:.3g} kHz"
    return f"{x:.3g} Hz"


def _configure_freq_axis(ax, freqs, sweep):
    fmt = ticker.FuncFormatter(_freq_fmt_tick)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    if sweep == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
        ax.xaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=[2,3,4,5,6,7,8,9], numticks=50))
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune="both"))
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)


def _make_net(result: dict) -> "skrf.Network | None":
    """
    Build a skrf.Network from the simulation result dict.
    Returns None if skrf is unavailable.
    """
    if not HAS_SKRF:
        return None
    gamma  = result["gamma"]
    freqs  = result["frequencies"]
    Z0     = result.get("Z0", 50.0)
    s      = gamma.reshape(len(freqs), 1, 1)
    return skrf.Network(
        frequency=skrf.Frequency.from_f(freqs, unit="hz"),
        s=s,
        z0=np.full((len(freqs), 1), Z0),
    )


# =============================================================================
# VNACanvas
# =============================================================================

class VNACanvas(FigureCanvas):
    """
    Multi-mode VNA plot canvas.

    skrf.Network is used as the source of truth for every RF quantity:
      • s_db, s_mag, s_vswr, s_deg  — via skrf properties
      • group_delay                  — via skrf.Network.group_delay
      • z (input impedance)          — via skrf.Network.z

    Raw numpy fallback runs when skrf is absent.
    """

    def __init__(self, parent=None, dpi=95):
        self.fig      = Figure(figsize=(7, 4), dpi=dpi, facecolor=C["bg"])
        self._ax_cart = self.fig.add_subplot(111)
        self._ax_polar = None
        self._result  = None
        self._net     = None        # skrf.Network — refreshed with each result
        self._mode    = "Log Mag (dB)"
        self._markers = []
        self._is_polar = False

        self._style_ax(self._ax_cart)
        super().__init__(self.fig)
        self.setParent(parent)

    # ── Axes style ────────────────────────────────────────────────────────────

    def _style_ax(self, ax, polar=False):
        ax.set_facecolor(C["axes_bg"])
        ax.tick_params(colors=C["text"], labelsize=7)
        if not polar:
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
            ax.grid(True, which="both", ls="--", lw=0.4, color=C["grid"])
        else:
            ax.grid(True, ls="--", lw=0.4, color=C["grid"])
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
        self._net    = _make_net(result)
        self._markers.clear()
        self._redraw()

    def clear_markers(self):
        self._markers.clear()
        if self._result is not None:
            self._redraw()

    # ── Data extraction (skrf-first) ──────────────────────────────────────────

    def _get_y(self, result: dict, mode: str):
        """
        Return (y_array, ylabel, trace_color).

        Uses skrf.Network properties wherever available so the quantities
        are computed consistently with skrf's own conventions.
        """
        net = self._net

        if mode == "Log Mag (dB)":
            if net is not None:
                y = net.s_db[:, 0, 0]
            else:
                mag = np.clip(np.abs(result["gamma"]), _EPS, None)
                y   = 20*np.log10(mag)
            return y, "S11 (dB)", C["blue"]

        if mode == "Linear Mag":
            if net is not None:
                y = net.s_mag[:, 0, 0]
            else:
                y = np.abs(result["gamma"])
            return y, "|Γ|", C["blue"]

        if mode == "VSWR":
            if net is not None:
                # skrf.Network.s_vswr is available on 1-ports
                try:
                    y = net.s_vswr[:, 0, 0]
                except AttributeError:
                    gm = np.clip(np.abs(result["gamma"]), 0, 1-_EPS)
                    y  = (1+gm)/(1-gm)
            else:
                gm = np.clip(np.abs(result["gamma"]), 0, 1-_EPS)
                y  = (1+gm)/(1-gm)
            return np.clip(y, 1.0, 50.0), "VSWR", C["yellow"]

        if mode == "Phase (deg)":
            if net is not None:
                y = net.s_deg[:, 0, 0]
            else:
                y = np.angle(result["gamma"], deg=True)
            return y, "Phase (°)", C["green"]

        if mode == "Unwrapped Phase":
            if net is not None:
                y = np.degrees(np.unwrap(np.angle(net.s[:, 0, 0])))
            else:
                y = np.degrees(np.unwrap(np.angle(result["gamma"])))
            return y, "Unwrapped Phase (°)", C["green"]

        if mode == "Group Delay (ns)":
            if net is not None:
                try:
                    # skrf.Network.group_delay: shape (N, nports, nports), seconds
                    gd_s = net.group_delay[:, 0, 0]
                    y    = np.array(gd_s, dtype=float) * 1e9
                except Exception:
                    y = self._numpy_group_delay(result)
            else:
                y = self._numpy_group_delay(result)
            # ── Hard NaN/Inf sanitisation BEFORE any percentile or plot call ──
            # np.gradient and skrf.group_delay both produce ±Inf at edge points
            # when the frequency spacing is uneven (log sweeps).  Passing Inf
            # to np.percentile causes a RuntimeWarning that can propagate to a
            # C-level heap corruption on Windows (exit code 0xC0000409).
            y = np.where(np.isfinite(y), y, np.nan)   # ±Inf → NaN first
            finite = y[np.isfinite(y)]
            if len(finite) < 4:
                # Degenerate case: almost no usable data — return flat zeros
                y = np.zeros_like(y)
                return y, "Group Delay (ns)", C["teal"]
            # IQR clip on finite values only, then fill NaNs with median
            q25, q75 = float(np.nanpercentile(y, 25)), float(np.nanpercentile(y, 75))
            iqr   = max(q75 - q25, 1.0)
            lo    = q25 - 5.0 * iqr
            hi    = q75 + 5.0 * iqr
            y     = np.clip(y, lo, hi)                 # NaNs survive clip
            med   = float(np.nanmedian(y))
            y     = np.where(np.isfinite(y), y, med)   # fill residual NaNs
            return y, "Group Delay (ns)", C["teal"]

        if mode == "Real Z (Ω)":
            if net is not None:
                # skrf.Network.z: complex impedance array shape (N,1,1)
                y = net.z[:, 0, 0].real
            else:
                y = result.get("z_real", result["Z_L"].real)
            return y, "Re(Z) (Ω)", C["peach"]

        if mode == "Imaginary Z (Ω)":
            if net is not None:
                y = net.z[:, 0, 0].imag
            else:
                y = result.get("z_imag", result["Z_L"].imag)
            return y, "Im(Z) (Ω)", C["mauve"]

        # Fallback
        return np.abs(result["gamma"]), "|Γ|", C["blue"]

    @staticmethod
    def _numpy_group_delay(result):
        """
        Compute group delay in nanoseconds via finite difference on unwrapped phase.

        GD(f) = −dφ/dω   [seconds] = −dφ/(2π·df)

        On log sweeps the frequency spacing is non-uniform, so np.gradient
        must receive the ω array explicitly (not just the phase array).
        The edge points from np.gradient are one-sided differences and can
        be extremely large on log sweeps — they are replaced by the nearest
        interior value to prevent Inf from propagating to matplotlib.
        """
        phase_rad = np.unwrap(np.angle(result["gamma"]))
        freqs     = np.asarray(result["frequencies"], dtype=float)
        omega     = 2.0 * np.pi * freqs

        # np.gradient handles non-uniform spacing when the coordinate array
        # is supplied as the second argument.
        gd_s = -np.gradient(phase_rad, omega)
        gd_ns = gd_s * 1e9

        # Replace any ±Inf produced at the edges (one-sided differences on
        # log sweeps can be huge) with the nearest finite neighbour.
        if not np.isfinite(gd_ns[0]):
            for i in range(1, len(gd_ns)):
                if np.isfinite(gd_ns[i]):
                    gd_ns[0] = gd_ns[i]
                    break
        if not np.isfinite(gd_ns[-1]):
            for i in range(len(gd_ns)-2, -1, -1):
                if np.isfinite(gd_ns[i]):
                    gd_ns[-1] = gd_ns[i]
                    break

        return gd_ns

    # ── Redraw dispatcher ─────────────────────────────────────────────────────

    def _redraw(self):
        try:
            self._redraw_inner()
        except Exception as exc:
            # Rendering failure must never crash the Qt process.
            # Show a minimal error message on the axes instead.
            try:
                ax = self._ax_cart
                ax.cla()
                ax.set_facecolor(C["axes_bg"])
                ax.text(0.5, 0.5,
                        f"Render error ({self._mode}):\n{exc}",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        color="#f85149", fontsize=8, wrap=True)
                self.draw()
            except Exception:
                pass  # if even the fallback fails, stay silent

    def _redraw_inner(self):
        result = self._result
        mode   = self._mode
        freqs  = result["frequencies"]
        sweep  = result.get("sweep_type", "log")

        if mode == "Polar":
            self._switch_to_polar()
            self._draw_polar(result)
        else:
            self._switch_to_cart()
            self._draw_cartesian(result, mode, freqs, sweep)

        self.fig.tight_layout(pad=1.2)
        self.draw()

    def _switch_to_cart(self):
        if self._is_polar and self._ax_polar is not None:
            self._ax_polar.remove()
            self._ax_polar = None
        self._is_polar = False
        self._ax_cart.set_visible(True)
        self._ax_cart.cla()
        self._style_ax(self._ax_cart)

    def _switch_to_polar(self):
        self._ax_cart.set_visible(False)
        if self._ax_polar is None:
            self._ax_polar = self.fig.add_subplot(111, projection="polar")
        self._ax_polar.cla()
        self._style_ax(self._ax_polar, polar=True)
        self._is_polar = True

    # ── Cartesian plot ────────────────────────────────────────────────────────

    def _draw_cartesian(self, result, mode, freqs, sweep):
        ax = self._ax_cart
        y, ylabel, ycolor = self._get_y(result, mode)

        _configure_freq_axis(ax, freqs, sweep)
        ax.set_xlabel("Frequency (log)" if sweep == "log" else "Frequency",
                      color=C["subtext"], fontsize=8)
        ax.set_ylabel(ylabel, color=C["subtext"], fontsize=8)
        ax.set_title(f"{ylabel} vs Frequency", color=C["text"], fontsize=9)

        ax.plot(freqs, y, color=ycolor, lw=1.8, zorder=3)

        # Y limits
        ylim = Y_LIMITS.get(mode)
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif mode == "Log Mag (dB)":
            yc    = y[np.isfinite(y)]
            floor = max(float(yc.min()) - 5.0, -80.0) if len(yc) else -40.0
            ax.set_ylim(floor, 2.0)
        else:
            yfinite = y[np.isfinite(y)]
            if len(yfinite) >= 2:
                ymin = float(yfinite.min())
                ymax = float(yfinite.max())
            else:
                ymin, ymax = -1.0, 1.0
            span = ymax - ymin
            pad  = max(span * 0.12, abs(ymax) * 0.1, 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)

        # Reference lines
        y_lo, y_hi = ax.get_ylim()
        for ref, col, lbl in REF_LINES.get(mode, []):
            if y_lo <= ref <= y_hi:
                ax.axhline(ref, color=col, ls="--", lw=0.7,
                           alpha=0.65, label=lbl, zorder=2)

        # Resonance vline + BW shading
        bw = result["bandwidth"]
        rc = result.get("resonance_check", {})
        if bw["valid"] and not rc.get("outside_sweep", False):
            ax.axvline(bw["f_res"], color=C["red"], ls=":",
                       lw=1.0, alpha=0.75, zorder=2,
                       label=f"Res {_fmt_freq(bw['f_res'])}")
        if mode == "Log Mag (dB)" and bw["valid"]:
            ax.axvspan(bw["f_low"], bw["f_high"],
                       alpha=0.10, color=C["blue"], zorder=1)
            ax.axvline(bw["f_low"],  color=C["blue"], ls="--", lw=0.8, alpha=0.45)
            ax.axvline(bw["f_high"], color=C["blue"], ls="--", lw=0.8, alpha=0.45)

        ax.legend(fontsize=6.5, facecolor=C["grid"], labelcolor=C["text"],
                  edgecolor=C["border"], loc="upper right")

        # Markers
        self._redraw_markers_cartesian(result, mode, freqs, y)

    # ── Polar plot ────────────────────────────────────────────────────────────

    def _draw_polar(self, result):
        ax    = self._ax_polar
        gamma = result["gamma"]
        mag   = np.abs(gamma)
        ang   = np.angle(gamma)

        ax.set_title("Polar  (Γ plane)", color=C["text"], pad=12)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_tick_params(labelsize=6, colors=C["subtext"])
        ax.xaxis.set_tick_params(labelsize=7, colors=C["text"])

        # Unit circle
        th = np.linspace(0, 2*np.pi, 400)
        ax.plot(th, np.ones(400), color=C["border"], lw=0.8, ls="--")

        # Gradient trace
        n = len(gamma)
        for i in range(n-1):
            t   = i / max(n-2, 1)
            col = (t*0.9, 0.25, 1.0-t*0.9, 0.85)
            ax.plot(ang[i:i+2], mag[i:i+2], color=col, lw=1.8)

        res_idx = int(np.argmin(result["s11_db"]))
        freqs   = result["frequencies"]
        ax.plot(ang[0],       mag[0],       "o", color=C["green"],
                ms=7,  zorder=6, label=f"Start {_fmt_freq(freqs[0])}")
        ax.plot(ang[res_idx], mag[res_idx], "*", color=C["yellow"],
                ms=11, zorder=7, label=f"Res {_fmt_freq(freqs[res_idx])}")
        ax.plot(ang[-1],      mag[-1],      "s", color=C["red"],
                ms=6,  zorder=6, label=f"Stop {_fmt_freq(freqs[-1])}")

        ax.legend(fontsize=6, facecolor=C["grid"], labelcolor=C["text"],
                  edgecolor=C["border"], loc="lower left",
                  bbox_to_anchor=(-0.12, -0.05))

    # ── Marker system ─────────────────────────────────────────────────────────

    def place_marker(self, x_data: float, y_data: float = None):
        if self._result is None:
            return None
        if len(self._markers) >= MAX_MARKERS:
            self._markers.pop(0)

        result = self._result
        freqs  = result["frequencies"]

        if self._is_polar:
            gamma = result["gamma"]
            adiff = np.arctan2(
                np.sin(np.angle(gamma) - x_data),
                np.cos(np.angle(gamma) - x_data))
            dists = adiff**2 + (np.abs(gamma) - (y_data or 0))**2
            idx   = int(np.argmin(dists))
        else:
            log_f = np.log(np.maximum(freqs, 1e-30))
            idx   = int(np.argmin(np.abs(log_f - np.log(max(x_data, 1e-30)))))

        num   = len(self._markers) + 1
        color = MARKER_COLORS[(num-1) % len(MARKER_COLORS)]
        data  = self._build_marker_data(result, idx)
        self._markers.append(dict(idx=idx, num=num, color=color, data=data))
        self._redraw()
        return data

    def _build_marker_data(self, result, idx):
        """
        Build the marker readout dict, using skrf.Network for Z if available.
        """
        freqs = result["frequencies"]
        gamma = result["gamma"]
        net   = self._net

        if net is not None:
            Z_val = complex(net.z[idx, 0, 0])
        else:
            Z_val = result["Z_L"][idx]

        gd_raw = float(result["group_delay_ns"][idx])
        gd_ns  = gd_raw if np.isfinite(gd_raw) else float(np.nanmedian(result["group_delay_ns"]))
        return dict(
            idx   = idx,
            f     = freqs[idx],
            s11   = float(result["s11_db"][idx]),
            vswr  = float(result["vswr"][idx]),
            gmag  = float(np.abs(gamma[idx])),
            phase = float(result["phase_deg"][idx]),
            gd_ns = gd_ns,
            z_re  = Z_val.real,
            z_im  = Z_val.imag,
            Z_L   = Z_val,
            Z0    = result.get("Z0", 50.0),
        )

    def _redraw_markers_cartesian(self, result, mode, freqs, y):
        ax = self._ax_cart
        net = self._net
        for mk in self._markers:
            idx   = mk["idx"]
            color = mk["color"]
            num   = mk["num"]
            f     = freqs[idx]

            # Pick y value for this mode using skrf where possible
            if mode == "Log Mag (dB)":
                yv = float(net.s_db[idx, 0, 0]) if net else float(result["s11_db"][idx])
            elif mode == "Linear Mag":
                yv = float(net.s_mag[idx, 0, 0]) if net else float(np.abs(result["gamma"][idx]))
            elif mode == "VSWR":
                try:
                    yv = float(net.s_vswr[idx, 0, 0]) if net else float(result["vswr"][idx])
                except Exception:
                    yv = float(result["vswr"][idx])
            elif mode == "Phase (deg)":
                yv = float(net.s_deg[idx, 0, 0]) if net else float(result["phase_deg"][idx])
            elif mode == "Unwrapped Phase":
                yv = float(result["phase_unwrapped"][idx])
            elif mode == "Group Delay (ns)":
                raw = float(result["group_delay_ns"][idx])
                # Sanitise: a marker on a ±Inf point would crash ax.annotate
                yv = raw if np.isfinite(raw) else float(np.nanmedian(result["group_delay_ns"]))
            elif mode == "Real Z (Ω)":
                yv = float(net.z[idx, 0, 0].real) if net else float(result["z_real"][idx])
            elif mode == "Imaginary Z (Ω)":
                yv = float(net.z[idx, 0, 0].imag) if net else float(result["z_imag"][idx])
            else:
                yv = float(y[idx])

            y_lo, y_hi = ax.get_ylim()
            yv = float(np.clip(yv, y_lo, y_hi))

            ax.axvline(f, color=color, ls="-.", lw=0.9, alpha=0.8, zorder=6)
            ax.plot(f, yv, "D", color=color, ms=7, zorder=8)
            ax.annotate(
                str(num), xy=(f, yv), xytext=(6, 6),
                textcoords="offset points",
                fontsize=7, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"],
                          edgecolor=color, alpha=0.9),
                zorder=9,
            )

    def get_marker_data(self):
        if self._result is None:
            return []
        return [dict(**mk["data"], num=mk["num"]) for mk in self._markers]