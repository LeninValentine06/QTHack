"""
smith_chart.py  —  Smith chart canvas with numbered markers.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False

C = dict(
    bg      = "#1e1e2e",
    axes_bg = "#181825",
    grid    = "#2a2a3c",
    border  = "#45475a",
    text    = "#cdd6f4",
    subtext = "#a6adc8",
    blue    = "#89b4fa",
    red     = "#f38ba8",
    green   = "#a6e3a1",
    yellow  = "#f9e2af",
    mauve   = "#cba6f7",
    peach   = "#fab387",
    teal    = "#94e2d5",
    lavender= "#b4befe",
    sky     = "#89dceb",
    sapphire= "#74c7ec",
)

MARKER_COLORS = [C["mauve"], C["peach"], C["green"], C["teal"],
                 C["sapphire"], C["lavender"], C["sky"], C["yellow"]]


def _fmt_freq(f: float) -> str:
    if f >= 1e9:  return f"{f/1e9:.4f} GHz"
    if f >= 1e6:  return f"{f/1e6:.4f} MHz"
    if f >= 1e3:  return f"{f/1e3:.4f} kHz"
    return f"{f:.2f} Hz"


class SmithCanvas(FigureCanvas):

    def __init__(self, parent=None, dpi=95):
        self.fig = Figure(figsize=(5, 4.5), dpi=dpi, facecolor=C["bg"])
        self.ax  = self.fig.add_subplot(111)
        self._result  = None
        self._markers = []   # list of {idx, num, color}
        self._draw_empty()
        super().__init__(self.fig)
        self.setParent(parent)

    # ── Smith grid ────────────────────────────────────────────────────────────

    def _draw_empty(self):
        ax = self.ax
        ax.set_facecolor(C["axes_bg"])
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.tick_params(colors=C["text"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.set_title("Smith Chart  (Γ plane)", color=C["text"], fontsize=9)
        ax.set_xlabel("Re(Γ)", color=C["subtext"], fontsize=8)
        ax.set_ylabel("Im(Γ)", color=C["subtext"], fontsize=8)
        self._draw_smith_grid(ax)
        self.fig.tight_layout(pad=1.5)

    @staticmethod
    def _draw_smith_grid(ax):
        theta = np.linspace(0, 2*np.pi, 500)
        ax.plot(np.cos(theta), np.sin(theta), color=C["border"], linewidth=1.4)

        for r in [0, 0.2, 0.5, 1.0, 2.0, 5.0]:
            cx  = r / (r + 1.0)
            rad = 1.0 / (r + 1.0)
            x   = cx + rad * np.cos(theta)
            y   =      rad * np.sin(theta)
            mask = x**2 + y**2 <= 1.002
            ax.plot(np.where(mask, x, np.nan),
                    np.where(mask, y, np.nan),
                    color=C["grid"], linewidth=0.7)
            if r > 0:
                ax.text(cx + rad + 0.02, 0.02, f"{r}",
                        fontsize=5, color=C["border"], ha="left")

        t = np.linspace(0, 2*np.pi, 700)
        for xv in [0.2, 0.5, 1.0, 2.0, 5.0]:
            for sign in [1, -1]:
                cx, cy = 1.0, sign / xv
                rad    = abs(1.0 / xv)
                pts    = np.column_stack([cx + rad*np.cos(t), cy + rad*np.sin(t)])
                inside = pts[:,0]**2 + pts[:,1]**2 <= 1.002
                ax.plot(np.where(inside, pts[:,0], np.nan),
                        np.where(inside, pts[:,1], np.nan),
                        color=C["grid"], linewidth=0.7)
                if inside.any():
                    lbl_i = np.where(inside)[0][len(np.where(inside)[0])//2]
                    ax.text(pts[lbl_i,0], pts[lbl_i,1],
                            f"{sign*xv:+.1f}j",
                            fontsize=4.5, color=C["border"],
                            ha="center", va="center")

        ax.axhline(0, color=C["grid"], linewidth=0.7)
        ax.plot(0, 0, "+", color=C["border"], markersize=8)

    # ── Plot ─────────────────────────────────────────────────────────────────

    def plot_result(self, result: dict):
        self._result = result
        self._markers.clear()
        self._redraw()

    def clear_markers(self):
        self._markers.clear()
        if self._result:
            self._redraw()

    def _redraw(self):
        result = self._result
        self.ax.cla()
        self._draw_empty()

        freqs = result["frequencies"]
        gamma = result["gamma"]
        re, im = gamma.real, gamma.imag
        n = len(re)

        # Trace
        if HAS_SKRF:
            self._plot_via_skrf(freqs, gamma)
        else:
            self._plot_gradient_trace(re, im, n)

        # Key points
        res_idx = int(np.argmin(result["s11_db"]))
        self.ax.plot(re[0], im[0], "o", color=C["green"], markersize=7,
                     zorder=6, label=f"Start {_fmt_freq(freqs[0])}")
        self.ax.plot(re[res_idx], im[res_idx], "*", color=C["yellow"],
                     markersize=10, zorder=7,
                     label=f"Res {_fmt_freq(freqs[res_idx])}")
        self.ax.plot(re[-1], im[-1], "s", color=C["red"], markersize=6,
                     zorder=6, label=f"Stop {_fmt_freq(freqs[-1])}")

        # Numbered markers
        for mk in self._markers:
            idx, num, col = mk["idx"], mk["num"], mk["color"]
            self.ax.plot(re[idx], im[idx], "D", color=col,
                         markersize=8, zorder=9)
            self.ax.annotate(
                str(num), xy=(re[idx], im[idx]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"],
                          edgecolor=col, alpha=0.9),
                zorder=10,
            )

        self.ax.legend(fontsize=6, facecolor=C["grid"], labelcolor=C["text"],
                       edgecolor=C["border"], loc="lower left")
        self.fig.tight_layout(pad=1.5)
        self.draw()

    def _plot_via_skrf(self, freqs, gamma):
        n        = len(freqs)
        s_array  = gamma.reshape(n, 1, 1)
        freq_obj = skrf.Frequency.from_f(freqs, unit="hz")
        net      = skrf.Network(frequency=freq_obj, s=s_array)
        try:
            net.plot_s_smith(ax=self.ax, show_legend=False,
                             draw_labels=False,
                             color=C["blue"], linewidth=1.8)
        except Exception:
            self._plot_gradient_trace(gamma.real, gamma.imag, n)

    def _plot_gradient_trace(self, re, im, n):
        # Fix #7: single LineCollection instead of O(N) ax.plot calls
        pts  = np.column_stack([re, im]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t    = np.linspace(0, 1, max(n - 1, 1))
        cols = [(v * 0.9, 0.25, 1.0 - v * 0.9, 0.85) for v in t]
        lc   = LineCollection(segs, colors=cols, linewidth=1.8)
        self.ax.add_collection(lc)

    # ── Marker placement ──────────────────────────────────────────────────────

    def place_marker(self, x: float, y: float):
        if self._result is None:
            return None
        if len(self._markers) >= 8:
            self._markers.pop(0)

        gamma  = self._result["gamma"]
        freqs  = self._result["frequencies"]
        dists  = (gamma.real - x)**2 + (gamma.imag - y)**2
        idx    = int(np.argmin(dists))
        num    = len(self._markers) + 1
        color  = MARKER_COLORS[(num-1) % len(MARKER_COLORS)]

        self._markers.append(dict(idx=idx, num=num, color=color))
        self._redraw()

        Z_L = self._result["Z_L"][idx]
        Z0  = self._result["Z0"]
        return dict(
            f    = freqs[idx],
            Z_L  = Z_L,
            gamma= gamma[idx],
            s11  = self._result["s11_db"][idx],
            vswr = self._result["vswr"][idx],
            gmag = float(np.abs(gamma[idx])),
        )