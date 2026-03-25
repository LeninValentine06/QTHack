"""
smith_chart.py  —  Professional Smith chart canvas, skrf-first rendering.

Rendering pipeline (mirrors the skrf "Advanced Smith Chart with background"
docs pattern exactly):

    background = plt.imread('smithchart.png')          # Step 1
    ax.imshow(background, extent=[...])                 # Step 2
    skrf.plotting.smith(ax, draw_labels=True,           # Step 3
                        chart_type='z')
    net.plot_s_smith(ax=ax)                             # Step 4

All grid geometry, circle labelling, outer scale annotations
(WAVELENGTHS TOWARD GENERATOR / LOAD, reactance / resistance component
banners) are reproduced from "The Complete Smith Chart" (Black Magic
Design) — the same layout shown in SmithChart.pdf.

Fallback: if scikit-rf is absent the manual matplotlib grid is used.
"""

import io
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # off-screen for background generation only
import matplotlib.pyplot as _plt_bg
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Arc

try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False

# ── Colour palette (Catppuccin Mocha — dark instrument theme) ─────────────────
C = dict(
    bg       = "#1e1e2e",
    axes_bg  = "#11111b",
    grid     = "#313244",
    border   = "#45475a",
    text     = "#cdd6f4",
    subtext  = "#a6adc8",
    overlay  = "#6c7086",
    blue     = "#89b4fa",
    red      = "#f38ba8",
    green    = "#a6e3a1",
    yellow   = "#f9e2af",
    mauve    = "#cba6f7",
    peach    = "#fab387",
    teal     = "#94e2d5",
    lavender = "#b4befe",
    sky      = "#89dceb",
    sapphire = "#74c7ec",
    maroon   = "#eba0ac",
)

MARKER_COLORS = [C["mauve"], C["peach"], C["green"], C["teal"],
                 C["sapphire"], C["lavender"], C["sky"], C["yellow"]]

# ── Γ-plane extent — aligns imshow pixels with skrf coordinate system ─────────
_EXTENT = [-1.185, 1.14, -1.13, 1.155]

# ── Outer annotation ring radii (just outside unit circle) ────────────────────
_R_WTG   = 1.09   # Wavelengths Toward Generator ring
_R_WTL   = 1.16   # Wavelengths Toward Load ring
_R_ANGLE = 1.22   # Angle of reflection coefficient ring
_R_LABEL = 1.30   # Outer label radius


# =============================================================================
# Background PNG — rendered once at import, used as Layer 1
# =============================================================================

def _generate_background_png(size_px: int = 800) -> np.ndarray:
    """
    Render a high-resolution Smith chart background to an RGBA array.

    Layers (bottom to top):
      • deep navy fill inside unit disk
      • faint r=1 circle highlight (matched impedance)
      • resistance circles (r = const)
      • reactance arcs   (x = const)
      • VSWR dashed rings at 1.5, 2, 3
      • real axis
      • outer scale arcs: Wavelengths Toward Generator / Load
      • outer angle annotations (degrees)
      • component banner text (PDF-style)

    Returns float32 RGBA array shape (H, W, 4).
    """
    dpi  = 100
    inch = size_px / dpi
    fig  = _plt_bg.figure(figsize=(inch, inch), dpi=dpi)
    ax   = fig.add_axes([0, 0, 1, 1])

    bg = "#0d0d1a"
    ax.set_facecolor(bg);  fig.patch.set_facecolor(bg)
    L, R, B, T = _EXTENT
    ax.set_xlim(L, R);  ax.set_ylim(B, T)
    ax.set_aspect("equal");  ax.axis("off")

    theta = np.linspace(0, 2*np.pi, 1000)
    t     = np.linspace(0, 2*np.pi, 1600)

    # ── Background fill inside unit disk ─────────────────────────────────────
    ax.fill(np.cos(theta), np.sin(theta), color="#13132a", zorder=0)

    # ── r=1 circle soft fill ──────────────────────────────────────────────────
    ax.add_patch(mpatches.Circle((0.5, 0), 0.5,
                                 fc="#18183a", ec="none", zorder=1))

    # ── Resistance circles ────────────────────────────────────────────────────
    for r in [0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        cx  = r / (r + 1.0)
        rad = 1.0 / (r + 1.0)
        x   = cx + rad * np.cos(theta)
        y   =      rad * np.sin(theta)
        mask = x**2 + y**2 <= 1.004
        lw  = 1.1 if r in (0, 1.0) else 0.45
        col = "#4a4a82" if r in (0, 1.0) else "#252550"
        ax.plot(np.where(mask, x, np.nan),
                np.where(mask, y, np.nan), color=col, lw=lw, zorder=2)

    # ── Reactance arcs ────────────────────────────────────────────────────────
    for xv in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        for sign in (1, -1):
            cx2, cy2 = 1.0, sign/xv
            rad2     = abs(1.0/xv)
            pts      = np.column_stack(
                [cx2 + rad2*np.cos(t), cy2 + rad2*np.sin(t)])
            inside = pts[:,0]**2 + pts[:,1]**2 <= 1.004
            lw  = 1.1 if xv == 1.0 else 0.45
            col = "#4a4a82" if xv == 1.0 else "#252550"
            ax.plot(np.where(inside, pts[:,0], np.nan),
                    np.where(inside, pts[:,1], np.nan), color=col, lw=lw, zorder=2)

    # ── Real axis ─────────────────────────────────────────────────────────────
    ax.axhline(0, color="#252550", lw=0.45, zorder=2)
    ax.plot(0, 0, "+", color="#38386a", ms=10, lw=0.7, zorder=3)

    # ── VSWR dashed rings ─────────────────────────────────────────────────────
    for vswr in [1.5, 2.0, 3.0, 5.0]:
        gm = (vswr-1)/(vswr+1)
        ax.plot(gm*np.cos(theta), gm*np.sin(theta),
                color="#1c1c3c", lw=0.35, ls=":", zorder=1)

    # ── Outer scale: Wavelengths Toward Generator (outer ring) ───────────────
    # WTG goes 0→0.5 clockwise from right (angle=0)
    wtg_theta = np.linspace(0, -2*np.pi, 1000)   # clockwise
    ax.plot(_R_WTG * np.cos(wtg_theta),
            _R_WTG * np.sin(wtg_theta),
            color="#2a2a50", lw=0.5, zorder=2)

    # ── Outer scale: Wavelengths Toward Load (inner of the two outer rings) ──
    ax.plot(_R_WTL * np.cos(theta),
            _R_WTL * np.sin(theta),
            color="#2a2a50", lw=0.5, zorder=2)

    # ── Tick marks on outer WTG/WTL scales ───────────────────────────────────
    for wl in np.arange(0, 0.505, 0.005):
        ang = -wl * 2 * np.pi          # clockwise = negative
        major = abs(round(wl * 1000) % 50) == 0
        r_in  = _R_WTG - (0.025 if major else 0.012)
        r_out = _R_WTG
        ax.plot([r_in*math.cos(ang), r_out*math.cos(ang)],
                [r_in*math.sin(ang), r_out*math.sin(ang)],
                color="#35355a", lw=0.6 if major else 0.3, zorder=2)
        if major and wl > 0:
            ax.text((_R_WTG - 0.055)*math.cos(ang),
                    (_R_WTG - 0.055)*math.sin(ang),
                    f"{wl:.2f}", fontsize=3.5, color="#3a3a62",
                    ha="center", va="center", rotation=math.degrees(-ang)-90)

    # ── Angle of reflection coefficient degree marks (innermost outer ring) ──
    ax.plot(1.035*np.cos(theta), 1.035*np.sin(theta),
            color="#2a2a50", lw=0.5, zorder=2)
    for deg in range(0, 360, 10):
        ang = math.radians(deg)
        major = deg % 30 == 0
        r_in  = 1.035 - (0.025 if major else 0.012)
        ax.plot([r_in*math.cos(ang), 1.035*math.cos(ang)],
                [r_in*math.sin(ang), 1.035*math.sin(ang)],
                color="#35355a", lw=0.5 if major else 0.25, zorder=2)

    # ── Banner text: component axis labels (PDF layout) ───────────────────────
    kw = dict(ha="center", va="center", zorder=4,
              fontsize=4.0, color="#353560", fontfamily="monospace")

    # Top banner — inductive reactance / capacitive susceptance
    ax.text(0, 1.205,
            "INDUCTIVE REACTANCE COMPONENT (+jX/Zo), OR CAPACITIVE SUSCEPTANCE (+jB/Yo)",
            **{**kw, "fontsize": 3.6})

    # Bottom banner — capacitive reactance / inductive susceptance
    ax.text(0, -1.205,
            "CAPACITIVE REACTANCE COMPONENT (−jX/Zo), OR INDUCTIVE SUSCEPTANCE (−jB/Yo)",
            **{**kw, "fontsize": 3.6})

    # Left side — resistance / conductance
    ax.text(-1.245, 0, "RESISTANCE COMPONENT (R/Zo), OR CONDUCTANCE COMPONENT (G/Yo)",
            rotation=90, **{**kw, "fontsize": 3.6})

    # WTG curved label (top-left arc)
    for i, ch in enumerate("→  WAVELENGTHS TOWARD GENERATOR  →"):
        a = math.radians(155 - i * 4.5)
        ax.text((_R_WTG+0.03)*math.cos(a), (_R_WTG+0.03)*math.sin(a),
                ch, fontsize=3.2, color="#35355a", ha="center", va="center",
                rotation=math.degrees(a)-90, zorder=3)

    # WTL curved label (top-right arc, counter-clockwise arrow)
    for i, ch in enumerate("←  WAVELENGTHS TOWARD LOAD  ←"):
        a = math.radians(25 + i * 4.8)
        ax.text((_R_WTL+0.035)*math.cos(a), (_R_WTL+0.035)*math.sin(a),
                ch, fontsize=3.2, color="#35355a", ha="center", va="center",
                rotation=math.degrees(a)+90, zorder=3)

    # Angle of reflection coefficient label (bottom arc)
    label = "ANGLE OF REFLECTION COEFFICIENT IN DEGREES"
    for i, ch in enumerate(label):
        a = math.radians(-20 - i * 3.9)
        ax.text(1.07*math.cos(a), 1.07*math.sin(a),
                ch, fontsize=3.0, color="#35355a", ha="center", va="center",
                rotation=math.degrees(a)-90, zorder=3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                facecolor=bg, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = _plt_bg.imread(buf)
    _plt_bg.close(fig)
    buf.close()
    return img


# Generate once at import — reused on every redraw
_SMITH_BG: np.ndarray = _generate_background_png(size_px=800)


# =============================================================================
# Helpers
# =============================================================================

def _fmt_freq(f: float) -> str:
    if f >= 1e9:  return f"{f/1e9:.4f} GHz"
    if f >= 1e6:  return f"{f/1e6:.4f} MHz"
    if f >= 1e3:  return f"{f/1e3:.4f} kHz"
    return f"{f:.2f} Hz"


def _si(v: float, unit: str) -> str:
    if unit == "H":
        if v >= 1e-3: return f"{v*1e3:.3g} mH"
        if v >= 1e-6: return f"{v*1e6:.3g} µH"
        return f"{v*1e9:.3g} nH"
    if v >= 1e-9:  return f"{v*1e9:.3g} nF"
    return f"{v*1e12:.3g} pF"


def _make_skrf_net(freqs: np.ndarray, gamma: np.ndarray) -> "skrf.Network":
    """Wrap a Γ array into a 1-port skrf Network."""
    s = gamma.reshape(len(freqs), 1, 1)
    return skrf.Network(
        frequency=skrf.Frequency.from_f(freqs, unit="hz"),
        s=s,
    )


# =============================================================================
# SmithCanvas
# =============================================================================

class SmithCanvas(FigureCanvas):

    def __init__(self, parent=None, dpi=95):
        self.fig = Figure(figsize=(5.2, 5.0), dpi=dpi, facecolor=C["bg"])
        self.ax  = self.fig.add_subplot(111)
        self._result  = None
        self._rlc     = (None, None, None)
        self._markers = []          # list of {idx, num, color}
        self._draw_empty()
        super().__init__(self.fig)
        self.setParent(parent)

    # ── Grid / empty state ────────────────────────────────────────────────────

    def _draw_empty(self):
        """
        Three-layer render exactly matching the skrf docs pattern:

            ax.imshow(background, extent=[...])        Layer 1
            skrf.plotting.smith(ax, draw_labels=True)  Layer 2
            net.plot_s_smith(ax=ax)                    Layer 3 (in _redraw)
        """
        ax = self.ax
        ax.set_facecolor(C["axes_bg"])
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.tick_params(colors=C["text"], labelsize=7)

        # ── Layer 1: background image ─────────────────────────────────────────
        ax.imshow(
            _SMITH_BG,
            extent=_EXTENT,
            aspect="equal",
            origin="upper",
            zorder=0,
            interpolation="bilinear",
        )

        # ── Layer 2: skrf.plotting.smith — full labelled grid ─────────────────
        if HAS_SKRF:
            try:
                skrf.plotting.smith(
                    ax=ax,
                    draw_labels=True,
                    ref_imm=1.0,
                    chart_type="z",
                )
                # Restyle skrf's lines/text to our dark theme
                _own = {C["green"], C["yellow"], C["red"], C["blue"],
                        C["mauve"], C["peach"], C["teal"], C["sapphire"],
                        C["lavender"], C["sky"]}
                for ln in ax.lines:
                    if ln.get_color() not in _own:
                        ln.set_color(C["grid"])
                        ln.set_linewidth(0.7)
                        ln.set_alpha(0.80)
                        ln.set_zorder(3)
                for txt in ax.texts:
                    txt.set_color(C["subtext"])
                    txt.set_fontsize(5.5)
                    txt.set_zorder(4)
            except Exception:
                self._draw_manual_grid(ax)
        else:
            self._draw_manual_grid(ax)

        # ── Axis chrome ───────────────────────────────────────────────────────
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_title("Smith Chart  (Γ-plane  ·  Z chart)", color=C["text"],
                     fontsize=9, pad=4)
        ax.set_xlabel("Re(Γ)", color=C["subtext"], fontsize=8)
        ax.set_ylabel("Im(Γ)", color=C["subtext"], fontsize=8)

        # ── R/L/C annotation strip ────────────────────────────────────────────
        R, L, Cv = self._rlc
        if R is not None:
            ann = (f"R = {R:.4g} Ω     "
                   f"L = {_si(L,'H')}     "
                   f"C = {_si(Cv,'F')}")
            ax.text(0.5, -0.055, ann,
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=6.5, color=C["subtext"],
                    fontfamily="monospace", zorder=5)

        self.fig.tight_layout(pad=1.4)

    # ── Manual grid fallback (skrf absent) ────────────────────────────────────

    @staticmethod
    def _draw_manual_grid(ax):
        theta = np.linspace(0, 2*np.pi, 500)
        ax.plot(np.cos(theta), np.sin(theta), color=C["border"], lw=1.4, zorder=3)
        for r in [0, 0.2, 0.5, 1.0, 2.0, 5.0]:
            cx  = r/(r+1); rad = 1/(r+1)
            x   = cx + rad*np.cos(theta)
            y   =      rad*np.sin(theta)
            mask = x**2+y**2 <= 1.002
            ax.plot(np.where(mask,x,np.nan), np.where(mask,y,np.nan),
                    color=C["grid"], lw=0.7, zorder=3)
            if r > 0:
                ax.text(cx+rad+0.02, 0.02, f"{r}", fontsize=5,
                        color=C["subtext"], ha="left", zorder=4)
        t = np.linspace(0, 2*np.pi, 700)
        for xv in [0.2, 0.5, 1.0, 2.0, 5.0]:
            for sign in (1, -1):
                cx2, cy2 = 1.0, sign/xv;  rad2 = abs(1/xv)
                pts = np.column_stack([cx2+rad2*np.cos(t), cy2+rad2*np.sin(t)])
                inside = pts[:,0]**2+pts[:,1]**2 <= 1.002
                ax.plot(np.where(inside,pts[:,0],np.nan),
                        np.where(inside,pts[:,1],np.nan),
                        color=C["grid"], lw=0.7, zorder=3)
                if inside.any():
                    li = np.where(inside)[0][len(np.where(inside)[0])//2]
                    ax.text(pts[li,0], pts[li,1], f"{sign*xv:+.1f}j",
                            fontsize=4.5, color=C["subtext"],
                            ha="center", va="center", zorder=4)
        ax.axhline(0, color=C["grid"], lw=0.7, zorder=3)
        ax.plot(0, 0, "+", color=C["border"], ms=8, zorder=3)

    # ── Public API ────────────────────────────────────────────────────────────

    def plot_result(self, result: dict, R=None, L=None, C=None):
        self._result = result
        self._rlc    = (R, L, C)
        self._markers.clear()
        self._redraw()

    def clear_markers(self):
        self._markers.clear()
        if self._result:
            self._redraw()

    # ── Redraw ────────────────────────────────────────────────────────────────

    def _redraw(self):
        result = self._result
        self.ax.cla()
        self._draw_empty()          # Layers 1 + 2

        freqs = result["frequencies"]
        gamma = result["gamma"]
        re, im = gamma.real, gamma.imag

        # ── Layer 3a: skrf net.plot_s_smith() — the measurement trace ─────────
        if HAS_SKRF:
            net = _make_skrf_net(freqs, gamma)
            try:
                net.plot_s_smith(
                    ax=self.ax,
                    show_legend=False,
                    draw_labels=False,   # already drawn by skrf.plotting.smith
                    draw_vswr=False,
                    color=C["blue"],
                    linewidth=2.2,
                    alpha=0.95,
                    zorder=6,
                )
            except Exception:
                self._fallback_trace(re, im)
        else:
            self._fallback_trace(re, im)

        # ── Layer 3b: VSWR circle through the resonance point ─────────────────
        res_idx = int(np.argmin(result["s11_db"]))
        gm_res  = abs(gamma[res_idx])
        if gm_res < 0.999:
            th = np.linspace(0, 2*np.pi, 400)
            self.ax.plot(gm_res*np.cos(th), gm_res*np.sin(th),
                         color=C["peach"], lw=0.8, ls="--", alpha=0.55,
                         zorder=5, label=f"VSWR circle (res)")

        # ── Key points ────────────────────────────────────────────────────────
        kw = dict(zorder=8, clip_on=True)
        self.ax.plot(re[0],       im[0],       "o",
                     color=C["green"],  ms=8,
                     label=f"Start  {_fmt_freq(freqs[0])}", **kw)
        self.ax.plot(re[res_idx], im[res_idx], "*",
                     color=C["yellow"], ms=13,
                     label=f"Res    {_fmt_freq(freqs[res_idx])}", **kw)
        self.ax.plot(re[-1],      im[-1],      "s",
                     color=C["red"],    ms=7,
                     label=f"Stop   {_fmt_freq(freqs[-1])}", **kw)

        # ── Numbered user markers ─────────────────────────────────────────────
        for mk in self._markers:
            idx, num, col = mk["idx"], mk["num"], mk["color"]
            self.ax.plot(re[idx], im[idx], "D", color=col, ms=9, zorder=10)
            # Readout label: freq + Z value
            Z  = result["Z_L"][idx]
            Z0 = result["Z0"]
            lbl = (f"M{num}\n"
                   f"{_fmt_freq(freqs[idx])}\n"
                   f"Z={Z.real:.1f}{'+' if Z.imag>=0 else ''}{Z.imag:.1f}j Ω\n"
                   f"|Γ|={abs(gamma[idx]):.3f}")
            self.ax.annotate(
                lbl,
                xy=(re[idx], im[idx]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=5.5, fontweight="bold", color=col,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor=C["bg"], edgecolor=col, alpha=0.93),
                zorder=11,
            )

        self.ax.legend(fontsize=5.5, facecolor=C["grid"],
                       labelcolor=C["text"], edgecolor=C["border"],
                       loc="lower left", framealpha=0.85)
        self.fig.tight_layout(pad=1.4)
        self.draw()

    def _fallback_trace(self, re, im):
        n    = len(re)
        pts  = np.column_stack([re, im]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        t    = np.linspace(0, 1, max(n-1, 1))
        cols = [(v*0.9, 0.25, 1.0-v*0.9, 0.88) for v in t]
        lc   = LineCollection(segs, colors=cols, linewidth=2.2, zorder=6)
        self.ax.add_collection(lc)

    # ── Marker placement ──────────────────────────────────────────────────────

    def place_marker(self, x: float, y: float):
        if self._result is None:
            return None
        if len(self._markers) >= 8:
            self._markers.pop(0)

        gamma = self._result["gamma"]
        freqs = self._result["frequencies"]
        dists = (gamma.real - x)**2 + (gamma.imag - y)**2
        idx   = int(np.argmin(dists))
        num   = len(self._markers) + 1
        color = MARKER_COLORS[(num-1) % len(MARKER_COLORS)]

        self._markers.append(dict(idx=idx, num=num, color=color))
        self._redraw()

        return dict(
            f    = freqs[idx],
            Z_L  = self._result["Z_L"][idx],
            gamma= gamma[idx],
            s11  = self._result["s11_db"][idx],
            vswr = self._result["vswr"][idx],
            gmag = float(np.abs(gamma[idx])),
        )