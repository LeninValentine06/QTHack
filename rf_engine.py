"""
rf_engine.py  —  VNA Reflection Simulator  ·  Core physics engine

Scope: RLC → Z(f) → Γ → S11 → VSWR → Phase → Group Delay → Bandwidth
Out of scope: noise, cable delay, IF-BW smoothing, skin effect,
              instrument-level modelling of any kind.

IDEAL_MODE = True (default)
    Only the five core physics equations run. No post-processing.
    Set False only if you need to re-add instrument effects later.

Public API (unchanged for GUI compatibility):
    run_simulation(**kwargs) -> dict
"""

import numpy as np

# ── Module-level mode flag ────────────────────────────────────────────────────
IDEAL_MODE: bool = True          # True = pure physics, no extras

# ── Numerical stability constant ─────────────────────────────────────────────
# One practical near-zero guard used everywhere.
# 1e-12 is below any physically meaningful impedance but well above
# float64 underflow, keeping all divisions and log() calls safe.
_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Input validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_inputs(f_start: float, f_stop: float, n_points: int,
                    R: float, L: float, C: float) -> None:
    """
    Check every simulation parameter before any computation begins.
    Raises ValueError with a descriptive message on the first bad value.
    """
    if f_start <= 0:
        raise ValueError(f"f_start must be > 0 Hz  (got {f_start})")
    if f_stop <= f_start:
        raise ValueError(
            f"f_stop must be > f_start  (got f_stop={f_stop}, f_start={f_start})")
    if int(n_points) < 2:
        raise ValueError(f"n_points must be >= 2  (got {n_points})")
    if R < 0:
        raise ValueError(f"R must be >= 0 Ohm  (got {R})")
    if L <= 0:
        raise ValueError(f"L must be > 0 H  (got {L})")
    if C <= 0:
        raise ValueError(f"C must be > 0 F  (got {C})")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Frequency sweep
# ─────────────────────────────────────────────────────────────────────────────

def frequency_sweep(f_start: float, f_stop: float, n_points: int,
                    sweep_type: str = "log") -> np.ndarray:
    """
    Return an array of N frequency points (Hz).

    sweep_type = "log"    -> logarithmically spaced (default, preferred for RF)
    sweep_type = "linear" -> linearly spaced
    """
    n = int(n_points)
    if sweep_type == "linear":
        return np.linspace(f_start, f_stop, n)
    return np.logspace(np.log10(f_start), np.log10(f_stop), n)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Core physics functions (the five equations)
# ─────────────────────────────────────────────────────────────────────────────

def compute_impedance(freqs: np.ndarray,
                      R: float, L: float, C: float) -> np.ndarray:
    """
    Series RLC antenna impedance:

        Z(f) = R + j*(omega*L  -  1/(omega*C))      omega = 2*pi*f

    Returns a complex128 array, one value per frequency point.
    The omega = 0 guard keeps 1/(omega*C) finite; it never fires on valid
    sweeps because validate_inputs() already enforced f_start > 0.
    """
    omega = 2.0 * np.pi * freqs
    omega = np.where(omega == 0.0, _EPS, omega)   # protect 1/(omega*C) at DC
    return R + 1j * (omega * L - 1.0 / (omega * C))


def compute_gamma(Z: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """
    Reflection coefficient:

        Gamma = (Z - Z0) / (Z + Z0)

    The denominator guard handles the pathological case Z = -Z0.
    Cannot arise from a passive RLC load, but protects against
    extreme floating-point edge cases.
    """
    denom = Z + Z0
    denom = np.where(np.abs(denom) < _EPS, _EPS + 0j, denom)
    return (Z - Z0) / denom


def compute_s11_db(gamma: np.ndarray) -> np.ndarray:
    """
    S11 in dB:

        S11(dB) = 20 * log10(|Gamma|)

    The floor clip prevents log10(0) = -inf at a perfect match (|Gamma|=0).
    """
    mag = np.clip(np.abs(gamma), _EPS, None)
    return 20.0 * np.log10(mag)


def compute_vswr(gamma: np.ndarray) -> np.ndarray:
    """
    Voltage Standing Wave Ratio:

        VSWR = (1 + |Gamma|) / (1 - |Gamma|)

    Upper clip on |Gamma| keeps the denominator from reaching zero.
    For a passive load |Gamma| < 1 always; the clip is a pure safety guard.
    """
    mag = np.clip(np.abs(gamma), 0.0, 1.0 - _EPS)
    return (1.0 + mag) / (1.0 - mag)


def compute_phase(gamma: np.ndarray) -> np.ndarray:
    """
    Phase of Gamma in degrees, wrapped to (-180, +180].

        phi = angle(Gamma)
    """
    return np.angle(gamma, deg=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Derived display quantities (required by GUI and plot modules)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_return_loss(gamma: np.ndarray) -> np.ndarray:
    """RL(dB) = -S11(dB)   positive value = good match"""
    return -compute_s11_db(gamma)


def _compute_phase_unwrapped(gamma: np.ndarray) -> np.ndarray:
    """Unwrapped phase in degrees — removes 360-degree discontinuities."""
    return np.degrees(np.unwrap(np.angle(gamma)))


def _compute_group_delay(freqs: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Group delay:  tau = -dphi/domega  (nanoseconds)

    np.gradient(y, x) computes dy/dx with correct non-uniform spacing —
    this is the right call for log sweeps.  The old manual dphi/domega
    computation was equivalent only for uniform spacing and gave wrong
    values at every interior point on a log sweep.

    IQR spike clip removes boundary artefacts (display only).
    """
    phase_rad = np.unwrap(np.angle(gamma))
    omega     = 2.0 * np.pi * freqs

    gd_ns = -np.gradient(phase_rad, omega) * 1e9     # Fix #1: single call

    q25, q75 = np.percentile(gd_ns, [25, 75])
    iqr       = max(q75 - q25, 1.0)
    gd_ns     = np.clip(gd_ns, q25 - 5.0 * iqr, q75 + 5.0 * iqr)
    return gd_ns


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Bandwidth and resonance detection
# ─────────────────────────────────────────────────────────────────────────────

def _refine_resonance(freqs: np.ndarray, s11_db: np.ndarray) -> float:
    """
    Sub-sample resonance location via 3-point Lagrange quadratic interpolation
    in log-frequency space.  Gives sub-bin accuracy without extra computation.
    Falls back to the raw argmin point if the parabola is ill-conditioned.
    """
    k = int(np.argmin(s11_db))
    if k == 0 or k == len(freqs) - 1:
        return float(freqs[k])

    lf = np.log(freqs[k-1:k+2])
    sv = s11_db[k-1:k+2]
    x0, x1, x2 = lf
    y0, y1, y2 = sv

    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < _EPS:
        return float(freqs[k])

    A = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
    B = (x2**2*(y0-y1) + x1**2*(y2-y0) + x0**2*(y1-y2)) / denom

    if abs(A) < _EPS or A > 0:
        return float(freqs[k])

    log_f_min = -B / (2.0 * A)
    if not (lf[0] <= log_f_min <= lf[2]):
        return float(freqs[k])
    return float(np.exp(log_f_min))


def _compute_bandwidth(freqs: np.ndarray, s11_db: np.ndarray,
                        threshold_db: float = -10.0) -> dict:
    """
    Find the -10 dB bandwidth using log-frequency crossing interpolation.
    Returns a dict consumed directly by gui.py and plot_s11.py.
    """
    res_idx = int(np.argmin(s11_db))
    s11_min = float(s11_db[res_idx])
    f_res   = _refine_resonance(freqs, s11_db)

    result = dict(f_res=f_res, s11_min=s11_min,
                  f_low=None, f_high=None, bandwidth=None, valid=False)

    if s11_min > threshold_db:
        return result

    # Walk left from resonance to find f_low
    f_low = None
    for i in range(res_idx, -1, -1):
        if s11_db[i] >= threshold_db:
            if i + 1 <= res_idx:
                t = ((threshold_db - s11_db[i])
                     / (s11_db[i+1] - s11_db[i] + _EPS))
                f_low = np.exp(
                    np.log(freqs[i])
                    + t * (np.log(freqs[i+1]) - np.log(freqs[i])))
            else:
                f_low = freqs[i]
            break

    # Walk right from resonance to find f_high
    f_high = None
    for i in range(res_idx, len(freqs)):
        if s11_db[i] >= threshold_db:
            if i > 0:
                t = ((threshold_db - s11_db[i-1])
                     / (s11_db[i] - s11_db[i-1] + _EPS))
                f_high = np.exp(
                    np.log(freqs[i-1])
                    + t * (np.log(freqs[i]) - np.log(freqs[i-1])))
            else:
                f_high = freqs[i]
            break

    if f_low is not None and f_high is not None:
        result.update(f_low=f_low, f_high=f_high,
                      bandwidth=float(f_high - f_low), valid=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Full simulation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(f_start: float, f_stop: float, n_points: int,
                   R: float, L: float, C: float,
                   Z0: float = 50.0,
                   sweep_type: str = "log",
                   if_bw: float = 1000.0,
                   output_power_dbm: float = -10.0,
                   **_ignored) -> dict:
    """
    Run the complete reflection physics simulation.

    Parameters
    ----------
    f_start, f_stop  : sweep range in Hz
    n_points         : number of frequency points
    R, L, C          : series RLC antenna model parameters
    Z0               : reference impedance, default 50 Ohm
    sweep_type       : "log" (default) or "linear"
    if_bw            : accepted for GUI compatibility, unused in IDEAL_MODE
    output_power_dbm : accepted for GUI compatibility, unused in IDEAL_MODE
    **_ignored       : absorbs cable_length, k_skin, seed etc. silently
                       so any existing caller keeps working without changes

    Returns
    -------
    dict — all keys expected by gui.py, plot_s11.py, smith_chart.py
    """
    # 1. Validate — raises ValueError before any numpy work
    validate_inputs(f_start, f_stop, n_points, R, L, C)

    # 2. Frequency array
    freqs = frequency_sweep(f_start, f_stop, int(n_points), sweep_type)

    # 3. Core physics — five equations, nothing else
    Z_L   = compute_impedance(freqs, R, L, C)    # Z(f) = R + j(wL - 1/wC)
    gamma = compute_gamma(Z_L, Z0)               # Gamma = (Z-Z0)/(Z+Z0)
    s11   = compute_s11_db(gamma)                # 20*log10|Gamma|
    vswr  = compute_vswr(gamma)                  # (1+|G|)/(1-|G|)
    phase = compute_phase(gamma)                 # angle(Gamma) degrees

    # 4. Derived display quantities (no new physics — math on gamma/Z only)
    rl       = _compute_return_loss(gamma)
    phase_uw = _compute_phase_unwrapped(gamma)
    gd       = _compute_group_delay(freqs, gamma)
    z_re, z_im = Z_L.real, Z_L.imag

    # 5. Bandwidth detection and resonance cross-check
    bw = _compute_bandwidth(freqs, s11)

    f_theory  = 1.0 / (2.0 * np.pi * np.sqrt(L * C))
    f_sim     = bw["f_res"]
    dev_pct   = abs(f_sim - f_theory) / f_theory * 100.0
    out_range = not (f_start <= f_theory <= f_stop)
    warn      = dev_pct > 1.0 or out_range

    if out_range:
        warn_msg = (f"Warning: theoretical resonance {f_theory/1e6:.3f} MHz "
                    f"is outside the sweep range.")
    elif dev_pct > 1.0:
        warn_msg = (f"Warning: resonance deviates {dev_pct:.2f}% from "
                    f"theoretical {f_theory/1e6:.4f} MHz.")
    else:
        warn_msg = ""

    # 6. Return structured result dict
    #    Key names are fixed — gui.py, plot_s11.py, smith_chart.py all
    #    depend on them.  Do not rename without updating those modules.
    return dict(
        # ── Core physics arrays ───────────────────────────────────────────
        frequencies     = freqs,
        Z_L             = Z_L,
        gamma           = gamma,
        s11_db          = s11,
        vswr            = vswr,
        return_loss     = rl,
        phase_deg       = phase,
        phase_unwrapped = phase_uw,
        group_delay_ns  = gd,
        z_real          = z_re,
        z_imag          = z_im,

        # ── Scalars passed through for GUI readout ────────────────────────
        Z0              = Z0,
        sweep_type      = sweep_type,
        if_bw_hz        = if_bw,
        output_power_dbm= output_power_dbm,

        # ── Bandwidth sub-dict ────────────────────────────────────────────
        bandwidth       = bw,

        # ── Resonance validation sub-dict ─────────────────────────────────
        resonance_check = dict(
            f_theoretical = f_theory,
            f_simulated   = f_sim,
            deviation_pct = dev_pct,
            outside_sweep = out_range,
            warning       = warn,
            message       = warn_msg,
        ),
    )