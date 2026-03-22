"""
rf_engine.py  —  Core RF computation engine
All antenna math: impedance, reflection coefficient, S11, VSWR,
phase, group delay, polar, bandwidth.
"""

import numpy as np


# ─── Frequency sweep ──────────────────────────────────────────────────────────

def frequency_sweep(f_start: float, f_stop: float, n_points: int,
                    sweep_type: str = "log") -> np.ndarray:
    """Log or linear frequency array."""
    n = int(n_points)
    if sweep_type == "linear":
        return np.linspace(f_start, f_stop, n)
    return np.logspace(np.log10(f_start), np.log10(f_stop), n)


# ─── Impedance model ─────────────────────────────────────────────────────────

def compute_antenna_impedance(freqs: np.ndarray, R: float, L: float,
                               C: float) -> np.ndarray:
    """Series RLC: Z_L(f) = R + j*(ωL - 1/ωC)"""
    omega = 2.0 * np.pi * freqs
    omega = np.where(omega == 0, 1e-30, omega)
    return R + 1j * (omega * L - 1.0 / (omega * C))


# ─── Reflection coefficient ───────────────────────────────────────────────────

def compute_gamma(Z_L: np.ndarray, Z0: float = 50.0) -> np.ndarray:
    """Γ = (Z_L - Z0) / (Z_L + Z0)"""
    denom = Z_L + Z0
    denom = np.where(np.abs(denom) < 1e-30, 1e-30 + 0j, denom)
    return (Z_L - Z0) / denom


# ─── Derived quantities ───────────────────────────────────────────────────────

def compute_s11_db(gamma: np.ndarray) -> np.ndarray:
    """S11 (dB) = 20·log10|Γ|"""
    mag = np.clip(np.abs(gamma), 1e-15, None)
    return 20.0 * np.log10(mag)


def compute_vswr(gamma: np.ndarray) -> np.ndarray:
    """VSWR = (1+|Γ|)/(1-|Γ|).  Guard prevents div/0 at |Γ|=1."""
    mag = np.clip(np.abs(gamma), 0.0, 1.0 - 1e-9)
    return (1.0 + mag) / (1.0 - mag)


def compute_return_loss(gamma: np.ndarray) -> np.ndarray:
    """Return Loss (dB) = -S11(dB)  (positive = good match)"""
    return -compute_s11_db(gamma)


def compute_phase_deg(gamma: np.ndarray) -> np.ndarray:
    """Phase of Γ in degrees, wrapped to (-180, 180]."""
    return np.angle(gamma, deg=True)


def compute_phase_unwrapped_deg(gamma: np.ndarray) -> np.ndarray:
    """Unwrapped phase of Γ in degrees — no 360° discontinuities."""
    return np.degrees(np.unwrap(np.angle(gamma)))


def compute_group_delay(freqs: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Group delay τ = -dφ/dω  (nanoseconds).
    Uses central differences on the unwrapped phase.
    """
    phase_rad = np.unwrap(np.angle(gamma))
    omega     = 2.0 * np.pi * freqs
    # Central difference; forward/backward at edges
    dphi  = np.gradient(phase_rad, omega)
    return -dphi * 1e9   # convert s → ns


def compute_polar(gamma: np.ndarray):
    """Return (magnitude, phase_deg) for polar display."""
    return np.abs(gamma), np.angle(gamma, deg=True)


def compute_z_real_imag(Z_L: np.ndarray):
    """Return (Re(Z), Im(Z)) normalised to Ω."""
    return Z_L.real, Z_L.imag


# ─── Resonance refinement ─────────────────────────────────────────────────────

def _refine_resonance(freqs: np.ndarray, s11_db: np.ndarray) -> float:
    """Sub-sample resonance via 3-point Lagrange quadratic in log-freq."""
    k = int(np.argmin(s11_db))
    if k == 0 or k == len(freqs) - 1:
        return float(freqs[k])
    lf = np.log(freqs[k-1:k+2])
    sv = s11_db[k-1:k+2]
    x0, x1, x2 = lf; y0, y1, y2 = sv
    denom = (x0-x1)*(x0-x2)*(x1-x2)
    if abs(denom) < 1e-30:
        return float(freqs[k])
    A = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
    B = (x2**2*(y0-y1) + x1**2*(y2-y0) + x0**2*(y1-y2)) / denom
    if abs(A) < 1e-30 or A > 0:
        return float(freqs[k])
    log_f_min = -B / (2.0 * A)
    if not (lf[0] <= log_f_min <= lf[2]):
        return float(freqs[k])
    return float(np.exp(log_f_min))


# ─── Bandwidth detection ──────────────────────────────────────────────────────

def compute_bandwidth(freqs: np.ndarray, s11_db: np.ndarray,
                      threshold_db: float = -10.0) -> dict:
    """Find -10 dB bandwidth with log-frequency interpolation."""
    res_idx = int(np.argmin(s11_db))
    s11_min = s11_db[res_idx]
    f_res   = _refine_resonance(freqs, s11_db)

    result = dict(f_res=f_res, s11_min=s11_min,
                  f_low=None, f_high=None, bandwidth=None, valid=False)

    if s11_min > threshold_db:
        return result

    f_low = None
    for i in range(res_idx, -1, -1):
        if s11_db[i] >= threshold_db:
            if i + 1 <= res_idx:
                t = (threshold_db - s11_db[i]) / (s11_db[i+1] - s11_db[i] + 1e-30)
                f_low = np.exp(np.log(freqs[i]) + t*(np.log(freqs[i+1])-np.log(freqs[i])))
            else:
                f_low = freqs[i]
            break

    f_high = None
    for i in range(res_idx, len(freqs)):
        if s11_db[i] >= threshold_db:
            if i > 0:
                t = (threshold_db - s11_db[i-1]) / (s11_db[i]-s11_db[i-1] + 1e-30)
                f_high = np.exp(np.log(freqs[i-1]) + t*(np.log(freqs[i])-np.log(freqs[i-1])))
            else:
                f_high = freqs[i]
            break

    if f_low is not None and f_high is not None:
        result.update(f_low=f_low, f_high=f_high,
                      bandwidth=f_high-f_low, valid=True)
    return result


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_simulation(f_start: float, f_stop: float, n_points: int,
                   R: float, L: float, C: float, Z0: float = 50.0,
                   sweep_type: str = "log",
                   if_bw: float = 1000.0,
                   output_power_dbm: float = -10.0) -> dict:
    """
    Run the complete VNA simulation pipeline.
    Returns all computed arrays plus resonance validation.
    """
    freqs = frequency_sweep(f_start, f_stop, n_points, sweep_type)
    Z_L   = compute_antenna_impedance(freqs, R, L, C)
    gamma = compute_gamma(Z_L, Z0)
    s11   = compute_s11_db(gamma)
    vswr  = compute_vswr(gamma)
    rl    = compute_return_loss(gamma)
    phase = compute_phase_deg(gamma)
    phase_uw = compute_phase_unwrapped_deg(gamma)
    gd    = compute_group_delay(freqs, gamma)
    z_re, z_im = compute_z_real_imag(Z_L)
    bw    = compute_bandwidth(freqs, s11)

    # Theoretical resonance check
    f_theoretical = 1.0 / (2.0 * np.pi * np.sqrt(L * C))
    f_simulated   = bw["f_res"]
    deviation_pct = abs(f_simulated - f_theoretical) / f_theoretical * 100.0
    outside_sweep = not (f_start <= f_theoretical <= f_stop)
    warn = deviation_pct > 1.0 or outside_sweep
    warning_msg = ""
    if outside_sweep:
        warning_msg = (f"⚠ Theoretical resonance {f_theoretical/1e6:.3f} MHz "
                       f"is outside sweep range.")
    elif deviation_pct > 1.0:
        warning_msg = (f"⚠ Resonance deviates {deviation_pct:.2f}% from "
                       f"theoretical {f_theoretical/1e6:.4f} MHz.")

    return dict(
        frequencies      = freqs,
        Z_L              = Z_L,
        gamma            = gamma,
        s11_db           = s11,
        vswr             = vswr,
        return_loss      = rl,
        phase_deg        = phase,
        phase_unwrapped  = phase_uw,
        group_delay_ns   = gd,
        z_real           = z_re,
        z_imag           = z_im,
        bandwidth        = bw,
        Z0               = Z0,
        sweep_type       = sweep_type,
        if_bw_hz         = if_bw,
        output_power_dbm = output_power_dbm,
        resonance_check  = dict(
            f_theoretical = f_theoretical,
            f_simulated   = f_simulated,
            deviation_pct = deviation_pct,
            outside_sweep = outside_sweep,
            warning       = warn,
            message       = warning_msg,
        ),
    )