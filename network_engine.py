"""
network_engine.py  —  VNA Simulator v2 · Cascaded Network Backend
===================================================================

Drop-in computation layer that replaces the single-RLC physics in
rf_engine.py with a full cascaded-network solver backed by scikit-rf.

Public API
----------
compute_network_response(network_config, freq_array, Z0=50.0,
                         mode="skrf")  -> dict

The returned dict is a strict superset of what rf_engine.run_simulation()
produces, so every existing GUI / plot consumer keeps working unchanged.

Network config format
---------------------
A list of component dicts processed left-to-right (source → load).

The LAST element is the terminal load (antenna, resistor, etc.).
All preceding elements are in-line 2-ports (TL, matching elements, losses).

    network_config = [
        {"type": "R",  "value": 10.0},           # series loss (Ω)
        {"type": "TL", "Z0": 50.0,               # coax run
                       "length": 0.5, "vf": 0.66},
        {"type": "TL", "Z0": 50.0,               # electrical-length TL
                       "el_deg": 90.0, "el_ref_hz": 2.4e9},
        {"type": "RLC",                           # terminal antenna load
                       "R": 73.0, "L": 35e-9, "C": 7e-12},
    ]

Supported types
---------------
    "R"    – series resistor (Ω)
    "L"    – series inductor (H)
    "C"    – series capacitor (F)
    "TL"   – lossless transmission line (physical or electrical length)
    "RLC"  – series RLC shorthand (R + jωL + 1/jωC)
    "Z"    – arbitrary complex impedance array (advanced use)

Load vs in-line elements
------------------------
    Terminal load  : last element; its impedance Z_load feeds into the
                     signal-flow-graph termination equation.
    In-line element: all preceding elements; each is a 2-port network.

Signal-flow-graph termination (the physics)
-------------------------------------------
Given a 2-port cascade with S-params [S11 S12; S21 S22] and a load
reflection coefficient Γ_L = (Z_load − Z0) / (Z_load + Z0):

    Γ_in = S11 + (S12 · Γ_L · S21) / (1 − S22 · Γ_L)

This is exact for lossless and lossy networks (Pozar §4.3).
For a trivial cascade (no in-line elements), S11=0, S12=S21=1, S22=0,
so Γ_in = Γ_L — identical to rf_engine.compute_gamma().

mode flag
---------
    "skrf"   (default) – use scikit-rf cascade for in-line 2-ports
    "manual" – analytical series-impedance summation (R/L/C/RLC only)

Design constraints respected
-----------------------------
- Does NOT touch gui.py, plot_s11.py, smith_chart.py, export_utils.py
- Output dict keys are identical to rf_engine.run_simulation()
- Numerically stable at all frequencies
- Graceful ImportError when scikit-rf is absent (manual mode still works)
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# ── Optional scikit-rf import ─────────────────────────────────────────────────
try:
    import skrf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False

# ── Shared constants ──────────────────────────────────────────────────────────
_EPS = 1e-12
_C0  = 299_792_458.0   # speed of light in vacuum (m/s)


# =============================================================================
# Section 1 — Impedance functions (component physics)
# =============================================================================

def _z_resistor(freqs: np.ndarray, R: float) -> np.ndarray:
    return np.full(len(freqs), R + 0j)


def _z_inductor(freqs: np.ndarray, L: float) -> np.ndarray:
    return 1j * 2.0 * np.pi * freqs * L


def _z_capacitor(freqs: np.ndarray, C: float) -> np.ndarray:
    omega = 2.0 * np.pi * freqs
    omega = np.where(np.abs(omega) < _EPS, _EPS, omega)
    return 1.0 / (1j * omega * C)


def _z_rlc(freqs: np.ndarray,
           R: float, L: float, C: float) -> np.ndarray:
    """Series RLC: Z = R + jωL + 1/(jωC)"""
    omega = 2.0 * np.pi * freqs
    omega = np.where(np.abs(omega) < _EPS, _EPS, omega)
    return R + 1j * (omega * L - 1.0 / (omega * C))


# =============================================================================
# Section 2 — 2-port S-matrix builders (in-line elements)
# =============================================================================

def _make_freq_obj(freq_array: np.ndarray) -> "skrf.Frequency":
    return skrf.Frequency.from_f(freq_array, unit="hz")


def _series_2port(freq_obj: "skrf.Frequency",
                  Z_series: np.ndarray,
                  Z0: float) -> "skrf.Network":
    """
    Series-impedance 2-port.  ABCD = [[1, Z], [0, 1]].

    S-parameters (Z0-normalised, Pozar §4.4):
        S11 = S22 = Z / (Z + 2·Z0)
        S12 = S21 = 2·Z0 / (Z + 2·Z0)
    """
    n = len(freq_obj)
    s = np.zeros((n, 2, 2), dtype=complex)
    denom = Z_series + 2.0 * Z0
    denom = np.where(np.abs(denom) < _EPS, _EPS + 0j, denom)
    s[:, 0, 0] = Z_series / denom
    s[:, 1, 1] = s[:, 0, 0]
    s[:, 0, 1] = 2.0 * Z0 / denom
    s[:, 1, 0] = s[:, 0, 1]
    return skrf.Network(frequency=freq_obj, s=s)


def _tl_2port(freq_obj: "skrf.Frequency",
              tl_Z0: float,
              Z_ref: float,
              length: Optional[float] = None,
              vf: float = 1.0,
              el_deg: Optional[float] = None,
              el_ref_hz: Optional[float] = None) -> "skrf.Network":
    """
    Lossless transmission line 2-port.

    Electrical length θ(f) — two options:
      a) Physical length (m) + velocity factor:
             θ = 2πf · L / (vf · c0)
      b) Electrical length in degrees at a reference frequency:
             θ = el_deg · (f / f_ref) · π/180

    ABCD matrix (lossless TL, Pozar §2.4):
        [[cos θ,         j·Z_tl·sin θ],
         [j·sin θ/Z_tl,  cos θ       ]]

    Converted to S-parameters normalised to Z_ref (Pozar §4.4).
    """
    freqs = freq_obj.f
    n = len(freqs)

    if el_deg is not None:
        ref_f = el_ref_hz if el_ref_hz is not None else freqs[0]
        if abs(ref_f) < _EPS:
            raise ValueError("el_ref_hz must be > 0 when el_deg is given.")
        theta = el_deg * (freqs / ref_f) * (np.pi / 180.0)
    elif length is not None:
        if not (0.0 < vf <= 1.0):
            raise ValueError(f"vf must be in (0, 1]. Got {vf}.")
        theta = 2.0 * np.pi * freqs * length / (vf * _C0)
    else:
        raise ValueError("Supply either 'length' or 'el_deg'.")

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    A = cos_t
    B = 1j * tl_Z0 * sin_t
    C = 1j * sin_t / tl_Z0
    D = cos_t

    denom = A + B / Z_ref + C * Z_ref + D
    denom = np.where(np.abs(denom) < _EPS, _EPS + 0j, denom)

    s = np.zeros((n, 2, 2), dtype=complex)
    s[:, 0, 0] = (A + B / Z_ref - C * Z_ref - D) / denom
    s[:, 0, 1] = 2.0 * (A * D - B * C) / denom
    s[:, 1, 0] = 2.0 / denom
    s[:, 1, 1] = (-A + B / Z_ref - C * Z_ref + D) / denom
    return skrf.Network(frequency=freq_obj, s=s)


# =============================================================================
# Section 3 — Component dispatchers
# =============================================================================

def _build_inline_2port(comp: Dict[str, Any],
                        freq_obj: "skrf.Frequency",
                        Z0: float) -> "skrf.Network":
    """Return a 2-port Network for one in-line component."""
    t = comp["type"].upper()

    if t == "R":
        return _series_2port(freq_obj, _z_resistor(freq_obj.f, comp["value"]), Z0)
    elif t == "L":
        return _series_2port(freq_obj, _z_inductor(freq_obj.f, comp["value"]), Z0)
    elif t == "C":
        return _series_2port(freq_obj, _z_capacitor(freq_obj.f, comp["value"]), Z0)
    elif t == "RLC":
        return _series_2port(
            freq_obj,
            _z_rlc(freq_obj.f, comp.get("R", 0.0),
                   comp.get("L", 1e-30), comp.get("C", 1e-30)),
            Z0,
        )
    elif t == "TL":
        return _tl_2port(
            freq_obj,
            tl_Z0    = comp.get("Z0", Z0),
            Z_ref    = Z0,
            length   = comp.get("length", None),
            vf       = comp.get("vf", 1.0),
            el_deg   = comp.get("el_deg", None),
            el_ref_hz= comp.get("el_ref_hz", None),
        )
    elif t == "Z":
        Z = np.asarray(comp["value"], dtype=complex)
        if Z.ndim == 0:
            Z = np.full(len(freq_obj), Z)
        return _series_2port(freq_obj, Z, Z0)
    else:
        raise ValueError(
            f"Unknown component type '{comp['type']}'. "
            f"Supported in-line types: R, L, C, RLC, TL, Z"
        )


def _load_impedance(comp: Dict[str, Any],
                    freq_array: np.ndarray,
                    Z0: float) -> np.ndarray:
    """
    Return Z_load(f) for the terminal element.
    TL cannot be a terminal load — it must have a load after it.
    """
    t = comp["type"].upper()

    if t == "R":
        return _z_resistor(freq_array, comp["value"])
    elif t == "L":
        return _z_inductor(freq_array, comp["value"])
    elif t == "C":
        return _z_capacitor(freq_array, comp["value"])
    elif t == "RLC":
        return _z_rlc(freq_array,
                      comp.get("R", 0.0),
                      comp.get("L", 1e-30),
                      comp.get("C", 1e-30))
    elif t == "Z":
        Z = np.asarray(comp["value"], dtype=complex)
        if Z.ndim == 0:
            Z = np.full(len(freq_array), Z)
        return Z
    elif t == "TL":
        raise ValueError(
            "'TL' cannot be the last (terminal) element — "
            "a transmission line must drive a load. "
            "Add an R/RLC load after it."
        )
    else:
        raise ValueError(
            f"Unknown terminal load type '{comp['type']}'. "
            f"Supported: R, L, C, RLC, Z"
        )


# =============================================================================
# Section 4 — Cascade and signal-flow-graph termination
# =============================================================================

def _cascade(networks: List["skrf.Network"]) -> Optional["skrf.Network"]:
    """Cascade 2-port Networks via skrf ** operator. Returns None if empty."""
    if not networks:
        return None
    result = networks[0]
    for net in networks[1:]:
        result = result ** net
    return result


def _sfg_terminate(cascaded: Optional["skrf.Network"],
                   Z_load: np.ndarray,
                   Z0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Signal-flow-graph load termination:

        Γ_in = S11 + (S12 · Γ_L · S21) / (1 - S22 · Γ_L)

    where  Γ_L = (Z_load - Z0) / (Z_load + Z0).

    cascaded = None  →  trivial through (Γ_in = Γ_L directly).

    Returns (gamma_in, Z_in).
    """
    denom_l = Z_load + Z0
    denom_l = np.where(np.abs(denom_l) < _EPS, _EPS + 0j, denom_l)
    gamma_L = (Z_load - Z0) / denom_l

    if cascaded is None:
        gamma_in = gamma_L
    else:
        S11 = cascaded.s[:, 0, 0]
        S12 = cascaded.s[:, 0, 1]
        S21 = cascaded.s[:, 1, 0]
        S22 = cascaded.s[:, 1, 1]

        denom_sfg = 1.0 - S22 * gamma_L
        denom_sfg = np.where(np.abs(denom_sfg) < _EPS, _EPS + 0j, denom_sfg)
        gamma_in  = S11 + (S12 * gamma_L * S21) / denom_sfg

    # Recover input impedance from Γ_in
    denom_z = 1.0 - gamma_in
    denom_z = np.where(np.abs(denom_z) < _EPS, _EPS + 0j, denom_z)
    Z_in    = Z0 * (1.0 + gamma_in) / denom_z

    return gamma_in, Z_in


# =============================================================================
# Section 5 — Manual (analytical) path for validation
# =============================================================================

def _manual_response(network_config: List[Dict[str, Any]],
                     freq_array: np.ndarray,
                     Z0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical mode: sum all impedances as a series chain.
    Identical to rf_engine.compute_impedance / compute_gamma for RLC-only nets.
    Raises ValueError for TL (not analytically supported).
    """
    Z_total = np.zeros(len(freq_array), dtype=complex)
    for comp in network_config:
        t = comp["type"].upper()
        if t == "R":
            Z_total += _z_resistor(freq_array, comp["value"])
        elif t == "L":
            Z_total += _z_inductor(freq_array, comp["value"])
        elif t == "C":
            Z_total += _z_capacitor(freq_array, comp["value"])
        elif t == "RLC":
            Z_total += _z_rlc(freq_array,
                               comp.get("R", 0.0),
                               comp.get("L", 1e-30),
                               comp.get("C", 1e-30))
        else:
            raise ValueError(
                f"Manual mode does not support component type '{comp['type']}'. "
                f"Use mode='skrf' for transmission lines and complex topologies."
            )
    denom = Z_total + Z0
    denom = np.where(np.abs(denom) < _EPS, _EPS + 0j, denom)
    gamma = (Z_total - Z0) / denom
    return gamma, Z_total


# =============================================================================
# Section 6 — Derived scalar quantities (mirrors rf_engine helpers exactly)
# =============================================================================

def _s11_db(gamma: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.clip(np.abs(gamma), _EPS, None))

def _vswr(gamma: np.ndarray) -> np.ndarray:
    mag = np.clip(np.abs(gamma), 0.0, 1.0 - _EPS)
    return (1.0 + mag) / (1.0 - mag)

def _phase_deg(gamma: np.ndarray) -> np.ndarray:
    return np.angle(gamma, deg=True)

def _phase_unwrapped(gamma: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.angle(gamma)))

def _return_loss(gamma: np.ndarray) -> np.ndarray:
    return -_s11_db(gamma)

def _group_delay_ns(freqs: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Non-uniform finite-difference group delay (ns). Mirrors rf_engine."""
    phase_rad = np.unwrap(np.angle(gamma))
    omega     = 2.0 * np.pi * freqs
    dphi      = np.gradient(phase_rad)
    domega    = np.gradient(omega)
    domega    = np.where(np.abs(domega) < _EPS, _EPS, domega)
    gd_ns     = -(dphi / domega) * 1e9
    q25, q75  = np.percentile(gd_ns, [25, 75])
    iqr        = max(q75 - q25, 1.0)
    return np.clip(gd_ns, q25 - 5.0 * iqr, q75 + 5.0 * iqr)

def _compute_bandwidth(freqs: np.ndarray,
                        s11_db: np.ndarray,
                        threshold_db: float = -10.0) -> dict:
    """Log-interpolated −10 dB bandwidth. Mirrors rf_engine._compute_bandwidth."""
    res_idx = int(np.argmin(s11_db))
    s11_min = float(s11_db[res_idx])
    f_res   = float(freqs[res_idx])
    result  = dict(f_res=f_res, s11_min=s11_min,
                   f_low=None, f_high=None, bandwidth=None, valid=False)

    if s11_min > threshold_db:
        return result

    f_low = None
    for i in range(res_idx, -1, -1):
        if s11_db[i] >= threshold_db:
            if i + 1 <= res_idx:
                t = (threshold_db - s11_db[i]) / (s11_db[i+1] - s11_db[i] + _EPS)
                f_low = np.exp(
                    np.log(freqs[i]) + t * (np.log(freqs[i+1]) - np.log(freqs[i])))
            else:
                f_low = float(freqs[i])
            break

    f_high = None
    for i in range(res_idx, len(freqs)):
        if s11_db[i] >= threshold_db:
            if i > 0:
                t = (threshold_db - s11_db[i-1]) / (s11_db[i] - s11_db[i-1] + _EPS)
                f_high = np.exp(
                    np.log(freqs[i-1]) + t * (np.log(freqs[i]) - np.log(freqs[i-1])))
            else:
                f_high = float(freqs[i])
            break

    if f_low is not None and f_high is not None:
        result.update(f_low=float(f_low), f_high=float(f_high),
                      bandwidth=float(f_high - f_low), valid=True)
    return result


# =============================================================================
# Section 7 — Result packer (output contract with gui/plot modules)
# =============================================================================

def _pack_result(freqs: np.ndarray,
                 gamma: np.ndarray,
                 Z_in:  np.ndarray,
                 Z0:    float,
                 mode:  str,
                 n_comp: int) -> dict:
    """
    Build the result dict that gui.py, plot_s11.py, smith_chart.py consume.
    Key names are the contract defined in rf_engine.run_simulation().
    """
    s11   = _s11_db(gamma)
    vswr  = _vswr(gamma)
    phase = _phase_deg(gamma)
    ph_uw = _phase_unwrapped(gamma)
    rl    = _return_loss(gamma)
    gd    = _group_delay_ns(freqs, gamma)
    bw    = _compute_bandwidth(freqs, s11)

    return dict(
        # ── Core physics arrays (exact keys expected by GUI/plots) ────────
        frequencies      = freqs,
        Z_L              = Z_in,        # "Z_L" kept for GUI compatibility
        gamma            = gamma,
        s11_db           = s11,
        vswr             = vswr,
        return_loss      = rl,
        phase_deg        = phase,
        phase_unwrapped  = ph_uw,
        group_delay_ns   = gd,
        z_real           = Z_in.real,
        z_imag           = Z_in.imag,

        # ── Scalar pass-throughs ──────────────────────────────────────────
        Z0               = Z0,
        sweep_type       = "log",
        if_bw_hz         = 1000.0,
        output_power_dbm = -10.0,

        # ── Sub-dicts ─────────────────────────────────────────────────────
        bandwidth        = bw,
        resonance_check  = dict(
            f_theoretical = float("nan"),  # no unique L/C for arbitrary nets
            f_simulated   = bw["f_res"],
            deviation_pct = float("nan"),
            outside_sweep = False,
            warning       = False,
            message       = "",
        ),

        # ── Extra metadata (not consumed by GUI) ──────────────────────────
        mode             = mode,
        n_components     = n_comp,
        has_skrf         = HAS_SKRF,
    )


# =============================================================================
# Section 8 — Main public API
# =============================================================================

def compute_network_response(
    network_config: List[Dict[str, Any]],
    freq_array: np.ndarray,
    Z0: float = 50.0,
    mode: str = "skrf",
) -> dict:
    """
    Compute the 1-port reflection response of a cascaded RF network.

    Parameters
    ----------
    network_config : list of component dicts.
                     Last element = terminal load.
                     All preceding = in-line 2-port elements.
    freq_array     : 1-D numpy array of frequencies (Hz).
    Z0             : reference impedance in Ω (default 50 Ω).
    mode           : "skrf"   — scikit-rf cascade (default)
                     "manual" — analytical summation, R/L/C/RLC only

    Returns
    -------
    dict — identical key set to rf_engine.run_simulation().

    Raises
    ------
    ValueError  — bad type, TL as terminal load, missing fields.
    ImportError — mode="skrf" but scikit-rf not installed.
    """
    freq_array = np.asarray(freq_array, dtype=float)
    n = len(network_config)

    # ── Empty network → open circuit ─────────────────────────────────────────
    if n == 0:
        gamma = np.ones(len(freq_array), dtype=complex)  # Γ = +1 (open)
        Z_in  = np.full(len(freq_array), 1e9 + 0j)       # 1 GΩ proxy
        return _pack_result(freq_array, gamma, Z_in, Z0, "skrf", 0)

    # ── Manual mode ───────────────────────────────────────────────────────────
    if mode == "manual":
        gamma, Z_in = _manual_response(network_config, freq_array, Z0)
        return _pack_result(freq_array, gamma, Z_in, Z0, "manual", n)

    # ── skrf mode ─────────────────────────────────────────────────────────────
    if not HAS_SKRF:
        raise ImportError(
            "scikit-rf is not installed.  Run:\n"
            "    pip install scikit-rf\n"
            "or use mode='manual' for R/L/C-only networks."
        )

    freq_obj = _make_freq_obj(freq_array)

    if n == 1:
        # Single element: it is the load, no in-line cascade
        Z_load = _load_impedance(network_config[0], freq_array, Z0)
        gamma, Z_in = _sfg_terminate(None, Z_load, Z0)
    else:
        # All but last are in-line 2-ports; last is the load
        inline_nets = [_build_inline_2port(c, freq_obj, Z0)
                       for c in network_config[:-1]]
        cascade = _cascade(inline_nets)
        Z_load  = _load_impedance(network_config[-1], freq_array, Z0)
        gamma, Z_in = _sfg_terminate(cascade, Z_load, Z0)

    return _pack_result(freq_array, gamma, Z_in, Z0, "skrf", n)


# =============================================================================
# Section 9 — Convenience shim: drop-in for rf_engine.run_simulation()
# =============================================================================

def run_simulation_network(
    network_config: List[Dict[str, Any]],
    f_start: float,
    f_stop: float,
    n_points: int = 401,
    Z0: float = 50.0,
    sweep_type: str = "log",
    mode: str = "skrf",
    **_ignored,
) -> dict:
    """
    Drop-in replacement for rf_engine.run_simulation() at the call site.

    One-line swap in gui.py:
        from network_engine import run_simulation_network as run_simulation

    The 'network_config' list replaces the R/L/C scalar arguments.
    """
    if sweep_type == "linear":
        freqs = np.linspace(f_start, f_stop, int(n_points))
    else:
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), int(n_points))

    result = compute_network_response(network_config, freqs, Z0=Z0, mode=mode)
    result["sweep_type"] = sweep_type
    return result


# =============================================================================
# Section 10 — Example preset configs
# =============================================================================

NETWORK_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "Half-wave Dipole (~321 MHz)": [
        {"type": "RLC", "R": 73.0, "L": 35e-9, "C": 7.0e-12},
    ],
    "Dipole via 50 cm coax (vf=0.66)": [
        {"type": "TL", "Z0": 50.0, "length": 0.5, "vf": 0.66},
        {"type": "RLC", "R": 73.0, "L": 35e-9, "C": 7.0e-12},
    ],
    "Loss + dipole (10 Ω series)": [
        {"type": "R",   "value": 10.0},
        {"type": "RLC", "R": 73.0, "L": 35e-9, "C": 7.0e-12},
    ],
    "λ/4 transformer → 100 Ω load (at 300 MHz)": [
        {"type": "TL",  "Z0": 70.71, "el_deg": 90.0, "el_ref_hz": 300e6},
        {"type": "R",   "value": 100.0},
    ],
    "Patch Antenna (~2.4 GHz)": [
        {"type": "RLC", "R": 100.0, "L": 3.3e-9, "C": 1.3e-12},
    ],
}


# =============================================================================
# Section 11 — Self-test / demo
# =============================================================================

def _cross_validate(verbose: bool = True) -> dict:
    """
    Compare skrf vs manual for a series RLC — should agree to < 1e-10.
    Returns error magnitudes for programmatic assertion.
    """
    R, L, C = 73.0, 35e-9, 7.0e-12
    config = [{"type": "RLC", "R": R, "L": L, "C": C}]
    freqs  = np.logspace(np.log10(50e6), np.log10(1e9), 501)

    r_skrf   = compute_network_response(config, freqs, mode="skrf")
    r_manual = compute_network_response(config, freqs, mode="manual")

    errs = {
        "s11_db" : float(np.max(np.abs(r_skrf["s11_db"]    - r_manual["s11_db"]))),
        "vswr"   : float(np.max(np.abs(r_skrf["vswr"]      - r_manual["vswr"]))),
        "phase"  : float(np.max(np.abs(r_skrf["phase_deg"] - r_manual["phase_deg"]))),
        "gamma"  : float(np.max(np.abs(r_skrf["gamma"]     - r_manual["gamma"]))),
    }

    if verbose:
        print("=" * 64)
        print("  Cross-validation: skrf vs manual (single RLC element)")
        print("=" * 64)
        print(f"  R={R} Ω  L={L*1e9:.0f} nH  C={C*1e12:.1f} pF  |  501 points")
        for k, v in errs.items():
            print(f"  |Δ {k:<8}| : {v:.2e}")
        f_theory = 1.0 / (2.0 * np.pi * np.sqrt(L * C))
        f_skrf   = r_skrf["bandwidth"]["f_res"]
        f_manual = r_manual["bandwidth"]["f_res"]
        print(f"\n  f_theory : {f_theory/1e6:.3f} MHz")
        print(f"  f_skrf   : {f_skrf/1e6:.3f} MHz")
        print(f"  f_manual : {f_manual/1e6:.3f} MHz\n")

    return errs


def _demo_cascaded():
    config = [
        {"type": "R",   "value": 10.0},
        {"type": "TL",  "Z0": 50.0, "length": 0.5, "vf": 0.66},
        {"type": "RLC", "R": 73.0, "L": 35e-9, "C": 7e-12},
    ]
    freqs  = np.logspace(np.log10(50e6), np.log10(1e9), 501)
    result = compute_network_response(config, freqs)

    print("=" * 64)
    print("  Cascaded: R(10Ω) → TL(50Ω, 50cm, vf=0.66) → RLC dipole")
    print("=" * 64)
    print(f"  {'Freq (MHz)':>12}  {'S11 (dB)':>10}  {'VSWR':>8}  {'|Γ|':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}")
    for i in np.linspace(0, len(freqs)-1, 10, dtype=int):
        f   = freqs[i] / 1e6
        s11 = result["s11_db"][i]
        v   = result["vswr"][i]
        gm  = np.abs(result["gamma"][i])
        print(f"  {f:12.1f}  {s11:10.2f}  {v:8.3f}  {gm:8.4f}")
    print()


def _demo_quarter_wave_transformer():
    """λ/4 transformer: Z_tl = √(Z0·Z_load) should give perfect match at f0."""
    f0, Z_load_val = 300e6, 100.0
    Z_tl = np.sqrt(50.0 * Z_load_val)   # = 70.71 Ω

    config_bare  = [{"type": "R", "value": Z_load_val}]
    config_xfmr  = [
        {"type": "TL", "Z0": Z_tl, "el_deg": 90.0, "el_ref_hz": f0},
        {"type": "R",  "value": Z_load_val},
    ]
    freqs = np.linspace(100e6, 500e6, 401)
    r_bare  = compute_network_response(config_bare,  freqs)
    r_xfmr  = compute_network_response(config_xfmr, freqs)
    idx     = int(np.argmin(np.abs(freqs - f0)))

    print("=" * 64)
    print(f"  λ/4 transformer  Z_tl={Z_tl:.2f} Ω  f0={f0/1e6:.0f} MHz")
    print("=" * 64)
    print(f"  At f0={f0/1e6:.0f} MHz:")
    print(f"    Bare {Z_load_val:.0f} Ω   : S11={r_bare['s11_db'][idx]:.2f} dB  "
          f"VSWR={r_bare['vswr'][idx]:.3f}")
    print(f"    With λ/4 xfmr : S11={r_xfmr['s11_db'][idx]:.2f} dB  "
          f"VSWR={r_xfmr['vswr'][idx]:.3f}")
    print(f"  (Ideal: S11 → −∞ dB, VSWR → 1.000)\n")


if __name__ == "__main__":
    print(f"\n  scikit-rf available: {HAS_SKRF}\n")

    errs = _cross_validate(verbose=True)
    _demo_cascaded()
    _demo_quarter_wave_transformer()

    TOL = 1e-6
    for key, val in errs.items():
        assert val < TOL, f"FAIL {key}: {val:.2e} > {TOL:.2e}"
    print("  All cross-validation assertions PASSED.\n")