"""
quantum_network.py  —  Quantum-specific sweep wrapper around network_engine
===========================================================================

Wraps network_engine.compute_network_response() with quantum-physics
post-processing: loaded Q, T₁ estimate, dispersive shift χ, and dispersive
regime detection.

Public API
----------
run_quantum_sweep(config_name, f_start, f_stop, n_points, Z0,
                  network_prefix=None) -> dict
compute_q_loaded(freqs, s11_db) -> float
compute_t1_estimate(q_loaded, f_res) -> float
compute_dispersive_shift(f_qubit_hz, f_readout_hz, g_hz, alpha_hz) -> float

The returned dict is a strict superset of the standard result dict
(same keys as rf_engine.run_simulation()), plus extra quantum keys:
    q_loaded        float
    q_internal_est  float
    t1_us           float
    chi_mhz         float
    dispersive_regime bool
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from network_engine  import compute_network_response
from quantum_models  import QUBIT_SYSTEM_CONFIGS, QUBIT_SYSTEM_FREQ_HINTS

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Q-factor and resonance helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_q_loaded(freqs: np.ndarray, s11_db: np.ndarray,
                     threshold_db: float = -3.0) -> float:
    """
    Loaded Q from the −3 dB bandwidth of the S11 resonance dip.

        Q_loaded = f_res / BW_3dB

    Parameters
    ----------
    freqs       : frequency array in Hz
    s11_db      : S11 in dB (same length as freqs)
    threshold_db: crossing level in dB (default −3 dB from resonance floor)

    Returns
    -------
    Loaded Q (float).  Returns 0.0 if resonance not found or BW = 0.
    """
    res_idx = int(np.argmin(s11_db))
    s11_min = float(s11_db[res_idx])
    f_res   = float(freqs[res_idx])

    # Absolute threshold relative to the resonance floor
    abs_threshold = s11_min + abs(threshold_db)

    # Walk left for f_low
    f_low = None
    for i in range(res_idx, -1, -1):
        if s11_db[i] >= abs_threshold:
            if i + 1 <= res_idx:
                t = ((abs_threshold - s11_db[i])
                     / (s11_db[i+1] - s11_db[i] + _EPS))
                f_low = np.exp(
                    np.log(max(freqs[i], _EPS))
                    + t * (np.log(max(freqs[i+1], _EPS))
                           - np.log(max(freqs[i], _EPS))))
            else:
                f_low = float(freqs[i])
            break

    # Walk right for f_high
    f_high = None
    for i in range(res_idx, len(freqs)):
        if s11_db[i] >= abs_threshold:
            if i > 0:
                t = ((abs_threshold - s11_db[i-1])
                     / (s11_db[i] - s11_db[i-1] + _EPS))
                f_high = np.exp(
                    np.log(max(freqs[i-1], _EPS))
                    + t * (np.log(max(freqs[i], _EPS))
                           - np.log(max(freqs[i-1], _EPS))))
            else:
                f_high = float(freqs[i])
            break

    if f_low is None or f_high is None:
        return 0.0
    bw = max(f_high - f_low, _EPS)
    return float(f_res / bw)


def compute_t1_estimate(q_loaded: float, f_res_hz: float) -> float:
    """
    Estimate T₁ relaxation time from loaded Q and resonance frequency.

        T₁ = Q_loaded / (π · f_res)   [seconds]

    Returns T₁ in **microseconds**.

    Derivation: the energy relaxation rate κ = ω_res / Q_loaded,
    and T₁ = 1/κ = Q / (2π f · 1/2π) = Q / (π f).
    """
    f_res_hz = max(abs(f_res_hz), _EPS)
    q_loaded = max(q_loaded, _EPS)
    t1_s = q_loaded / (np.pi * f_res_hz)
    return float(t1_s * 1e6)   # microseconds


def compute_dispersive_shift(f_qubit_hz: float, f_readout_hz: float,
                             g_hz: float,
                             alpha_hz: float = -200e6) -> float:
    """
    Dispersive approximation for the qubit-readout coupling shift χ.

    In the dispersive regime (|Δ| = |f_r − f_q| >> g):

        χ = g² · α / (Δ · (Δ + α))

    where
        Δ = f_readout − f_qubit        (detuning)
        α = anharmonicity of the qubit  (negative for transmon, ~−200 MHz)
        g = coupling strength

    Returns χ in **MHz**.

    Reference: Koch et al. PRA 76, 042319 (2007), eq. (4.10).
    """
    delta = f_readout_hz - f_qubit_hz
    if abs(delta) < _EPS:
        return 0.0
    chi_hz = (g_hz**2 * alpha_hz) / (delta * (delta + alpha_hz) + _EPS)
    return float(chi_hz / 1e6)   # MHz


def _is_dispersive(f_qubit_hz: float, f_readout_hz: float,
                   g_hz: float) -> bool:
    """
    Check dispersive condition: |Δ| >> g  (use factor-of-10 criterion).
    """
    delta = abs(f_readout_hz - f_qubit_hz)
    return delta > 10.0 * abs(g_hz)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Internal Q estimate
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_q_internal(q_loaded: float,
                         q_coupling: float = 2000.0) -> float:
    """
    Estimate internal (unloaded) Q from loaded Q and coupling Q.

        1/Q_loaded = 1/Q_internal + 1/Q_coupling
        → Q_internal = Q_loaded · Q_coupling / (Q_coupling - Q_loaded)

    For over-coupled resonators Q_loaded < Q_coupling.
    Falls back to 10 × Q_loaded when Q_coupling <= Q_loaded.
    """
    if q_coupling <= q_loaded:
        return q_loaded * 10.0
    return (q_loaded * q_coupling) / max(q_coupling - q_loaded, _EPS)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Main sweep function
# ─────────────────────────────────────────────────────────────────────────────

def run_quantum_sweep(
    config_name:    str,
    f_start:        float,
    f_stop:         float,
    n_points:       int   = 500,
    Z0:             float = 50.0,
    network_prefix: Optional[list] = None,
    g_hz:           float = 100e6,
    alpha_hz:       float = -200e6,
    mode:           str   = "skrf",
) -> dict:
    """
    Run a quantum network sweep and return quantum parameters alongside
    the standard VNA result dict.

    Parameters
    ----------
    config_name     : key in QUBIT_SYSTEM_CONFIGS
    f_start, f_stop : sweep range in Hz
    n_points        : number of frequency points
    Z0              : reference impedance (Ω)
    network_prefix  : optional list of component dicts (e.g. cryo chain)
                      prepended to the system config before the sweep
    g_hz            : qubit–readout coupling strength in Hz (default 100 MHz)
    alpha_hz        : qubit anharmonicity in Hz (default −200 MHz for transmon)
    mode            : "skrf" or "manual" (passed to network_engine)

    Returns
    -------
    Standard result dict (all keys from network_engine / rf_engine), plus:
        q_loaded        float  — loaded Q from −3 dB bandwidth
        q_internal_est  float  — estimated internal (unloaded) Q
        t1_us           float  — T₁ estimate in µs
        chi_mhz         float  — dispersive shift χ in MHz
        dispersive_regime bool  — True when |Δ| > 10g
    """
    if config_name not in QUBIT_SYSTEM_CONFIGS:
        raise KeyError(
            f"Unknown quantum config '{config_name}'. "
            f"Available: {list(QUBIT_SYSTEM_CONFIGS.keys())}"
        )

    base_config = list(QUBIT_SYSTEM_CONFIGS[config_name])
    if network_prefix:
        network_config = list(network_prefix) + base_config
    else:
        network_config = base_config

    freqs = _make_freq_array(f_start, f_stop, n_points)

    result = compute_network_response(
        network_config = network_config,
        freq_array     = freqs,
        Z0             = Z0,
        mode           = mode,
    )

    # ── Quantum post-processing ───────────────────────────────────────────────
    s11_db = result["s11_db"]

    q_loaded  = compute_q_loaded(freqs, s11_db, threshold_db=-3.0)
    q_int_est = _estimate_q_internal(q_loaded)

    res_idx   = int(np.argmin(s11_db))
    f_res_hz  = float(freqs[res_idx])

    t1_us     = compute_t1_estimate(q_loaded, f_res_hz)

    # Infer qubit and readout frequencies from the config
    f_qubit_hz, f_readout_hz = _guess_qubit_readout_freqs(
        config_name, base_config, f_res_hz)

    chi_mhz   = compute_dispersive_shift(
        f_qubit_hz, f_readout_hz, g_hz, alpha_hz)
    disp      = _is_dispersive(f_qubit_hz, f_readout_hz, g_hz)

    result.update(
        q_loaded        = q_loaded,
        q_internal_est  = q_int_est,
        t1_us           = t1_us,
        chi_mhz         = chi_mhz,
        dispersive_regime = disp,
        f_qubit_hz      = f_qubit_hz,
        f_readout_hz    = f_readout_hz,
        g_hz            = g_hz,
        alpha_hz        = alpha_hz,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_freq_array(f_start: float, f_stop: float,
                     n_points: int) -> np.ndarray:
    """Logarithmically-spaced frequency array, matching rf_engine default."""
    return np.logspace(np.log10(max(f_start, _EPS)),
                       np.log10(max(f_stop, _EPS)),
                       int(n_points))


def _guess_qubit_readout_freqs(config_name: str,
                                config: list,
                                f_fallback: float) -> tuple:
    """
    Heuristically extract qubit and readout frequencies from a config list
    for dispersive-shift calculation.

    Strategy: collect resonance frequencies of all RLC elements, sorted
    ascending; assign lowest = qubit, next = readout.
    """
    rlc_freqs = []
    for comp in config:
        if comp.get("type") == "RLC":
            R = comp.get("R", 0.0)
            L = comp.get("L", 1e-9)
            C = comp.get("C", 1e-12)
            try:
                f = 1.0 / (2.0 * np.pi * np.sqrt(max(L * C, _EPS)))
                rlc_freqs.append(f)
            except Exception:
                pass

    rlc_freqs.sort()

    if len(rlc_freqs) >= 2:
        return rlc_freqs[0], rlc_freqs[1]
    elif len(rlc_freqs) == 1:
        return rlc_freqs[0], rlc_freqs[0] * 1.3   # assume readout 30% higher
    else:
        return f_fallback, f_fallback * 1.3


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — standalone unit test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== quantum_network standalone test ===\n")

    configs = list(QUBIT_SYSTEM_CONFIGS.keys())
    for cfg_name in configs:
        hints = QUBIT_SYSTEM_FREQ_HINTS.get(cfg_name, (4e9, 8e9))
        try:
            res = run_quantum_sweep(
                config_name = cfg_name,
                f_start     = hints[0],
                f_stop      = hints[1],
                n_points    = 300,
                Z0          = 50.0,
                g_hz        = 100e6,
                alpha_hz    = -200e6,
            )
            print(f"Config: {cfg_name}")
            print(f"  f_res    = {float(res['frequencies'][int(np.argmin(res['s11_db']))])/1e9:.4f} GHz")
            print(f"  Q_loaded = {res['q_loaded']:.2f}")
            print(f"  Q_int_est= {res['q_internal_est']:.2f}")
            print(f"  T1_est   = {res['t1_us']:.3f} µs")
            print(f"  χ        = {res['chi_mhz']:.4f} MHz")
            print(f"  Dispersive regime: {res['dispersive_regime']}")
            print()
        except Exception as e:
            print(f"  ERROR for '{cfg_name}': {e}")
            print()

    # Standalone Q/T1/chi functions
    print("--- compute_q_loaded (synthetic Lorentzian) ---")
    f = np.linspace(4.5e9, 5.5e9, 1000)
    f0 = 5.0e9; bw = 5e6
    s11 = -20.0 / (1.0 + ((f - f0) / (bw/2))**2)
    Q = compute_q_loaded(f, s11)
    print(f"  Expected Q ~ {f0/bw:.0f}  got {Q:.1f}")

    print("--- compute_t1_estimate ---")
    t1 = compute_t1_estimate(1e6, 5e9)
    print(f"  Q=1e6, f=5 GHz → T1 = {t1:.3f} µs  (expected ~63.7 µs)")

    print("--- compute_dispersive_shift ---")
    chi = compute_dispersive_shift(5e9, 6.5e9, 100e6, -200e6)
    print(f"  g=100 MHz, Δ=1.5 GHz, α=−200 MHz → χ = {chi:.4f} MHz")
