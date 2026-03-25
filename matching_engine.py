"""
matching_engine.py  —  Impedance Matching Network Solver
=========================================================

Supported topologies (L and Pi only):
    "L_lo_pass"   — series L  + shunt C  (low-pass)
    "L_hi_pass"   — series C  + shunt L  (high-pass)
    "L_auto"      — pick whichever L-network has lower Q
    "pi_lo_pass"  — shunt C1 — series L — shunt C2
    "pi_hi_pass"  — shunt L1 — series C — shunt L2

Each component dict carries a "_shunt" flag so the GUI schematic
knows whether to place it as a series inline or a shunt-to-ground
element.  The sweep engine (sweep_matched_network) uses the proper
shunt 2-port model so S11 results are physically correct.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

_EPS = 1e-30


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class MatchResult:
    topology      : str
    components    : List[Dict[str, Any]]   # component dicts for schematic + sweep
    element_values: Dict[str, str]         # human-readable: {"L_series": "35 nH", ...}
    Q             : float
    f0            : float
    Z_S           : complex
    Z_L           : complex
    Z0            : float
    bandwidth_hz  : float
    notes         : str  = ""
    valid         : bool = True
    error         : str  = ""
    # Optional: RLC params of terminal load for frequency-varying Z_L(f) in sweeps
    _rlc_R        : float = None
    _rlc_L        : float = None
    _rlc_C        : float = None


# =============================================================================
# SI helpers
# =============================================================================

def _si_L(v: float) -> str:
    if v >= 1e-3:  return f"{v*1e3:.4g} mH"
    if v >= 1e-6:  return f"{v*1e6:.4g} µH"
    return f"{v*1e9:.4g} nH"

def _si_C(v: float) -> str:
    if v >= 1e-9:  return f"{v*1e9:.4g} nF"
    return f"{v*1e12:.4g} pF"

def _si_Z(z: complex) -> str:
    s = "+" if z.imag >= 0 else ""
    return f"{z.real:.4g}{s}{z.imag:.4g}j Ω"

def _si_f(f: float) -> str:
    if f >= 1e9:  return f"{f/1e9:.4g} GHz"
    if f >= 1e6:  return f"{f/1e6:.4g} MHz"
    if f >= 1e3:  return f"{f/1e3:.4g} kHz"
    return f"{f:.4g} Hz"


# =============================================================================
# Section 1 — L-network  (Pozar §5.1)
# =============================================================================

def _l_network(Z_S: complex, Z_L: complex, f0: float,
               hi_pass: bool = False, Z0: float = 50.0) -> MatchResult:
    """
    L-network matching Z_S → Z_L at f0.

    Convention: shunt element is on the HIGH-resistance side.
    When R_S > R_L: source side gets shunt, load side gets series.
    When R_L > R_S: flip — source side gets series, load side gets shunt.

    Reactances in Z_S and Z_L are absorbed into the element values.
    """
    omega = 2 * math.pi * f0
    RS, XS = Z_S.real, Z_S.imag
    RL, XL = Z_L.real, Z_L.imag

    if RS <= 0 or RL <= 0:
        return MatchResult(
            topology="L_network", components=[], element_values={},
            Q=0, f0=f0, Z_S=Z_S, Z_L=Z_L, Z0=Z0, bandwidth_hz=0,
            valid=False,
            error="R_S and R_L must be > 0 for L-network design.")

    # Shunt always goes on the high-R side
    flip = RS < RL
    if flip:
        RS, RL, XS, XL = RL, RS, XL, XS

    Q = math.sqrt(RS / RL - 1.0)

    if not hi_pass:
        B_shunt  =  Q / RS   # positive → capacitive
        X_series =  Q * RL   # positive → inductive
    else:
        B_shunt  = -Q / RS   # negative → inductive
        X_series = -Q * RL   # negative → capacitive

    # Absorb existing source/load reactances
    B_shunt_net  = B_shunt - (-XS / (RS**2 + XS**2))
    X_series_net = X_series - XL

    # Convert to L/C
    if X_series_net >= 0:
        L_s = X_series_net / omega;  C_s = None
    else:
        L_s = None;  C_s = -1.0 / (omega * X_series_net)

    if B_shunt_net >= 0:
        C_p = B_shunt_net / omega;  L_p = None
    else:
        L_p = -1.0 / (omega * B_shunt_net);  C_p = None

    # Build schematic component list
    # _shunt=True  → shunt-to-ground (vertical) element in schematic
    # _shunt=False → series (inline) element in schematic
    comps: List[Dict] = []
    ev: Dict[str, str] = {}

    def _series_comp():
        if L_s is not None:
            comps.append({"type": "L", "value": L_s, "_shunt": False})
            ev["L_series"] = _si_L(L_s)
        else:
            comps.append({"type": "C", "value": C_s, "_shunt": False})
            ev["C_series"] = _si_C(C_s)

    def _shunt_comp():
        if C_p is not None:
            comps.append({"type": "C", "value": C_p, "_shunt": True})
            ev["C_shunt"] = _si_C(C_p)
        else:
            comps.append({"type": "L", "value": L_p, "_shunt": True})
            ev["L_shunt"] = _si_L(L_p)

    if not flip:
        # High-R source: shunt at source end, series toward load
        _shunt_comp(); _series_comp()
    else:
        # High-R load: series first, shunt at load end
        _series_comp(); _shunt_comp()

    topo = ("L_hi_pass" if hi_pass else "L_lo_pass") + ("_flipped" if flip else "")
    bw   = f0 / Q if Q > 0 else float("inf")

    return MatchResult(
        topology=topo, components=comps, element_values=ev,
        Q=Q, f0=f0, Z_S=Z_S, Z_L=Z_L, Z0=Z0,  # Z0 is system ref, not RL
        bandwidth_hz=bw,
        notes=(f"L-network  ({'high-pass' if hi_pass else 'low-pass'})\n"
               f"Q = {Q:.3f}     BW ≈ {_si_f(bw)}\n"
               f"Z_S = {_si_Z(Z_S)}\nZ_L = {_si_Z(Z_L)}"),
    )


# =============================================================================
# Section 2 — Pi-network  (Pozar §5.2)
# =============================================================================

def _pi_network(Z_S: complex, Z_L: complex, f0: float,
                Q_target: float = 5.0, hi_pass: bool = False,
                Z0: float = 50.0) -> MatchResult:
    """
    Pi-network with user-specified Q.

    Low-pass  : shunt C1 ─── series L ─── shunt C2
    High-pass : shunt L1 ─── series C ─── shunt L2

    Q_min = sqrt(R_max/R_min - 1); Q must exceed this.
    Virtual resistance R_virt = R_max / (Q² + 1).
    """
    omega = 2 * math.pi * f0
    RS = Z_S.real
    RL = Z_L.real

    if RS <= 0 or RL <= 0:
        return MatchResult(
            topology="pi", components=[], element_values={},
            Q=0, f0=f0, Z_S=Z_S, Z_L=Z_L, Z0=Z0, bandwidth_hz=0,
            valid=False, error="R_S and R_L must be > 0.")

    Q_min = math.sqrt(max(RS, RL) / min(RS, RL) - 1.0)
    Q = max(Q_target, Q_min + 0.01)

    R_virt = max(RS, RL) / (Q**2 + 1.0)
    Q1 = math.sqrt(RS / R_virt - 1.0)
    Q2 = math.sqrt(RL / R_virt - 1.0)

    if not hi_pass:
        L_s  = (Q1 * R_virt) / omega
        C1   = (Q1 / RS)     / omega
        C2   = (Q2 / RL)     / omega
        ev   = {"C1_shunt": _si_C(C1), "L_series": _si_L(L_s),
                "C2_shunt": _si_C(C2)}
        comps = [
            {"type": "C", "value": C1,  "_shunt": True},
            {"type": "L", "value": L_s, "_shunt": False},
            {"type": "C", "value": C2,  "_shunt": True},
        ]
        topo = "pi_lo_pass"
    else:
        C_s = 1.0 / (Q1 * R_virt * omega)
        L1  = RS  / (Q1 * omega)
        L2  = RL  / (Q2 * omega)
        ev  = {"L1_shunt": _si_L(L1), "C_series": _si_C(C_s),
               "L2_shunt": _si_L(L2)}
        comps = [
            {"type": "L", "value": L1,  "_shunt": True},
            {"type": "C", "value": C_s, "_shunt": False},
            {"type": "L", "value": L2,  "_shunt": True},
        ]
        topo = "pi_hi_pass"

    bw = f0 / Q
    return MatchResult(
        topology=topo, components=comps, element_values=ev,
        Q=Q, f0=f0, Z_S=Z_S, Z_L=Z_L, Z0=Z0,  # Z0 is system ref
        bandwidth_hz=bw,
        notes=(f"Pi-network  ({'high-pass' if hi_pass else 'low-pass'})\n"
               f"Q = {Q:.3f}  (Q_min = {Q_min:.3f})   BW ≈ {_si_f(bw)}\n"
               f"R_virtual = {R_virt:.4g} Ω"),
    )


# =============================================================================
# Section 3 — Public dispatcher
# =============================================================================

def solve_matching(
    Z_S: complex,
    Z_L: complex,
    f0: float,
    topology: str = "L_auto",
    Z0: float = 50.0,
    Q_target: float = 5.0,
    rlc_R: float = None,
    rlc_L: float = None,
    rlc_C: float = None,
) -> MatchResult:
    """
    Compute an L or Pi matching network between Z_S and Z_L at f0.

    topology : "L_auto" | "L_lo_pass" | "L_hi_pass"
             | "pi_lo_pass" | "pi_hi_pass"
    Q_target : desired Q for Pi networks (must exceed Q_min)
    rlc_R/L/C: series-RLC parameters of the load antenna.  When supplied,
               sweep_matched_network uses Z_L(f) across the whole sweep
               (physically correct) instead of the scalar design-point Z_L.
    """
    t = topology.lower()
    if t == "l_lo_pass":
        mr = _l_network(Z_S, Z_L, f0, hi_pass=False, Z0=Z0)
    elif t == "l_hi_pass":
        mr = _l_network(Z_S, Z_L, f0, hi_pass=True, Z0=Z0)
    elif t == "l_auto":
        lp = _l_network(Z_S, Z_L, f0, hi_pass=False, Z0=Z0)
        hp = _l_network(Z_S, Z_L, f0, hi_pass=True, Z0=Z0)
        if not lp.valid:   mr = hp
        elif not hp.valid: mr = lp
        else:              mr = lp if lp.Q <= hp.Q else hp
    elif t == "pi_lo_pass":
        mr = _pi_network(Z_S, Z_L, f0, Q_target=Q_target, hi_pass=False, Z0=Z0)
    elif t == "pi_hi_pass":
        mr = _pi_network(Z_S, Z_L, f0, Q_target=Q_target, hi_pass=True, Z0=Z0)
    else:
        mr = MatchResult(
            topology=topology, components=[], element_values={},
            Q=0, f0=f0, Z_S=Z_S, Z_L=Z_L, Z0=Z0, bandwidth_hz=0,
            valid=False,
            error=f"Unknown topology '{topology}'. "
                  f"Use: L_auto, L_lo_pass, L_hi_pass, pi_lo_pass, pi_hi_pass")
    # Store RLC params so sweep_matched_network can build Z_L(f) array
    mr._rlc_R = rlc_R
    mr._rlc_L = rlc_L
    mr._rlc_C = rlc_C
    return mr


# =============================================================================
# Section 4 — Sweep the matched network with proper shunt 2-port modelling
# =============================================================================

def _shunt_2port_s(Z_shunt: np.ndarray, Z0: float) -> np.ndarray:
    """
    S-parameters of a shunt impedance (Pozar §4.4):
        S11 = S22 = -Z0 / (2·Z_shunt + Z0)
        S12 = S21 = 2·Z_shunt / (2·Z_shunt + Z0)
    Returns shape (N, 2, 2).
    """
    n = len(Z_shunt)
    s = np.zeros((n, 2, 2), dtype=complex)
    denom = 2.0 * Z_shunt + Z0
    denom = np.where(np.abs(denom) < _EPS, _EPS + 0j, denom)
    s[:, 0, 0] = -Z0 / denom
    s[:, 1, 1] = s[:, 0, 0]
    s[:, 0, 1] = 2.0 * Z_shunt / denom
    s[:, 1, 0] = s[:, 0, 1]
    return s


def _series_2port_s(Z_series: np.ndarray, Z0: float) -> np.ndarray:
    """
    S-parameters of a series impedance:
        S11 = S22 = Z / (Z + 2·Z0)
        S12 = S21 = 2·Z0 / (Z + 2·Z0)
    """
    n = len(Z_series)
    s = np.zeros((n, 2, 2), dtype=complex)
    denom = Z_series + 2.0 * Z0
    denom = np.where(np.abs(denom) < _EPS, _EPS + 0j, denom)
    s[:, 0, 0] = Z_series / denom
    s[:, 1, 1] = s[:, 0, 0]
    s[:, 0, 1] = 2.0 * Z0 / denom
    s[:, 1, 0] = s[:, 0, 1]
    return s


def _cascade_s(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Cascade two 2-port S-matrices (shape N×2×2) using the wave-cascade formula."""
    n = s1.shape[0]
    out = np.zeros_like(s1)
    for i in range(n):
        a, b = s1[i], s2[i]
        denom = 1.0 - b[0, 0] * a[1, 1]
        if abs(denom) < _EPS:
            denom = _EPS
        t = a[0, 1] * b[1, 0] / denom
        out[i, 0, 0] = a[0, 0] + a[1, 1] * b[0, 0] * t / a[0, 1] * a[0, 1] if False else \
                       a[0, 0] + t * a[1, 0] / a[0, 1] * a[0, 1] if False else \
                       a[0, 0] + (a[0, 1] * b[0, 0] * a[1, 0]) / denom
        out[i, 0, 1] = (a[0, 1] * b[0, 1]) / denom
        out[i, 1, 0] = (b[1, 0] * a[1, 0]) / denom
        out[i, 1, 1] = b[1, 1] + (b[1, 0] * a[1, 1] * b[0, 1]) / denom
    return out


def _comp_impedance(comp: Dict, freqs: np.ndarray) -> np.ndarray:
    """Ideal (lossless) impedance of a single L or C component."""
    omega = 2.0 * np.pi * freqs
    omega_safe = np.where(np.abs(omega) < _EPS, _EPS, omega)
    t = comp["type"].upper()
    v = comp["value"]
    if t == "L":
        return 1j * omega * v
    elif t == "C":
        return 1.0 / (1j * omega_safe * v)
    return np.full(len(freqs), complex(v))


def _comp_impedance_lossy(comp: Dict, freqs: np.ndarray,
                          Q_L: float = 50.0, Q_C: float = 200.0) -> np.ndarray:
    """
    Physically realistic impedance with finite component Q (series ESR).

    Inductor: Z = jωL + ωL/Q_L        (ESR increases with frequency)
    Capacitor: Z = 1/(jωC) + 1/(ωCQ_C) (ESR decreases with frequency)

    Typical values: Q_L=50 (mid-grade RF choke), Q_C=200 (NP0/C0G ceramic).
    These losses prevent the ideal perfect-cancellation null (~-80 dB) and
    give physically realistic S11_min of -20 to -35 dB.
    """
    omega = 2.0 * np.pi * freqs
    omega_safe = np.where(np.abs(omega) < _EPS, _EPS, omega)
    t = comp["type"].upper()
    v = comp["value"]
    if t == "L":
        return 1j * omega * v + omega_safe * v / Q_L
    elif t == "C":
        return 1.0 / (1j * omega_safe * v) + 1.0 / (omega_safe * v * Q_C)
    return np.full(len(freqs), complex(v))


def sweep_matched_network(
    match: MatchResult,
    f_start: float,
    f_stop: float,
    n: int = 501,
    sweep_type: str = "log",
    Q_L: float = 50.0,
    Q_C: float = 200.0,
    freqs_override: np.ndarray = None,
) -> dict:
    """
    Compute the S11 response of the complete matched network:

        Source (Z0) ── [matching network] ── Load (Z_L)

    Physical correctness requirements enforced here:

    1. Z0 consistency: match.Z0 is the system reference (50 Ω), used both
       to normalise the cascade S-matrices AND in the SFG termination gL.
       Previously match.Z0 was set to RL (10 Ω for a 50→10 Ω L-match),
       causing the two sides of the SFG to use different reference planes —
       giving unrealistically deep S11 nulls (~-80 dB).

    2. Shunt 2-port: shunt elements use _shunt_2port_s(), series use
       _series_2port_s(). The flag comp["_shunt"] is set by matching_engine.

    3. Component Q loss: _comp_impedance_lossy() adds series ESR to each
       L and C, giving S11_min of -20 to -35 dB (physically realistic for
       RF-grade components) instead of the -80 dB ideal-cancellation null.

    Parameters
    ----------
    Q_L           : inductor Q factor (default 50)
    Q_C           : capacitor Q factor (default 200)
    freqs_override: if supplied, use this exact frequency array instead of
                    generating one from f_start/f_stop/n/sweep_type.  Pass
                    result["frequencies"] from a VNA sweep so both tabs
                    compute S11 on identical frequency points.
    """
    if freqs_override is not None:
        freqs = np.asarray(freqs_override, dtype=float)
    elif sweep_type == "log":
        freqs = np.logspace(np.log10(max(f_start, 1.0)), np.log10(f_stop), n)
    else:
        freqs = np.linspace(f_start, f_stop, n)

    # System reference impedance — must be consistent throughout
    Z0 = float(match.Z0) if match.Z0 > 0 else 50.0

    # Build cascade S-matrix — identity initialisation
    cascade = np.zeros((len(freqs), 2, 2), dtype=complex)
    cascade[:, 0, 1] = 1.0
    cascade[:, 1, 0] = 1.0

    for comp in match.components:
        # Lossy model for physical realism
        Z_comp = _comp_impedance_lossy(comp, freqs, Q_L=Q_L, Q_C=Q_C)
        if comp.get("_shunt", False):
            s_new = _shunt_2port_s(Z_comp, Z0)
        else:
            s_new = _series_2port_s(Z_comp, Z0)
        cascade = _cascade_s(cascade, s_new)

    # SFG termination: use frequency-varying Z_L(f) so the load
    # impedance is correctly dispersed across the sweep.
    # If the load is a series RLC (most common: antenna), compute Z_L(f)
    # from the stored R/L/C parameters.  Otherwise fall back to the scalar
    # Z_L used at the design frequency — still physically better than
    # a frequency-flat assumption when the RLC params are unavailable.
    rlc_R = getattr(match, "_rlc_R", None)
    rlc_L = getattr(match, "_rlc_L", None)
    rlc_C = getattr(match, "_rlc_C", None)
    if rlc_R is not None and rlc_L is not None and rlc_C is not None:
        omega_f = 2.0 * np.pi * freqs
        omega_f = np.where(np.abs(omega_f) < _EPS, _EPS, omega_f)
        ZL_arr = rlc_R + 1j * (omega_f * rlc_L - 1.0 / (omega_f * rlc_C))
    else:
        ZL_arr = np.full(len(freqs), match.Z_L, dtype=complex)

    dL  = ZL_arr + Z0
    dL  = np.where(np.abs(dL) < _EPS, _EPS + 0j, dL)
    gL  = (ZL_arr - Z0) / dL

    S11  = cascade[:, 0, 0]
    S12  = cascade[:, 0, 1]
    S21  = cascade[:, 1, 0]
    S22  = cascade[:, 1, 1]

    denom = 1.0 - S22 * gL
    denom = np.where(np.abs(denom) < _EPS, _EPS + 0j, denom)
    gamma_in = S11 + (S12 * gL * S21) / denom

    # Recover Z_in
    dz = 1.0 - gamma_in
    dz = np.where(np.abs(dz) < 1e-9, 1e-9 + 0j, dz)
    Z_in = Z0 * (1.0 + gamma_in) / dz

    # Physical gamma floor — real RF components cannot achieve better than
    # ~−60 dB S11 due to parasitic losses, tolerance, and measurement noise.
    # Clamp |Γ| ≥ 1e-3 so the preview never shows unrealistic nulls.
    _GAMMA_FLOOR = 1e-3
    gamma_mag = np.abs(gamma_in)
    too_small = gamma_mag < _GAMMA_FLOOR
    if np.any(too_small):
        safe_mag = np.where(gamma_mag < _EPS, _EPS, gamma_mag)
        gamma_in = np.where(too_small,
                            gamma_in * (_GAMMA_FLOOR / safe_mag),
                            gamma_in)

    # Pack into result dict (same keys as network_engine / rf_engine)
    _eps2 = 1e-6          # floor for log10: -120 dB → hard cap at -60 dB
    s11_db  = 20.0 * np.log10(np.clip(np.abs(gamma_in), _eps2, None))
    gm      = np.clip(np.abs(gamma_in), 0.0, 1.0 - _eps2)
    vswr    = (1.0 + gm) / (1.0 - gm)
    rl      = -s11_db
    phase   = np.angle(gamma_in, deg=True)
    ph_uw   = np.degrees(np.unwrap(np.angle(gamma_in)))
    omega_a = 2.0 * np.pi * freqs
    gd      = -np.gradient(np.unwrap(np.angle(gamma_in)), omega_a) * 1e9
    q25, q75 = np.percentile(gd, [25, 75])
    gd      = np.clip(gd, q25 - 5*(q75-q25), q75 + 5*(q75-q25))

    res_idx = int(np.argmin(s11_db))
    f_res   = float(freqs[res_idx])

    return dict(
        frequencies     = freqs,
        gamma           = gamma_in,
        Z_L             = Z_in,
        Z_load_bare     = Z_in,
        s11_db          = s11_db,
        vswr            = vswr,
        return_loss     = rl,
        phase_deg       = phase,
        phase_unwrapped = ph_uw,
        group_delay_ns  = gd,
        z_real          = Z_in.real,
        z_imag          = Z_in.imag,
        Z0              = Z0,
        sweep_type      = sweep_type,
        if_bw_hz        = 1000.0,
        output_power_dbm= -10.0,
        bandwidth       = dict(
            f_res=f_res, s11_min=float(s11_db[res_idx]),
            f_low=None, f_high=None, bandwidth=None, valid=False,
        ),
        resonance_check = dict(
            f_theoretical=float("nan"), f_simulated=f_res,
            deviation_pct=float("nan"), outside_sweep=False,
            warning=False, message="",
        ),
        match           = match,
        Q_L             = Q_L,
        Q_C             = Q_C,
    )


# =============================================================================
# Section 5 — Compare both L topologies and both Pi topologies
# =============================================================================

def compare_l_and_pi(
    Z_S: complex,
    Z_L: complex,
    f0: float,
    Z0: float = 50.0,
    Q_target: float = 5.0,
    rlc_R: float = None,
    rlc_L: float = None,
    rlc_C: float = None,
) -> List[MatchResult]:
    """
    Return results for all four L+Pi topologies, sorted by Q.
    Invalid results are excluded.
    """
    results = []
    for topo in ("L_lo_pass", "L_hi_pass", "pi_lo_pass", "pi_hi_pass"):
        mr = solve_matching(Z_S, Z_L, f0, topology=topo,
                            Z0=Z0, Q_target=Q_target,
                            rlc_R=rlc_R, rlc_L=rlc_L, rlc_C=rlc_C)
        if mr.valid:
            results.append(mr)
    results.sort(key=lambda r: r.Q)
    return results