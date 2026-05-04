"""
quantum_models.py  —  Preset quantum device RLC parameters and system configs
==============================================================================

Analogous to antenna_models.py but for superconducting quantum circuits.

Each entry in QUANTUM_PRESETS maps a human-readable name to a dict with keys
    R, L, C, f_start, f_stop, note
that match the existing RLC field names consumed by rf_engine.run_simulation()
and network_engine.compute_network_response().

QUBIT_SYSTEM_CONFIGS provides pre-built network_config lists (lists of
component dicts) directly accepted by network_engine.compute_network_response().

All frequencies in Hz internally.
"""

import numpy as np

_EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Individual quantum device presets
# ─────────────────────────────────────────────────────────────────────────────

# Helper: compute L from f_res and C, Q_internal from R, L, f_res
def _rlc_from_f_Q(f_res_hz: float, Q_int: float, C_F: float) -> dict:
    """
    Given resonance frequency, internal Q, and capacitance, return R, L, C.

    Series RLC: f_res = 1/(2π√(LC)) → L = 1/(4π² f² C)
    Q_int = ω_res L / R → R = ω_res L / Q_int
    """
    omega = 2.0 * np.pi * f_res_hz
    L = 1.0 / (omega**2 * C_F + _EPS)
    R = omega * L / (Q_int + _EPS)
    return R, L, C_F


# Transmon qubit linearised at 5 GHz
# High internal Q (~1e6) because superconducting qubits have extremely low loss.
# C ~ 70 fF is typical for a transmon shunt capacitor.
_R_q, _L_q, _C_q = _rlc_from_f_Q(5.0e9, 1.0e6, 70e-15)

# Readout resonator at 6.5 GHz, loaded Q ~1000 (coupling to feedline degrades Q)
_R_r, _L_r, _C_r = _rlc_from_f_Q(6.5e9, 1.0e3, 50e-15)

# Purcell filter at 7 GHz, moderate Q ~200 (bandpass characteristic)
_R_p, _L_p, _C_p = _rlc_from_f_Q(7.0e9, 200.0, 40e-15)

# Coupled system: qubit at 5 GHz + readout at 6.5 GHz
# Each treated individually here; coupling via C_k = 2 fF between them
_R_cq, _L_cq, _C_cq = _rlc_from_f_Q(5.0e9, 5.0e5, 70e-15)
_R_cr, _L_cr, _C_cr = _rlc_from_f_Q(6.5e9, 1.0e3, 50e-15)

QUANTUM_PRESETS: dict = {
    "Custom": None,

    "Transmon Qubit (~5 GHz, Q~1e6)": {
        "R": _R_q,
        "L": _L_q,
        "C": _C_q,
        "f_start": 4.0e9,
        "f_stop":  6.0e9,
        "note": (
            "Linearised transmon qubit at 5 GHz. "
            "R represents residual quasiparticle/radiation loss; "
            "intrinsic Q ~ 1×10⁶ typical for Al transmons on Si."
        ),
    },

    "Readout Resonator (~6.5 GHz, Q~1000)": {
        "R": _R_r,
        "L": _L_r,
        "C": _C_r,
        "f_start": 5.5e9,
        "f_stop":  7.5e9,
        "note": (
            "Coupled readout resonator at 6.5 GHz. "
            "Loaded Q ~ 1000 due to intentional coupling to 50-Ω feedline "
            "for fast measurement (~100 ns ring-down)."
        ),
    },

    "Purcell Filter (~7 GHz, Q~200)": {
        "R": _R_p,
        "L": _L_p,
        "C": _C_p,
        "f_start": 6.0e9,
        "f_stop":  8.0e9,
        "note": (
            "Purcell bandpass filter at 7 GHz. "
            "Suppresses qubit emission at readout frequency, "
            "decoupling T1 from readout Q (Reed et al. 2010)."
        ),
    },

    "Coupled Qubit + Readout System": {
        # Approximation: effective series RLC for the lower (qubit) mode
        # in a two-resonator coupled system.
        "R": _R_cq,
        "L": _L_cq,
        "C": _C_cq,
        "f_start": 4.0e9,
        "f_stop":  8.0e9,
        "note": (
            "Two coupled RLC resonators: qubit at 5 GHz + readout at 6.5 GHz, "
            "coupling capacitor C_k ~ 2 fF. "
            "This preset shows the lower (qubit) mode; "
            "use 'Qubit + Readout Cascade' network config for both modes."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Pre-built network configs (for network_engine)
# ─────────────────────────────────────────────────────────────────────────────

#  Component dict format (same as network_engine.compute_network_response):
#      {"type": "R",   "value": <Ω>}
#      {"type": "L",   "value": <H>}
#      {"type": "C",   "value": <F>}
#      {"type": "RLC", "R": .., "L": .., "C": ..}
#      {"type": "TL",  "Z0": 50, "length": <m>, "vf": 0.85}

# (a) Bare transmon qubit — single RLC resonator
_BARE_QUBIT: list = [
    {
        "type": "RLC",
        "R": _R_q,
        "L": _L_q,
        "C": _C_q,
        "note": "Linearised transmon qubit @ 5 GHz",
    }
]

# (b) Qubit + readout resonator
#
# Network_engine topology rule: only ONE RLC block is allowed, and it must be
# the last (terminal) element.  Intermediate resonators are represented as
# individual series R / L / C components so each is a legitimate in-line 2-port.
#
# Physical meaning: the readout resonator appears in series with the feedline;
# the qubit (terminal load) is the final RLC.
#
_QUBIT_READOUT: list = [
    # Feed section
    {
        "type": "TL",
        "Z0": 50.0,
        "length": 1e-3,
        "vf": 0.85,
        "note": "Short feed section to readout resonator",
    },
    # Readout resonator as individual series elements @ 6.5 GHz
    {"type": "R", "value": _R_r,  "note": "Readout R @ 6.5 GHz"},
    {"type": "L", "value": _L_r,  "note": "Readout L @ 6.5 GHz"},
    {"type": "C", "value": _C_r,  "note": "Readout C @ 6.5 GHz"},
    # Coupling capacitor
    {
        "type": "C",
        "value": 2e-15,
        "note": "Coupling capacitor C_k ~ 2 fF",
    },
    # Transmon qubit — terminal load (must be RLC)
    {
        "type": "RLC",
        "R": _R_cq,
        "L": _L_cq,
        "C": _C_cq,
        "note": "Transmon qubit @ 5 GHz (terminal load)",
    },
]

# (c) Full Purcell + readout + qubit cascade (Pozar §4.3 / cryo-lab topology)
#     Signal flow: port → Purcell filter → TL → readout → C_k → qubit
#
# Again intermediate resonators (Purcell, readout) use individual R/L/C;
# only the terminal qubit uses the RLC shorthand.
#
_FULL_CASCADE: list = [
    # Room-temperature to chip coax
    {
        "type": "TL",
        "Z0": 50.0,
        "length": 2e-3,
        "vf": 0.85,
        "note": "Coax from room-temperature port to chip",
    },
    # Purcell filter as individual series elements @ 7 GHz
    {"type": "R", "value": _R_p,  "note": "Purcell R @ 7 GHz"},
    {"type": "L", "value": _L_p,  "note": "Purcell L @ 7 GHz"},
    {"type": "C", "value": _C_p,  "note": "Purcell C @ 7 GHz"},
    # On-chip TL
    {
        "type": "TL",
        "Z0": 50.0,
        "length": 0.5e-3,
        "vf": 0.85,
        "note": "On-chip transmission line segment",
    },
    # Readout resonator as individual series elements @ 6.5 GHz
    {"type": "R", "value": _R_r,  "note": "Readout R @ 6.5 GHz"},
    {"type": "L", "value": _L_r,  "note": "Readout L @ 6.5 GHz"},
    {"type": "C", "value": _C_r,  "note": "Readout C @ 6.5 GHz"},
    # Coupling capacitor
    {
        "type": "C",
        "value": 2e-15,
        "note": "Coupling capacitor C_k ~ 2 fF",
    },
    # Transmon qubit — terminal load (must be RLC)
    {
        "type": "RLC",
        "R": _R_cq,
        "L": _L_cq,
        "C": _C_cq,
        "note": "Transmon qubit @ 5 GHz (terminal load)",
    },
]

QUBIT_SYSTEM_CONFIGS: dict = {
    "Bare Qubit":                  _BARE_QUBIT,
    "Qubit + Readout Resonator":   _QUBIT_READOUT,
    "Full Purcell + Readout + Qubit Cascade": _FULL_CASCADE,
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Frequency range hints per system config
# ─────────────────────────────────────────────────────────────────────────────

QUBIT_SYSTEM_FREQ_HINTS: dict = {
    "Bare Qubit":                  (4.0e9, 6.0e9),
    "Qubit + Readout Resonator":   (4.0e9, 8.0e9),
    "Full Purcell + Readout + Qubit Cascade": (4.0e9, 8.5e9),
}


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — standalone unit test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== QUANTUM_PRESETS ===")
    for name, p in QUANTUM_PRESETS.items():
        if p is None:
            print(f"  {name}: Custom (None)")
            continue
        f_res_theory = 1.0 / (2.0 * np.pi * np.sqrt(p["L"] * p["C"]))
        Q_int = (2.0 * np.pi * f_res_theory * p["L"]) / (p["R"] + _EPS)
        print(f"  {name}")
        print(f"    R={p['R']:.4e} Ω  L={p['L']:.4e} H  C={p['C']:.4e} F")
        print(f"    f_res={f_res_theory/1e9:.4f} GHz   Q_int={Q_int:.2e}")

    print("\n=== QUBIT_SYSTEM_CONFIGS ===")
    for name, cfg in QUBIT_SYSTEM_CONFIGS.items():
        print(f"  {name}  ({len(cfg)} components)")
        for comp in cfg:
            print(f"    {comp}")
