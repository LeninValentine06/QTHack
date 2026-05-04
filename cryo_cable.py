"""
cryo_cable.py  —  Cryostat attenuation chain model
====================================================

Models the RF attenuation chain found in dilution-refrigerator-based
quantum computing setups.  Each temperature stage contributes:
  - Fixed attenuators (series resistors in the network_engine model)
  - A short coaxial transmission-line segment

All stage definitions are expressed as lists of component dicts that
are directly accepted by network_engine.compute_network_response().

Typical cryo-chain layout (Krantz et al., APL 6, 021318 — Fig. 4):
    300 K    →   4 K    →   900 mK   →   100 mK   →  device
     0 dB       −20 dB      −10 dB      −20 dB      (qubit)

Public API
----------
build_cryo_chain(stages: list[str]) -> list[dict]
total_attenuation_db(stages: list[str], f_hz: float) -> float
CRYO_STAGES : dict  (stage name → component list)
"""

from __future__ import annotations
import numpy as np

_EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Stage definitions
# ─────────────────────────────────────────────────────────────────────────────

# Attenuator value in dB → series resistance in Ω (mismatched-load approx)
# For a π-attenuator on 50 Ω:  R_series ≈ Z0 * (10^(A_dB/20) - 1)
# This is a first-order approximation; the network_engine models it as a
# series loss element which correctly degrades S11.

def _att_db_to_r(att_db: float, Z0: float = 50.0) -> float:
    """Convert attenuator value (dB) to equivalent series resistance (Ω)."""
    ratio = 10.0 ** (att_db / 20.0)
    return max(Z0 * (ratio - 1.0), _EPS)


CRYO_STAGES: dict = {

    # ── 300 K — Room temperature (no fixed attenuation; only cable)  ──────────
    "300K": [
        {
            "type": "TL",
            "Z0":    50.0,
            "length": 2.0,        # 2 m SMA-to-rack cable
            "vf":    0.66,        # PTFE coax (0.66c)
            "note":  "300 K room-temperature coax run (2 m, PTFE)",
        },
    ],

    # ── 4 K — First cold stage; 20 dB attenuator recommended ─────────────────
    "4K": [
        {
            "type":  "R",
            "value": _att_db_to_r(20.0),
            "note":  "4 K stage — 20 dB attenuator (π-pad, series R model)",
        },
        {
            "type":  "TL",
            "Z0":    50.0,
            "length": 0.30,       # ~30 cm stainless coax
            "vf":    0.70,
            "note":  "4 K → 900 mK stainless steel coax (0.30 m)",
        },
    ],

    # ── 900 mK — Still plate; 10 dB attenuator ───────────────────────────────
    "900mK": [
        {
            "type":  "R",
            "value": _att_db_to_r(10.0),
            "note":  "900 mK still plate — 10 dB attenuator",
        },
        {
            "type":  "TL",
            "Z0":    50.0,
            "length": 0.15,       # ~15 cm NbTi coax
            "vf":    0.75,
            "note":  "900 mK → 100 mK NbTi coax (0.15 m)",
        },
    ],

    # ── 100 mK — Mixing chamber (MXC); 20 dB attenuator ─────────────────────
    "100mK": [
        {
            "type":  "R",
            "value": _att_db_to_r(20.0),
            "note":  "100 mK MXC plate — 20 dB attenuator",
        },
        {
            "type":  "TL",
            "Z0":    50.0,
            "length": 0.05,       # 5 cm chip bond wire / package
            "vf":    0.85,
            "note":  "100 mK → device superconducting bondwire (0.05 m)",
        },
    ],
}

# Ordered stage sequence (source → device)
_STAGE_ORDER = ["300K", "4K", "900mK", "100mK"]


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Chain builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cryo_chain(stages: list) -> list:
    """
    Concatenate cryostat stage fragments into a single network_config prefix.

    Parameters
    ----------
    stages : list of stage names, e.g. ["300K", "4K", "900mK", "100mK"]
             Order is respected; typically source-to-device order.
             Stage names must be keys in CRYO_STAGES.

    Returns
    -------
    Flat list of component dicts that can be prepended to any qubit config
    and passed directly to network_engine.compute_network_response().

    Example
    -------
    >>> chain = build_cryo_chain(["4K", "900mK", "100mK"])
    >>> full_config = chain + QUBIT_SYSTEM_CONFIGS["Bare Qubit"]
    >>> result = compute_network_response(full_config, freq_array, Z0=50.0)
    """
    if not stages:
        return []

    chain = []
    for stage in stages:
        if stage not in CRYO_STAGES:
            raise KeyError(
                f"Unknown cryo stage '{stage}'. "
                f"Available: {_STAGE_ORDER}"
            )
        chain.extend(CRYO_STAGES[stage])
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Approximate total insertion loss
# ─────────────────────────────────────────────────────────────────────────────

# Attenuator dB per stage (the R elements)
_STAGE_ATT_DB = {
    "300K":  0.0,    # no fixed attenuator at room temperature
    "4K":   20.0,
    "900mK":10.0,
    "100mK":20.0,
}

# Approximate cable loss for UT-085 stainless: ~0.5 dB/m at 5 GHz (rough)
_CABLE_LOSS_DB_PER_M_PER_GHZ = 0.1   # conservative for NbTi/SS in cryo


def total_attenuation_db(stages: list, f_hz: float) -> float:
    """
    Return approximate total insertion loss (dB) for the specified stages
    at the given frequency.

    The estimate is:
        att_total = Σ_stage [att_fixed_dB(stage)
                             + cable_loss_dB/m · length_m · f_GHz]

    This is a conservative first-order model.  It does not account for
    impedance-mismatch reflections between attenuators or TL dispersion.
    Negative sign convention: returns a positive number for attenuation.

    Parameters
    ----------
    stages : list of stage name strings
    f_hz   : frequency in Hz

    Returns
    -------
    Total attenuation in dB (positive value).
    """
    f_ghz = f_hz / 1e9
    total = 0.0
    for stage in stages:
        if stage not in CRYO_STAGES:
            continue
        total += _STAGE_ATT_DB.get(stage, 0.0)
        for comp in CRYO_STAGES[stage]:
            if comp.get("type") == "TL":
                length_m = comp.get("length", 0.0)
                total += _CABLE_LOSS_DB_PER_M_PER_GHZ * length_m * f_ghz
    return float(total)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — standalone unit test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== cryo_cable standalone test ===\n")

    print("-- CRYO_STAGES component inventory --")
    for stage, comps in CRYO_STAGES.items():
        print(f"  {stage}: {len(comps)} components")
        for c in comps:
            print(f"    {c}")

    print("\n-- build_cryo_chain (all stages) --")
    chain = build_cryo_chain(_STAGE_ORDER)
    print(f"  Total components: {len(chain)}")

    print("\n-- total_attenuation_db --")
    for f_test in [1e9, 5e9, 10e9]:
        att = total_attenuation_db(_STAGE_ORDER, f_test)
        print(f"  f={f_test/1e9:.0f} GHz  →  total_att={att:.2f} dB")

    print("\n-- partial chain (4K + 900mK) --")
    partial = build_cryo_chain(["4K", "900mK"])
    print(f"  Components: {len(partial)}")
    att_partial = total_attenuation_db(["4K", "900mK"], 5e9)
    print(f"  Attenuation @ 5 GHz: {att_partial:.2f} dB")

    print("\n-- integration with network_engine (no GUI) --")
    try:
        import numpy as np
        from network_engine import compute_network_response
        from quantum_models import QUBIT_SYSTEM_CONFIGS

        full_config = build_cryo_chain(["4K", "100mK"]) + \
                      list(QUBIT_SYSTEM_CONFIGS["Bare Qubit"])
        freqs = np.logspace(np.log10(4e9), np.log10(6e9), 200)
        res = compute_network_response(full_config, freqs, Z0=50.0)
        print(f"  S11_min = {float(np.min(res['s11_db'])):.2f} dB")
        print("  Integration OK")
    except Exception as e:
        print(f"  Skipped: {e}")
