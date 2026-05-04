"""
export_utils.py  —  CSV export of simulation results
"""

import csv
import numpy as np


def export_csv(path: str, result: dict,
               quantum_params: "dict | None" = None) -> None:
    """
    Write all computed VNA data to a CSV file.

    Columns (always present):
        Frequency (Hz), Re(Z), Im(Z), |Z|,
        Gamma_Re, Gamma_Im, |Gamma|,
        S11 (dB), VSWR, Return Loss (dB)

    Optional quantum columns (appended when quantum_params is provided):
        Q_loaded, T1_us, chi_MHz, dispersive_regime

    Parameters
    ----------
    path           : output file path
    result         : standard result dict from rf_engine / network_engine
    quantum_params : optional dict with quantum scalar results; expected keys:
                       q_loaded         (float)
                       t1_us            (float)
                       chi_mhz          (float)
                       dispersive_regime (bool)
                     Missing keys are written as empty strings.
                     If None, no quantum columns are added.
    """
    freqs = result["frequencies"]
    Z_L   = result["Z_L"]
    gamma = result["gamma"]
    s11   = result["s11_db"]
    vswr  = result["vswr"]
    rl    = result["return_loss"]

    # ── Quantum scalars (written once per row, same value) ────────────────────
    has_q = quantum_params is not None
    if has_q:
        qp       = quantum_params
        q_loaded = qp.get("q_loaded",          "")
        t1_us    = qp.get("t1_us",             "")
        chi_mhz  = qp.get("chi_mhz",           "")
        disp_reg = qp.get("dispersive_regime",  "")
        if isinstance(q_loaded,  float): q_loaded = f"{q_loaded:.4f}"
        if isinstance(t1_us,     float): t1_us    = f"{t1_us:.4f}"
        if isinstance(chi_mhz,   float): chi_mhz  = f"{chi_mhz:.6f}"
        if isinstance(disp_reg,  bool):  disp_reg = "yes" if disp_reg else "no"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "Frequency (Hz)", "Re(Z) (Ohm)", "Im(Z) (Ohm)", "|Z| (Ohm)",
            "Gamma_Re", "Gamma_Im", "|Gamma|",
            "S11 (dB)", "VSWR", "Return Loss (dB)",
        ]
        if has_q:
            header += ["Q_loaded", "T1_us", "chi_MHz", "dispersive_regime"]
        writer.writerow(header)

        for i in range(len(freqs)):
            g  = gamma[i]
            zl = Z_L[i]
            row = [
                f"{freqs[i]:.6e}",
                f"{zl.real:.6f}", f"{zl.imag:.6f}", f"{abs(zl):.6f}",
                f"{g.real:.8f}",  f"{g.imag:.8f}",  f"{abs(g):.8f}",
                f"{s11[i]:.6f}",  f"{vswr[i]:.6f}", f"{rl[i]:.6f}",
            ]
            if has_q:
                # Quantum params are scalar — same value on every row
                row += [q_loaded, t1_us, chi_mhz, disp_reg]
            writer.writerow(row)