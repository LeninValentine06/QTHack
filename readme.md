# VNA Reflection & VSWR Simulator

**QtHack04 — Quantum Technologies Hackathon 2026**  
Track 03 · RF Technology & Instrumentation · Problem Statement #12  
SRMIST Kattankulathur · March 30–31, 2026

---

## Overview

A desktop Vector Network Analyzer (VNA) simulator that models impedance mismatch, computes reflection physics, and visualizes the results in a professional instrument-grade interface. The simulator covers the full measurement chain — from a cascaded RF network definition through to S11, VSWR, phase, group delay, and Smith chart display — and extends into superconducting quantum circuit simulation with a dedicated quantum tab and cryostat attenuation chain model, all without physical hardware.

Built for the problem statement: *"Impedance mismatch causes reflections. Simulate VNA-based reflection measurements."*

---

## Screenshots

| VNA Plots Tab | Network Editor Tab | Impedance Matching Tab | Quantum Systems Tab |
|---|---|---|---|
| CH1 S11 + CH2 VSWR + Smith chart + readout panel | Pick-and-place schematic canvas | L/Pi matching with before/after S11 comparison | Qubit sweep + cryo chain + T₁/χ readout |

---

## Physics Engines

### Core equations (rf_engine.py)

Every measurement derives from five equations applied across a swept frequency array:

```
1.  Z(f)    = R + j(ωL − 1/ωC)          Series RLC antenna impedance
2.  Γ       = (Z − Z₀) / (Z + Z₀)       Reflection coefficient
3.  S11(dB) = 20 · log₁₀|Γ|             S11 in decibels
4.  VSWR    = (1 + |Γ|) / (1 − |Γ|)     Voltage Standing Wave Ratio
5.  φ       = ∠Γ                         Phase of reflection coefficient
```

Additional derived quantities:

| Quantity | Formula |
|---|---|
| Return Loss | −S11 (dB) |
| Unwrapped Phase | `np.unwrap(∠Γ)` |
| Group Delay | −dφ/dω (ns), non-uniform finite difference |
| Re(Z), Im(Z) | Real and imaginary parts of input impedance |

### Cascaded network solver (network_engine.py)

For multi-component circuits the simulation uses the signal-flow-graph (SFG) termination equation (Pozar §4.3):

```
Γ_in = S11 + (S12 · Γ_L · S21) / (1 − S22 · Γ_L)
```

where Γ_L = (Z_load − Z₀) / (Z_load + Z₀) is the load reflection coefficient and [S11 S12; S21 S22] is the S-parameter matrix of all in-line 2-port elements cascaded before the terminal load. This is backed by scikit-rf for the 2-port cascade arithmetic.

Supported component types: `R`, `L`, `C`, `RLC` (series), `TL` (lossless transmission line — physical or electrical length).

### Impedance matching solver (matching_engine.py)

Implements analytic L-network and Pi-network matching design (Pozar §5.1–§5.2):

```
L-network Q  = √(R_high / R_low − 1)
Pi-network   = two back-to-back L-networks sharing a virtual resistance R_virt = R_max / (Q² + 1)
```

Supported topologies: `L_lo_pass`, `L_hi_pass`, `L_auto`, `pi_lo_pass`, `pi_hi_pass`.

Physical realism is enforced via finite component Q factors (series ESR model):

```
Inductor:   Z = jωL + ωL/Q_L      (Q_L ≈ 50, ESR rises with frequency)
Capacitor:  Z = 1/(jωC) + 1/(ωCQ_C)  (Q_C ≈ 200, NP0/C0G ceramic)
```

This prevents the unphysical −80 dB cancellation null and gives realistic S11_min of −20 to −35 dB. A gamma floor of |Γ| ≥ 10⁻³ is enforced to reflect real-world hardware limits.

### Quantum network engine (quantum_network.py)

Wraps `network_engine` with quantum-physics post-processing:

| Quantity | Formula |
|---|---|
| Loaded Q | f₀ / BW₋₃dB (log-interpolated bandwidth) |
| Internal Q estimate | Q_loaded · Q_coupling / (Q_coupling − Q_loaded) |
| T₁ relaxation time | Q_loaded / (π · f_res) [µs] |
| Dispersive shift χ | g² · α / (Δ · (Δ + α)) [MHz] — Koch et al. PRA 76, 042319 (2007) |
| Dispersive regime | \|Δ\| > 10g |

where Δ = f_readout − f_qubit is the qubit-resonator detuning, g is the coupling strength, and α is the transmon anharmonicity.

### Cryostat attenuation chain (cryo_cable.py)

Models the RF signal path inside a dilution refrigerator (Krantz et al., APL 6, 021318, Fig. 4):

| Stage | Fixed Attenuator | Cable |
|---|---|---|
| 300 K | 0 dB | 2 m PTFE coax (v_f = 0.66c) |
| 4 K | −20 dB | 30 cm stainless coax (v_f = 0.70c) |
| 900 mK | −10 dB | 15 cm NbTi coax (v_f = 0.75c) |
| 100 mK | −20 dB | 5 cm superconducting bondwire (v_f = 0.85c) |

Each stage is represented as a flat list of component dicts that can be prepended to any quantum system config and passed directly to the cascaded network solver. Total insertion loss at the full cryo chain is approximately −50 dB at DC, increasing with frequency due to cable skin-depth losses (~0.1 dB/m/GHz conservatively).

### Resonance detection

Sub-bin resonance frequency is found by 3-point Lagrange quadratic interpolation in log-frequency space. The −10 dB bandwidth uses log-frequency crossing interpolation, matching how real VNAs compute bandwidth on log sweeps.

---

## Features

### Tab 1 — VNA Plots

#### Measurement modes (CH2 selectable)

- Log Mag (dB) — S11 vs frequency
- Linear Magnitude — |Γ| vs frequency
- VSWR — capped at 50 for display clarity
- Phase (degrees) — wrapped, with NaN insertion at ±180° wrap boundaries
- Unwrapped Phase — continuous, multi-cycle
- Group Delay (ns) — non-uniform finite difference, IQR spike clipping
- Real Z (Ω) — resistance part of input impedance
- Imaginary Z (Ω) — reactance part
- Polar — |Γ| vs ∠Γ

#### Smith Chart

Rendered using scikit-rf's `plot_s_smith()` with graceful fallback to a gradient polyline if scikit-rf is unavailable. Displays resistance circles and reactance arcs. Start/resonance/stop markers are plotted automatically.

#### Marker system

Up to 8 numbered markers on any Cartesian plot or the Smith chart. Clicking the plot places a marker at the nearest frequency point (log-domain interpolation for log sweeps). With two active markers, a Δ row appears in the marker table showing frequency difference, S11 difference, VSWR difference, and wrap-safe phase difference.

#### Preset antenna models

| Preset | R (Ω) | L | C | f₀ |
|---|---|---|---|---|
| Half-wave Dipole | 73 | 35 nH | 7 pF | ~321 MHz |
| Quarter-wave Monopole | 36.5 | 60 nH | 19 pF | ~150 MHz |
| Patch Antenna | 50 | 3.3 nH | 1.3 pF | ~2.4 GHz |
| Loop Antenna | 50 | 2.5 µH | 10 pF | ~32 MHz |
| UHF RFID Tag | 20 | 8.7 nH | 3.5 pF | ~915 MHz |

#### Sweep configuration

- Frequency range: any start/stop in Hz with unit selectors (Hz / kHz / MHz / GHz)
- Sweep types: logarithmic (default, matches real VNA behaviour) or linear
- Points: 10 – 2000
- Reference impedance Z₀: configurable (default 50 Ω)

#### Export

CSV export writes 10 columns per frequency point: Frequency, Re(Z), Im(Z), |Z|, Γ_Re, Γ_Im, |Γ|, S11 (dB), VSWR, Return Loss (dB). When triggered from the Quantum tab, four additional columns are appended: Q_loaded, T₁ (µs), χ (MHz), dispersive_regime.

---

### Tab 2 — Network Editor (schematic canvas)

A pick-and-place graphical circuit editor built on `QGraphicsScene`. The canvas renders a two-rail schematic diagram (signal rail + ground rail) with textbook-correct component symbols:

- **R** → IEC shunt resistor (rectangle between rails)
- **L** → shunt inductor (vertical coil between rails)
- **C** → shunt capacitor (horizontal parallel plates between rails)
- **RLC** → series-RLC ladder section (series L on top rail, shunt C, series R on top rail — the standard antenna equivalent circuit)

Components are draggable; dropping one re-sorts the chain by X position. Double-clicking opens a value editor. The last component in the chain is always the terminal load (amber highlight).

---

### Tab 3 — Impedance Matching

An interactive matching-network design tool that takes source and load impedances and computes the required L or Pi network element values at a target frequency.

#### Inputs

| Field | Description |
|---|---|
| Z_S | Source impedance (real + imaginary, Ω) |
| Z_L | Load impedance; or click **⬇ Load from VNA** to use the current resonance impedance from the VNA tab |
| f₀ | Design frequency |
| Z₀ | System reference impedance (default 50 Ω) |
| Topology | L low-pass, L high-pass, Pi low-pass, Pi high-pass |
| Q target | Pi-network quality factor target (must exceed Q_min) |

#### Outputs and displays

- Computed element values (L, C in appropriate SI units) shown in a table
- Smith chart overlay: unmatched trajectory (dashed) vs matched trajectory (solid), illustrating the transformation from Z_L to Z₀
- S11 before/after comparison plot on the same frequency axis — showing the improvement in return loss across the band
- Design notes: Q factor, bandwidth estimate, topology description
- **≡ COMPARE ALL** button evaluates all four topologies and ranks them by Q in a comparison table, letting you pick the lowest-Q (broadest bandwidth) solution
- **→ APPLY TO NETWORK EDITOR** inserts the computed matching elements before the existing RLC load in the schematic canvas, with series and shunt flags correctly preserved

---

### Tab 4 — ⚛ Quantum Systems

A dedicated tab for superconducting quantum circuit simulation. Runs on a background thread so the GUI stays responsive during sweeps.

#### System selector

Three pre-built quantum network configs from `quantum_models.py`, each representing a physically realistic cQED topology:

| Config | Topology | Frequency range |
|---|---|---|
| Bare Qubit | Linearised transmon (5 GHz, Q ~ 10⁶) | 4–6 GHz |
| Qubit + Readout Resonator | Transmon + readout resonator (6.5 GHz) + 2 fF coupling cap | 4–8 GHz |
| Full Purcell + Readout + Qubit Cascade | Purcell filter (7 GHz) → feedline TL → readout → coupling cap → qubit | 4–8.5 GHz |

Selecting a config auto-populates the sweep frequency range from pre-computed hint values.

Additionally, five preset individual quantum device RLC models are available in `quantum_models.py` for use with the standard rf_engine:

| Preset | Device | f₀ | Q |
|---|---|---|---|
| Transmon Qubit | Linearised transmon, Al on Si | 5 GHz | ~10⁶ |
| Readout Resonator | Coupled feedline resonator | 6.5 GHz | ~10³ |
| Purcell Filter | Bandpass Purcell filter | 7 GHz | ~200 |
| Coupled Qubit + Readout | Two-resonator coupled system (qubit mode) | 5 GHz | ~5×10⁵ |

#### Sweep parameters

| Field | Description |
|---|---|
| f_start / f_stop | Frequency range in GHz |
| Points | Number of frequency samples (default 400) |
| Z₀ | Reference impedance (default 50 Ω) |
| g (MHz) | Qubit–resonator coupling strength |
| α (MHz) | Transmon anharmonicity (default −200 MHz) |

#### Cryo attenuation chain

Four checkboxes select which cryostat temperature stages to prepend to the quantum network before simulation:

| Checkbox | Stage | Attenuation |
|---|---|---|
| 300 K | Room temperature cable run (2 m PTFE) | 0 dB fixed |
| 4 K | First cold stage | −20 dB |
| 900 mK | Still plate | −10 dB |
| 100 mK | Mixing chamber (MXC) | −20 dB |

Any combination of stages can be enabled. The resulting component chain is prepended to the quantum system config, so the full signal path from room temperature to the qubit is simulated in a single S11 sweep.

#### Readout panel

Six live values updated after each sweep:

| Readout | Description |
|---|---|
| f₀ (GHz) | Resonance frequency (S11 minimum) |
| Loaded Q | Q_loaded from −3 dB bandwidth |
| Int. Q est. | Estimated internal (unloaded) Q |
| T₁ (µs) | Energy relaxation time estimate |
| χ (MHz) | Dispersive shift (qubit–readout coupling shift) |
| Dispersive | ✔ YES / ✘ NO badge — whether \|Δ\| > 10g |

#### Canvas

A dual-trace matplotlib figure embedded in the tab:

- **CH1 (blue, left axis)** — S11 in dB with a vertical resonance marker and frequency annotation
- **CH2 (teal, right axis)** — Rolling-window loaded Q vs frequency, showing how Q varies across the sweep band

The sweep result is also emitted via the `sweep_completed` signal, which updates the Smith chart in the main VNA Plots tab so the quantum result appears in the Γ-plane alongside the RF sweeps.

---

## Architecture

```
main.py                 Entry point — QApplication bootstrap
gui.py                  PyQt6 UI: four-tab layout
  ├── rf_engine.py          Core physics: RLC → Z → Γ → S11 → VSWR → phase → GD
  ├── network_engine.py     Cascaded solver: scikit-rf S-parameter cascade + SFG termination
  ├── plot_s11.py           Multi-mode Matplotlib canvas (9 display modes, 8-marker system)
  ├── smith_chart.py        Smith chart canvas (scikit-rf or gradient fallback)
  ├── antenna_models.py     Preset RLC parameters for 5 antenna types
  ├── export_utils.py       CSV export (RF + optional quantum columns)
  ├── matching_widget.py    Impedance Matching tab UI
  │     └── matching_engine.py   L/Pi matching solver, sweep with lossy ESR model
  └── quantum_gui_tab.py    ⚛ Quantum tab UI (background QThread sweep)
        ├── quantum_models.py    Preset quantum device RLC params + system network configs
        ├── quantum_network.py   Quantum sweep wrapper: Q, T₁, χ, dispersive detection
        └── cryo_cable.py        Cryostat attenuation chain (300K / 4K / 900mK / 100mK)
```

### Data flow

```
┌──────────────────────────────────────┐    ┌────────────────────────────────┐
│  Network Editor (schematic canvas)   │    │  ⚛ Quantum Tab                 │
│  list of component dicts             │    │  system config + cryo prefix   │
└──────────────┬───────────────────────┘    └──────────────┬─────────────────┘
               │                                           │
               ▼                                           ▼
  network_engine.compute_network_response()    quantum_network.run_quantum_sweep()
               │                                           │
               └──────────────┬────────────────────────────┘
                              ▼
          ┌───────────────────────────────────────────────────┐
          │  result dict                                       │
          │  frequencies, gamma, Z_L, s11_db, vswr,           │
          │  phase_deg, group_delay_ns, bandwidth,            │
          │  resonance_check, [q_loaded, t1_us, chi_mhz ...]  │
          └──────┬────────────────────────────────────────────┘
                 │
        ┌────────┴──────────────────────┐
        ▼                               ▼
plot_s11.VNACanvas               smith_chart.SmithCanvas
(CH1 + CH2 Cartesian)            (Γ-plane display)


┌────────────────────────────────────────────┐
│  Impedance Matching Tab                    │
│  Z_S, Z_L, f₀, topology, Q_target         │
│      ▼                                     │
│  matching_engine.solve_matching()          │
│      ▼                                     │
│  MatchResult: L/C values, Q, BW           │
│      ▼                                     │
│  matching_engine.sweep_matched_network()   │
│      ▼                                     │
│  S11 before/after + Smith overlay          │
└────────────────────────────────────────────┘
```

---

## Installation

```bash
# Clone / copy the project files
cd vna-simulator

# Install dependencies
pip install PyQt6 numpy matplotlib scikit-rf

# Run
python main.py
```

**Requirements:**

| Package | Version | Purpose |
|---|---|---|
| PyQt6 | ≥ 6.4.0 | GUI framework |
| numpy | ≥ 1.23.0 | Vectorised physics computation |
| matplotlib | ≥ 3.6.0 | Plot canvases (embedded in Qt) |
| scikit-rf | ≥ 0.29.0 | S-parameter cascade, Smith chart rendering |

Tested on Windows 10/11, Python 3.11+. Minimum screen resolution: 1440 × 900.

---

## Usage

### Basic RF sweep

1. Open the **VNA Plots** tab
2. Set Start/Stop frequency and sweep points in the **Sweep Control** panel
3. Select a preset from the **Preset** dropdown (or enter custom R/L/C values)
4. Click **▶ START SWEEP**
5. Results appear in CH1 (S11), CH2 (selectable mode), and the Smith chart
6. The readout panel shows resonance frequency, S11 minimum, bandwidth, and Q factor

### Placing markers

- Enable **Click → place marker** (checked by default)
- Click anywhere on CH1 or CH2 to place a numbered marker
- Click the Smith chart to place a marker at the nearest Γ point
- With two markers active, a Δ row appears in the marker table
- Clear all markers with **✕ Markers**

### Network Editor

1. Switch to the **Network Editor** tab
2. Use **＋R / ＋L / ＋C / ＋RLC** buttons to add components to the canvas
3. The rightmost component (amber) is always the terminal load
4. Drag components horizontally to reorder the cascade
5. Double-click any component to edit its values
6. Switch back to **VNA Plots** and run a sweep — the schematic defines the circuit

### Impedance Matching

1. Switch to the **Impedance Matching** tab
2. Enter source impedance Z_S and load impedance Z_L (or click **⬇ Load from VNA** to import the resonance point from the last sweep)
3. Set the design frequency f₀ and select a topology
4. Click **▶ CALCULATE** to compute element values and view the Smith chart transformation and S11 before/after plot
5. Click **≡ COMPARE ALL** to rank all four L/Pi topologies by Q
6. Click **→ APPLY TO NETWORK EDITOR** to insert the matching network into the schematic

### Quantum Systems sweep

1. Switch to the **⚛ Quantum** tab
2. Select a system config from the **QUANTUM SYSTEM** dropdown (frequency range auto-populates)
3. Adjust **g (MHz)** and **α (MHz)** for the coupling strength and anharmonicity
4. Optionally check one or more **CRYO ATTENUATION CHAIN** stages to simulate the full dilution refrigerator signal path
5. Click **▶ RUN QUANTUM SWEEP** — the sweep runs in a background thread
6. The readout panel updates with f₀, Q_loaded, Int. Q, T₁, χ, and dispersive regime status
7. CH1 (S11) and CH2 (rolling-window Q) appear on the dual-trace canvas; the Quantum tab also pushes its result to the Smith chart in the VNA Plots tab

### Exporting data

After any sweep, go to **File → Export CSV…** or click **↓ Export CSV** to save all computed parameters. Quantum sweeps append Q_loaded, T₁, χ, and dispersive_regime columns to the standard 10-column RF export.

---

## Key Design Decisions

**Why series RLC for the antenna model?**  
Near resonance, every antenna behaves electrically as a series RLC circuit. The parameters map directly to radiation resistance (R), conductor inductance (L), and charge distribution capacitance (C). This is standard antenna theory and sufficient for all demonstration purposes.

**Why logarithmic sweep by default?**  
RF resonances are symmetric on a log frequency axis. Log spacing gives equal resolution per decade, matching how all real VNAs operate. Bandwidth detection uses log-frequency interpolation for the same reason.

**Why signal-flow-graph termination instead of a full 2-port cascade?**  
The SFG approach correctly separates in-line 2-port elements (which contribute to the cascade matrix) from the terminal load (which is a 1-port impedance). This avoids the topology error of modelling a terminal antenna as a shunt 2-port element, which would compute a completely different S11.

**Why scikit-rf for the cascade arithmetic?**  
scikit-rf provides vectorised S-parameter operations across all frequency points simultaneously. The `**` cascade operator is numerically stable and handles frequency-dispersive elements correctly.

**Why linearised RLC for the transmon qubit?**  
The transmon Hamiltonian reduces to a harmonic oscillator at low drive powers. The anharmonicity (−200 MHz typical) is small compared to the transition frequency (5 GHz), so treating the qubit as a high-Q series RLC is valid for VNA-style linear response measurements. Nonlinear phenomena (AC Stark shift, Kerr effect) are captured via the χ and α parameters.

**Why a background thread for quantum sweeps?**  
A full quantum network sweep with cryo chain over 400 log-spaced frequency points involves a chain of 10+ S-parameter cascade operations per point. Running this synchronously would freeze the GUI for a noticeable interval. The `_SweepWorker` QThread keeps the event loop responsive, with the run button disabled during computation and re-enabled on `finished` or `error` signals.

**Why per-stage cryo chain rather than a single aggregate attenuator?**  
Each temperature stage has a different cable type with different propagation velocity and loss coefficient. Modelling each stage individually allows the simulator to correctly capture TL-length-dependent phase rotation effects at higher frequencies, not just the bulk attenuation figure.

---

## Team

**Abishek** — ECE, SRMIST Kattankulathur  
**Lenin Valentine C J** — ECE, SRMIST Kattankulathur  
**Abinesh** — ECE, SRMIST Kattankulathur  
**Vishal** — ECE, SRMIST Kattankulathur  

Supervised by **Dr. Sivakumar E**, Assistant Professor, ECE Department, SRMIST

---

## Hackathon Context

**QtHack04 — Quantum Technologies Hackathon**  
March 30–31, 2026 · SRMIST Kattankulathur  
Organized by the Faculty of Engineering and Technology  
SRM Institute of Science and Technology, Kattankulathur Campus

This project secured **1st Place** in **Track 03 — RF Technology & Instrumentation** and was awarded a **₹50,000 cash prize** for its RF system modelling accuracy, interactive visualization pipeline, and instrument-grade VNA simulation workflow.

**Track 03 — RF Technology & Instrumentation**  
Problem Statement #12: *VNA Reflection & VSWR Simulator*

> Impedance mismatch causes reflections. Simulate VNA-based reflection measurements.

**Objective:** Design and implement interactive visual simulations that demonstrate the working principles of RF systems. Model the physics, visualize system behaviour, and extract meaningful parameters through simulation.

---

## References

- Pozar, D. M. — *Microwave Engineering*, 4th ed. (Wiley, 2012) — §4.3 (SFG termination), §5.1–§5.2 (L/Pi matching)
- Koch, J. et al. — *Charge-insensitive qubit design derived from the Cooper pair box*, PRA 76, 042319 (2007) — dispersive shift formula
- Krantz, P. et al. — *A Quantum Engineer's Guide to Superconducting Qubits*, APL 6, 021318 (2019) — cryostat attenuation chain layout
- Reed, M. D. et al. — *Fast reset and suppressing spontaneous emission of a superconducting qubit*, APL 96, 203110 (2010) — Purcell filter

---

## License

Built for academic/hackathon demonstration. All physics implementations are original. scikit-rf is used under its BSD licence.
