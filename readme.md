# VNA Reflection & VSWR Simulator

**QtHack04 — Quantum Technologies Hackathon 2026**  
Track 03 · RF Technology & Instrumentation · Problem Statement #12  
SRMIST Kattankulathur · March 30–31, 2026

---

## Overview

A desktop Vector Network Analyzer (VNA) simulator that models impedance mismatch, computes reflection physics, and visualizes the results in a professional instrument-grade interface. The simulator covers the full measurement chain from a cascaded RF network definition through to S11, VSWR, phase, group delay, and Smith chart display — all without physical hardware.

Built for the problem statement: *"Impedance mismatch causes reflections. Simulate VNA-based reflection measurements."*

---

## Screenshots

| VNA Plots Tab | Network Editor Tab |
|---|---|
| CH1 S11 + CH2 VSWR + Smith chart + readout panel | Pick-and-place schematic canvas |

---

## Physics Engine

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
| Unwrapped Phase | np.unwrap(∠Γ) |
| Group Delay | −dφ/dω (ns), non-uniform finite difference |
| Re(Z), Im(Z) | Real and imaginary parts of input impedance |

### Cascaded network solver (network_engine.py)

For multi-component circuits the simulation uses the signal-flow-graph termination equation (Pozar §4.3):

```
Γ_in = S11 + (S12 · Γ_L · S21) / (1 − S22 · Γ_L)
```

where Γ_L = (Z_load − Z₀) / (Z_load + Z₀) is the load reflection coefficient and [S11 S12; S21 S22] is the S-parameter matrix of all in-line 2-port elements (R, L, C, transmission lines) cascaded before the terminal load. This is backed by scikit-rf for the 2-port cascade arithmetic.

Supported component types: `R`, `L`, `C`, `RLC` (series), `TL` (lossless transmission line — physical or electrical length).

### Resonance detection

Sub-bin resonance frequency is found by 3-point Lagrange quadratic interpolation in log-frequency space. The −10 dB bandwidth uses log-frequency crossing interpolation, matching how real VNAs compute bandwidth on log sweeps.

---

## Features

### Measurement modes (CH2 selectable)

- Log Mag (dB) — S11 vs frequency
- Linear Magnitude — |Γ| vs frequency
- VSWR — capped at 50 for display clarity
- Phase (degrees) — wrapped, with NaN insertion at ±180° wrap boundaries
- Unwrapped Phase — continuous, multi-cycle
- Group Delay (ns) — non-uniform finite difference, IQR spike clipping
- Real Z (Ω) — resistance part of input impedance
- Imaginary Z (Ω) — reactance part
- Polar — |Γ| vs ∠Γ

### Smith Chart

Rendered using scikit-rf's `plot_s_smith()` with graceful fallback to a gradient polyline if scikit-rf is unavailable. Displays resistance circles and reactance arcs. Start/resonance/stop markers are plotted automatically.

### Marker system

Up to 8 numbered markers on any Cartesian plot or the Smith chart. Clicking the plot places a marker at the nearest frequency point (log-domain interpolation for log sweeps). With two active markers, a Δ row appears in the marker table showing frequency difference, S11 difference, VSWR difference, and wrap-safe phase difference.

### Network Editor (schematic canvas)

A pick-and-place graphical circuit editor built on `QGraphicsScene`. The canvas renders a two-rail schematic diagram (signal rail + ground rail) with textbook-correct component symbols:

- **R** → IEC shunt resistor (rectangle between rails)
- **L** → shunt inductor (vertical coil between rails)
- **C** → shunt capacitor (horizontal parallel plates between rails)
- **RLC** → series-RLC ladder section (series L on top rail, shunt C, series R on top rail — the standard antenna equivalent circuit)

Components are draggable; dropping one re-sorts the chain by X position. Double-clicking opens a value editor. The last component in the chain is always the terminal load (amber highlight).

### Preset antenna models

| Preset | R (Ω) | L | C | f₀ |
|---|---|---|---|---|
| Half-wave Dipole | 73 | 35 nH | 7 pF | ~321 MHz |
| Quarter-wave Monopole | 36.5 | 60 nH | 19 pF | ~150 MHz |
| Patch Antenna | 100 | 3.3 nH | 1.3 pF | ~2.4 GHz |
| Loop Antenna | 5 | 2.5 µH | 10 pF | ~32 MHz |
| UHF RFID Tag | 20 | 8.7 nH | 3.5 pF | ~915 MHz |

### Sweep configuration

- Frequency range: any start/stop in Hz with unit selectors (Hz / kHz / MHz / GHz)
- Sweep types: logarithmic (default, matches real VNA behaviour) or linear
- Points: 10 – 2000
- Reference impedance Z₀: configurable (default 50 Ω)

### Export

CSV export writes 10 columns per frequency point: Frequency, Re(Z), Im(Z), |Z|, Γ_Re, Γ_Im, |Γ|, S11 (dB), VSWR, Return Loss (dB).

---

## Architecture

```
main.py              Entry point — QApplication bootstrap
gui.py               PyQt6 UI: two-tab layout (VNA Plots | Network Editor)
  ├── rf_engine.py       Core physics: RLC → Z → Γ → S11 → VSWR → phase → GD
  ├── network_engine.py  Cascaded solver: scikit-rf S-parameter cascade + SFG termination
  ├── plot_s11.py        Multi-mode Matplotlib canvas (9 display modes, 8-marker system)
  ├── smith_chart.py     Smith chart canvas (scikit-rf or gradient fallback)
  ├── antenna_models.py  Preset RLC parameters for 5 antenna types
  └── export_utils.py    CSV export
```

### Data flow

```
Network Editor (schematic)
        │
        ▼  list of component dicts
network_engine.compute_network_response()
        │
        ▼  result dict
 ┌──────────────────────────────────────────┐
 │  frequencies, gamma, Z_L, s11_db, vswr, │
 │  phase_deg, group_delay_ns, bandwidth,  │
 │  resonance_check, ...                   │
 └──────────────────────────────────────────┘
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
plot_s11.VNACanvas          smith_chart.SmithCanvas
(CH1 + CH2 Cartesian)       (Γ-plane display)
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

### Basic sweep

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
- With two markers active, a Δ row appears in the marker table showing differences
- Clear all markers with **✕ Markers**

### Network Editor

1. Switch to the **Network Editor** tab
2. Use **＋R / ＋L / ＋C / ＋RLC** buttons to add components to the canvas
3. The rightmost component (amber) is always the terminal load
4. Drag components horizontally to reorder the cascade
5. Double-click any component to edit its values
6. Switch back to **VNA Plots** and run a sweep — the schematic defines the circuit

### Exporting data

After a sweep, go to **File → Export CSV…** or click **↓ Export CSV** to save all computed parameters for every frequency point.

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

**Track 03 — RF Technology & Instrumentation**  
Problem Statement #12: *VNA Reflection & VSWR Simulator*

> Impedance mismatch causes reflections. Simulate VNA-based reflection measurements.

**Objective:** Design and implement interactive visual simulations that demonstrate the working principles of RF systems. Model the physics, visualize system behaviour, and extract meaningful parameters through simulation.

---

## License

Built for academic/hackathon demonstration. All physics implementations are original. scikit-rf is used under its BSD licence.