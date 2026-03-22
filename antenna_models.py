"""
antenna_models.py  —  Preset antenna RLC parameters
Each preset produces a realistic resonance curve.
"""

PRESETS = {
    "Custom": None,
    "Half-wave Dipole (~321 MHz)": {
        "R": 73.0, "L": 35e-9,  "C": 7.0e-12,
        "f_start": 50e6,  "f_stop": 1e9,
        "note": "Radiation resistance ~73 Ω, resonant ~321 MHz",
    },
    "Quarter-wave Monopole (~150 MHz)": {
        "R": 36.5, "L": 60e-9,  "C": 19e-12,
        "f_start": 50e6,  "f_stop": 500e6,
        "note": "Half the dipole impedance, over ground plane",
    },
    "Patch Antenna (~2.4 GHz)": {
        "R": 100.0, "L": 3.3e-9, "C": 1.3e-12,
        "f_start": 1e9,   "f_stop": 5e9,
        "note": "Microstrip patch, WiFi/BT band",
    },
    "Loop Antenna (~32 MHz)": {
        "R": 5.0,  "L": 2.5e-6, "C": 10e-12,
        "f_start": 5e6,  "f_stop": 150e6,
        "note": "Small loop, resonant ~32 MHz (was mislabelled ~1 MHz; sweep corrected)",
    },
    "UHF RFID (~915 MHz)": {
        "R": 20.0, "L": 8.7e-9,  "C": 3.5e-12,
        "f_start": 500e6, "f_stop": 1.5e9,
        "note": "RFID tag antenna, ISM 915 MHz",
    },
}