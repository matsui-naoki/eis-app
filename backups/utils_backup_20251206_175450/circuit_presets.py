"""
Preset equivalent circuit models for EIS Analyzer.
Each entry: (display_name, circuit_string, initial_guess)
"""
from collections import OrderedDict


# Preset circuit models with display name -> (circuit_string, initial_guess)
# Order: Custom, common solid electrolyte models, then others
PRESET_CIRCUITS = OrderedDict([
    # Primary choices
    ("Custom", ("", None)),
    ("(R/CPE)-CPE", ("p(R1,CPE1)-CPE2", [1e5, 1e-9, 0.9, 1e-7, 0.7])),
    ("(R/CPE)-(R/CPE)-CPE", ("p(R1,CPE1)-p(R2,CPE2)-CPE3", [1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9, 1e-6, 0.5])),
    ("(R/CPE)-(R/CPE)-(R/CPE)", ("p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)", [1e3, 1e-12, 0.98, 1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9])),
    ("R-CPE", ("R1-CPE1", [1e4, 1e-9, 0.9])),
    ("R-L-CPE", ("R1-L1-CPE1", [1e2, 1e-6, 1e-9, 0.9])),
    # --- Other models ---
    ("(R/CPE)", ("p(R1,CPE1)", [1e4, 1e-9, 0.9])),
    ("(R/CPE)-(R/CPE)", ("p(R1,CPE1)-p(R2,CPE2)", [1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9])),
    ("R-(R/CPE)", ("R1-p(R2,CPE1)", [1e2, 1e5, 1e-9, 0.9])),
    ("R-(R/CPE)-CPE", ("R1-p(R2,CPE1)-CPE2", [1e2, 1e5, 1e-9, 0.9, 1e-7, 0.7])),
    ("R-(R/CPE)-(R/CPE)", ("R1-p(R2,CPE1)-p(R3,CPE2)", [1e2, 1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9])),
    ("R-(R/CPE)-(R/CPE)-CPE", ("R1-p(R2,CPE1)-p(R3,CPE2)-CPE3", [1e2, 1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9, 1e-6, 0.5])),
    # Capacitor models
    ("R", ("R1", [1e4])),
    ("R-C", ("R1-C1", [1e4, 1e-9])),
    ("(R/C)", ("p(R1,C1)", [1e4, 1e-9])),
    ("R-(R/C)", ("R1-p(R2,C1)", [1e2, 1e5, 1e-10])),
    ("(R/C)-(R/C)", ("p(R1,C1)-p(R2,C2)", [1e4, 1e-11, 1e5, 1e-9])),
    # Warburg models
    ("R-W", ("R1-W1", [1e4, 1e-3])),
    ("(R/CPE)-W", ("p(R1,CPE1)-W1", [1e5, 1e-9, 0.9, 1e-3])),
    ("R-(R/CPE)-W", ("R1-p(R2,CPE1)-W1", [1e2, 1e5, 1e-9, 0.9, 1e-3])),
    ("(R/CPE)-(R/CPE)-W", ("p(R1,CPE1)-p(R2,CPE2)-W1", [1e4, 1e-11, 0.95, 1e5, 1e-9, 0.9, 1e-3])),
    # Randles circuit: Rs - ((Rct - W) // CPE)
    ("Randles: R-((R-W)/CPE)", ("R1-p(R2-W1,CPE1)", [1e2, 1e5, 1e-3, 1e-9, 0.9])),
])


def get_preset_circuit_strings():
    """Return dict of preset name -> circuit string"""
    return {k: v[0] for k, v in PRESET_CIRCUITS.items()}


def get_preset_initial_guesses():
    """Return dict of preset name -> initial guess list"""
    return {k: v[1] for k, v in PRESET_CIRCUITS.items()}


def get_preset_names():
    """Return list of preset names in order"""
    return list(PRESET_CIRCUITS.keys())


# Help text for preset circuits
PRESET_CIRCUITS_HELP = """**Preset Equivalent Circuits:**

**Simple models:**
- **R**, **R-C**, **R-CPE**: Series elements
- **(R/C)**, **(R/CPE)**: Parallel elements

**Two-element models:**
- **R-(R/CPE)**: Bulk + grain boundary
- **(R/CPE)-(R/CPE)**: Two semicircles

**Three-element (solid electrolytes):**
- **(R/CPE)-(R/CPE)-CPE**: Common for ionic conductors
- **R-(R/CPE)-(R/CPE)-CPE**: With contact resistance

**Warburg (diffusion):**
- **R-W**, **(R/CPE)-W**, **Randles**

Select preset or "Custom" for manual input."""
