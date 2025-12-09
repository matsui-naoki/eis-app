# EIS Analyzer

**Electrochemical Impedance Spectroscopy Analysis Web Application**

A browser-based EIS analysis tool built with Python and Streamlit.

**Live Demo:** https://eis-analyzer.streamlit.app

---

## Features

- **Multiple Analysis Modes**: Nyquist, Arrhenius, and Mapping analysis
- **Interactive Visualization**: Nyquist plots, Bode plots, Arrhenius plots with Plotly
- **Advanced Circuit Fitting**: Standard fit, Bayesian optimization, Monte Carlo, Auto-fit, Batch processing
- **Mapping Analysis**: 1D, 2D heatmaps, and ternary diagrams
- **Session Management**: Save and load complete analysis sessions
- **Export Options**: Igor Pro (.itx), CSV, JSON

---

## Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Supported File Formats

| Format | Extension | Instrument |
|--------|-----------|------------|
| BioLogic EC-Lab | .mpt | BioLogic |
| ZPlot | .z | Scribner |
| Gamry | .dta | Gamry Instruments |
| Keysight | .par | Keysight |
| Text/CSV | .txt, .csv, .dat | Generic |

---

## How to Use

### 1. Sample Information

Enter sample parameters in the sidebar:
- **Sample Name**: Label for your sample
- **Thickness**: Sample thickness in cm
- **Diameter/Area**: Choose input mode and enter electrode dimensions

### 2. Load Data

Click **Upload** in the sidebar to load EIS data files. Multiple files can be loaded simultaneously.

### 3. Analysis Modes

Select the analysis mode from the sidebar:

#### Nyquist Mode
- View Nyquist (Z' vs -Z'') and Bode plots
- Perform circuit fitting
- Analyze individual or multiple files

#### Arrhenius Mode
- Temperature-dependent conductivity analysis
- Plot log(σT) vs 1000/T
- Calculate activation energy from slope

#### Mapping Mode
- **1D**: Position vs conductivity
- **2D**: Spatial heatmap with interpolation
- **Ternary**: Three-component composition diagram

### 4. Circuit Fitting

#### Select Circuit Model
Choose from 25+ preset circuits or enter a custom circuit string:
- `p(R1,CPE1)-CPE2` - Single semicircle with spike
- `p(R1,CPE1)-p(R2,CPE2)-CPE3` - Two semicircles with spike
- `R1-p(R2-W1,CPE1)` - Randles circuit

#### Fitting Methods

| Method | Description |
|--------|-------------|
| **Fit** | Standard least-squares fitting |
| **Bayesian** | Global optimization with Optuna |
| **MC Fit** | Monte Carlo with noise injection |
| **Auto Fit** | Bayesian + Monte Carlo combined |
| **Batch** | Process multiple files sequentially |

#### Weighting Options
- **None**: Equal weight for all points
- **Proportional**: Weight by |Z| (emphasizes low frequency)
- **Modulus**: Weight by 1/|Z| (emphasizes high frequency)

### 5. Data Range Controls

| Control | Purpose |
|---------|---------|
| **Display Range** | Data points shown in plots |
| **Fitting Range** | Data points used for fitting |
| **Delete Points** | Exclude specific indices (e.g., `0,5,10` or `5:10`) |

**Apply Mode**:
- **Global**: Same range for all files
- **Individual**: Per-file range settings

### 6. Plot Customization

In the **Settings** tab:
- Marker: color, symbol, size, alpha, edge style
- Fit line: color, width
- Axis: label font size, tick font size
- Legend: position, font size, display mode
- Zero lines: toggle visibility

### 7. Export Data

#### Session Save/Load
- **Save**: Export complete session to JSON
- **Load**: Restore previous session

#### Igor Pro Export
- Export data and fits to .itx format
- Includes procedure file (.ipf) for axis formatting

#### CSV Export
- Download summary tables
- Export individual file data

---

## Circuit Notation

Uses [impedance.py](https://impedancepy.readthedocs.io/) notation:

| Symbol | Element | Parameters |
|--------|---------|------------|
| R | Resistor | R (Ω) |
| C | Capacitor | C (F) |
| CPE | Constant Phase Element | Q (F·s^(n-1)), n |
| W | Warburg | Aw |
| L | Inductor | L (H) |

**Operators**:
- `-` : Series connection
- `p(A,B)` : Parallel connection

**Examples**:
```
R1-CPE1                    # R in series with CPE
p(R1,CPE1)                 # R parallel with CPE
p(R1,CPE1)-p(R2,CPE2)      # Two RC parallel circuits in series
R1-p(R2-W1,CPE1)           # Randles circuit
```

---

## Analysis Results

### Multipoint Table

Summary of all fitted data:
- Temperature (K)
- Fitted parameters (R, CPE_Q, CPE_α)
- Conductivity (σ, log(σ), σT, log(σT))
- Effective capacitance (C_eff)
- RMSPE quality metric

### RMSPE Interpretation

| RMSPE | Quality |
|-------|---------|
| < 1% | Excellent |
| 1-3% | Good |
| 3-5% | Acceptable |
| > 5% | Poor |

---

## Keyboard Shortcuts

Standard Plotly interactions:
- **Scroll**: Zoom in/out
- **Drag**: Pan
- **Double-click**: Reset view
- **Hover**: Show data tooltip

---

## Tips

1. **Initial Values**: Good initial guesses improve fitting convergence
2. **Fitting Range**: Exclude noisy high/low frequency regions
3. **Weight Method**: Use "Proportional" for solid electrolytes
4. **Bayesian Fit**: Increase trials for complex circuits
5. **Batch Fit**: Enable "Use previous result" for sequential temperature data

---

## Requirements

- Python 3.8+
- streamlit
- numpy
- pandas
- scipy
- plotly
- impedance

Optional:
- optuna (Bayesian optimization)

---

## License

For research and educational purposes.

---

## Acknowledgments

- [impedance.py](https://impedancepy.readthedocs.io/) - Circuit modeling
- [Streamlit](https://streamlit.io/) - Web framework
- [Plotly](https://plotly.com/python/) - Interactive plots
- [Optuna](https://optuna.org/) - Bayesian optimization

---

**Created with Claude Code**
