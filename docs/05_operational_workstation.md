# Operational Workstation Guide (Duty Forecaster HUD)

The Pratyaya Operational Workstation is an interactive single-page application built with React, Vite, Tailwind CSS, and D3.js designed specifically for high-stress operational weather centers.

---

## 1. Visual Workstation Architecture

```
+------------------------------------------------------------------------------------+
| [PRATYAYA] Synoptic Analogue Diagnostic Engine · Cycle: 2002-07-07 00 UTC · D+6     |
+------------------------------------------------------------------------------------+
| [NATIONAL THREAT PULSE] Active Regime: Reg-2 (Break Monsoon) | OOD Risk: 0.0% Nominal |
+------------------------------------------+-----------------------------------------+
|                                          | [OPERATIONAL TRIAGE QUEUE]              |
|                                          | Rank  Subdivision   P(Bust)  ELT   Risk    |
|       INTERACTIVE NATIONAL               |  #1   Vidarbha       48%    D9.4  DRY     |
|      SYNOPTIC CONFIDENCE MAP             |  #2   Marathwada     42%    D8.8  DRY     |
|                                          |  #3   E. MP          37%    D8.2  WET     |
|   (36 IMD Meteorological Subdivisions   |-----------------------------------------+
|    Topologically Rendered via TopoJSON)  | [METEOROLOGICAL RISK SYNTHESIS]         |
|                                          | • Break monsoon trough displacement     |
|                                          | • 23-year historical analogue precedent|
|                                          | • Mean signed error bias: -18.4 mm/day  |
+------------------------------------------+-----------------------------------------+
| [SYNOPTIC PRECEDENT STRIP]                                                         |
| Match 1: 2000-07-12 (Sim: 0.94, Bust: YES) | Match 2: 2001-07-28 (Sim: 0.91, Bust: YES)|
+------------------------------------------------------------------------------------+
| [HORIZON SCRUBBER]  <-- [D+1] [D+2] [D+3] [D+4] [D+5] [[D+6]] [D+7] [D+8] [D+9] --> |
+------------------------------------------------------------------------------------+
```

---

## 2. Core Operational Views

### A. National Synoptic Confidence Heatmap
- **Choropleth Projection**: Visualizes bust risk across all 36 subdivisions using a perceptually uniform, colorblind-safe sequential color palette.
- **Topological Precision**: Built using canonical TopoJSON boundaries to eliminate boundary artifacts while retaining responsive $60\text{ fps}$ hover inspection.

### B. Dynamic Horizon Scrubber (Days 1–10)
- Allows the forecaster to scrub lead times seamlessly.
- Visualizes **Horizon Collapse**: dynamically displays the divergence between nominal lead day $\tau$ and Effective Lead Time ($\text{ELT}$). When a Day-6 forecast behaves like a Day-9 forecast, the collapse gauge alerts the forecaster immediately.

### C. Operational Triage Queue & Precedent Strip
- Ranks the most error-prone subdivisions prioritized by a joint utility function combining bust probability, numerical dispersion, and climatological hazard exposure.
- For any active selection, the precedent strip reveals the top historical analogue days with synoptic similarity scores, observed verification, and directional error biases.

### D. A4 Operational Dispatch Bulletin Modal
- Generates a printer-ready or PDF-exportable operational briefing document (compliant with standard meteorological office dispatch protocols) with one click (`Ctrl + P` or click *Export Bulletin*).
