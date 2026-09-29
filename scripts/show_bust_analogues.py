import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, 'src')
import pandas as pd
from pratyaya.models.serving import get_model_server

s = get_model_server()

test_queries = [
    ('5', '2002-07-16', 1),
    ('16', '2002-07-16', 1),
    ('33', '2002-07-16', 1),
    ('5', '2002-07-16', 4),
]

for rid, dt, ld in test_queries:
    card = s.predict_region_card(rid, date=dt, lead_day=ld)
    print("=" * 115)
    print(f"FORECAST CARD: Subdivision #{card.region_id} ({card.region_name}) | Horizon: Lead Day {ld} | Cycle: {dt}")
    print(f"P(Bust): {card.p_bust*100:.1f}% | ELT: Day {card.effective_lead_time:.1f} | Direction: {card.bust_direction}")
    print("-" * 115)
    print("Historical Analogues Table (Rendered through ForecastCard.tsx logic):")
    header = f"{'Rank/Date':<16} | {'Sim':<6} | {'NWP Fcst':<10} | {'Observed':<10} | {'Delta':<10} | {'Outcome Badge':<14} | Driver"
    print(header)
    print("-" * 115)
    for idx, a in enumerate(card.top_analogues):
        hasNwp = a.forecast_value is not None
        hasObs = a.observed_value is not None
        nwpFc = a.forecast_value if hasNwp else None
        actualObs = a.observed_value if hasObs else None
        delta = (actualObs - nwpFc) if (hasNwp and hasObs) else None
        closenessPct = f"{a.weight * 100:.1f}%"
        
        bdir = None
        if a.bust_direction and str(a.bust_direction).lower() != "none":
            bdir = str(a.bust_direction).split(".")[-1].upper()
        elif a.outcome and "wet" in a.outcome.lower():
            bdir = "WET"
        elif a.outcome and "dry" in a.outcome.lower():
            bdir = "DRY"
            
        badge = f"BUST ({bdir})" if a.is_bust else "NO BUST"
        nwp_str = f"{nwpFc:.1f} mm" if nwpFc is not None else "—"
        obs_str = f"{actualObs:.1f} mm" if actualObs is not None else "—"
        delta_str = f"{delta:+.1f} mm" if delta is not None else "—"
        date_str = f"#{idx+1} {a.date}"
        mech_str = a.mechanism if a.mechanism else "Synoptic circulation diagnostic in progress"
        
        marker = " <== [GENUINE BUST CASE]" if a.is_bust else ""
        print(f"{date_str:<16} | {closenessPct:<6} | {nwp_str:<10} | {obs_str:<10} | {delta_str:<10} | {badge:<14} | {mech_str[:38]}...{marker}")
    print()
