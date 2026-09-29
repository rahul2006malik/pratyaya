import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 80)
print("AUDIT: VERIFYING ALL 5 FIXES & 2 FOLLOW-UPS")
print("=" * 80)

# --- 1. Fix 1: export_static.py uses real cycle date & no flat fallback ---
print("\n[CHECK 1] Fix 1: Static Export Files Check")
f7_path = Path("dashboard/dist/data/forecast_region_7.json")
rq_path = Path("dashboard/dist/data/review_queue.json")

if f7_path.exists():
    f7 = json.loads(f7_path.read_text(encoding="utf-8"))
    print(f"  forecast_region_7.json: date={f7.get('init_date')}, lead={f7.get('lead_day')}, p_bust={f7.get('p_bust')}, ELT={f7.get('effective_lead_time')}")
else:
    print("  ERROR: forecast_region_7.json not found")

if rq_path.exists():
    rq = json.loads(rq_path.read_text(encoding="utf-8"))
    items = rq.get("queue", [])
    print(f"  review_queue.json: total items = {len(items)}")
    p_busts = [it.get("p_bust") for it in items[:10]]
    priorities = [it.get("priority_score") for it in items[:10]]
    print(f"  Top 10 p_bust values: {p_busts}")
    print(f"  Top 10 priority_score values: {priorities}")
    unique_p_busts = len(set(p_busts))
    print(f"  Unique p_bust count in top 10: {unique_p_busts} (is non-flat? {unique_p_busts > 1})")
else:
    print("  ERROR: review_queue.json not found")

# --- 2. Fix 1 (Investigation): Konkan & Goa p_bust across leads 1-10 ---
print("\n[CHECK 2] Konkan & Goa (Region 7) p_bust and Raw Fusion Probabilities across Leads 1-10")
sys.path.insert(0, "src")
from pratyaya.models.serving import get_model_server
s = get_model_server()

merged, _ = s._predict_cycle(date_str="2002-07-16")
konkan = merged[merged["region_id"] == "7"].sort_values("lead_day")
raw_probs = s.fusion_model.predict_proba(konkan)

# Check calibrator step for 0.15445
iso = s.calibrator.calibrator
step_found = False
for i in range(len(iso.X_thresholds_)):
    if abs(iso.y_thresholds_[i] - 0.15445) < 0.001:
        step_min = iso.X_thresholds_[max(0, i-1)]
        step_max = iso.X_thresholds_[min(len(iso.X_thresholds_)-1, i+1)]
        print(f"  Isotonic Step: raw X in [{step_min:.6f}, {step_max:.6f}] -> calibrated y = {iso.y_thresholds_[i]:.8f}")
        step_found = True
        break

for ld, raw, pb in zip(konkan["lead_day"], raw_probs, konkan["p_bust"]):
    print(f"  Lead {ld:2d}: raw_fusion = {raw:.6f} -> calibrated p_bust = {pb:.4f}")

# --- 3. Fix 2: Genuine Bust Analogues Verification against Real Data ---
print("\n[CHECK 3] Fix 2: Genuine Bust Analogues from 2000-2002 Data")
test_cases = [
    ("5", "2002-07-16", 1, "Assam & Meghalaya"),
    ("16", "2002-07-16", 1, "N M M T"),
    ("33", "2002-07-16", 1, "Coastal Karnataka"),
]

for rid, dt, ld, rname in test_cases:
    card = s.predict_region_card(rid, date=dt, lead_day=ld)
    bust_hits = [a for a in card.top_analogues if a.is_bust]
    print(f"  Region #{rid} ({rname}), Lead {ld}, Cycle {dt}:")
    print(f"    Total analogues returned: {len(card.top_analogues)}")
    print(f"    Genuine bust analogues found: {len(bust_hits)}")
    for bh in bust_hits:
        print(f"      Analogue {bh.date} (Rank #{bh.closeness_rank}, Sim {bh.weight*100:.1f}%): NWP={bh.forecast_value} mm, Obs={bh.observed_value} mm, Error={bh.forecast_value - bh.observed_value:+.1f} mm, Outcome={bh.outcome}")

# --- 4. Fix 3: Regime Discovery k in [3, 10] Ladder & Cluster Sizes ---
print("\n[CHECK 4] Fix 3: Regime Discovery k in [3, 10] Ladder & Cluster Sizes")
regimes_path = Path("models/regimes_loyo.json")
if regimes_path.exists():
    rd = json.loads(regimes_path.read_text(encoding="utf-8"))
    sil_dict = rd.get("silhouette_scores", {})
    print("  Silhouette Ladder in regimes_loyo.json:")
    for k, sc in sorted(sil_dict.items(), key=lambda x: int(x[0])):
        print(f"    k = {k}: silhouette = {sc:.4f}")
    print(f"  Optimal k recorded: {rd.get('optimal_k')}")
    clusters = rd.get("clusters", [])
    cluster_sizes = [len(c.get("member_dates", [])) for c in clusters]
    print(f"  Cluster member counts: {cluster_sizes}")
    print(f"  Smallest cluster size: {min(cluster_sizes)}")
else:
    print("  ERROR: regimes_loyo.json not found")

# Check documentation for any "true global optimum" phrases
print("\n[CHECK 4b] Grepping for 'global optimum' or 'true optimum' in docs:")
for doc in ["docs/FINDINGS.md", "docs/PRESENTATION_DECK.md", "docs/DECISIONS.md"]:
    text = Path(doc).read_text(encoding="utf-8")
    matches = [line.strip() for line in text.splitlines() if "global optimum" in line.lower() or "true optimum" in line.lower()]
    print(f"  {doc}: {len(matches)} match(es)")
    for m in matches:
        print(f"    -> {m}")

# --- 5. Fix 4: ForecasterBulletinModal.tsx checks ---
print("\n[CHECK 5] Fix 4: ForecasterBulletinModal Disclaimer & Calibrated Tiers")
modal_code = Path("dashboard/src/components/ForecasterBulletinModal.tsx").read_text(encoding="utf-8")
has_disclaimer = "PROTOTYPE RESEARCH ADVISORY" in modal_code and "operational research demonstration" in modal_code
has_calibrated_tiers = "0.25" in modal_code and "0.15" in modal_code and "0.08" in modal_code
print(f"  Has Research Prototype disclaimer: {has_disclaimer}")
print(f"  Has calibrated severity thresholds (CRITICAL >= 0.25, ORANGE >= 0.15, YELLOW >= 0.08): {has_calibrated_tiers}")

# --- 6. Fix 5: Screenshots & Live API Verification ---
print("\n[CHECK 6] Fix 5: Screenshots on Disk")
screenshots = sorted(list(Path("docs/screenshots").glob("*.png")))
print(f"  Total screenshots: {len(screenshots)}")
for sc in screenshots:
    print(f"    {sc.name} ({sc.stat().st_size / 1024:.1f} KB)")

print("\n" + "=" * 80)
print("AUDIT EXECUTION COMPLETE")
print("=" * 80)
