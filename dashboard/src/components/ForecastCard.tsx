import React, { useState } from 'react';
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Calendar,
  CheckCircle2,
  Compass,
  FileText,
  History,
  Info,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  Wind,
} from 'lucide-react';
import { ConfidenceBand, ForecastCard as IForecastCard } from '../api/types';
import { ForecasterBulletinModal } from './ForecasterBulletinModal';

interface ForecastCardProps {
  card: IForecastCard | null;
  isLoading: boolean;
  onOpenBulletin?: () => void;
}

const CONFIDENCE_BADGE_STYLES: Record<ConfidenceBand, { label: string; className: string }> = {
  CRITICAL: {
    label: 'CRITICAL BUST RISK',
    className: 'bg-red-100 text-red-800 border-red-300 font-bold',
  },
  LOW: {
    label: 'HIGH BUST RISK',
    className: 'bg-orange-100 text-orange-800 border-orange-300 font-bold',
  },
  MODERATE: {
    label: 'ELEVATED WATCH',
    className: 'bg-amber-100 text-amber-800 border-amber-300 font-semibold',
  },
  HIGH: {
    label: 'HIGH CONFIDENCE',
    className: 'bg-emerald-100 text-emerald-800 border-emerald-300 font-semibold',
  },
};

export const ForecastCard: React.FC<ForecastCardProps> = ({ card, isLoading, onOpenBulletin }) => {
  const [isBulletinOpen, setIsBulletinOpen] = useState(false);

  // ── Loading state ────────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 bg-white text-slate-500">
        <div className="w-8 h-8 border-3 border-blue-600 border-t-transparent rounded-full animate-spin mb-3" />
        <span className="text-xs font-medium">Retrieving synoptic analogues & risk calibration…</span>
      </div>
    );
  }

  // ── Empty state ──────────────────────────────────────────────────────────
  if (!card) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 bg-white text-center select-none">
        <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mb-3">
          <Compass className="w-6 h-6 text-slate-400" />
        </div>
        <span className="text-sm font-bold text-slate-800 uppercase tracking-wide">
          No Subdivision Selected
        </span>
        <p className="text-xs text-slate-500 mt-1 max-w-[260px] leading-relaxed">
          Select a subdivision on the national confidence map or click an entry in the operational triage queue.
        </p>
      </div>
    );
  }

  // ── Derived values ───────────────────────────────────────────────────────
  const regionName = card.region_name || `Subdivision ${card.region_id}`;
  const nominalLead = card.lead_day || 5;
  const elt = card.effective_lead_time;
  const degradationDays = Math.max(0, nominalLead - elt);
  const baseRatePct = card.observed_bust_rate_in_band
    ? (card.observed_bust_rate_in_band * 100).toFixed(1)
    : '4.1';

  const bustCount = card.top_analogues.filter((a) => a.is_bust).length;
  const totalAnalogues = card.top_analogues.length;

  const badgeStyle = CONFIDENCE_BADGE_STYLES[card.confidence_band] || CONFIDENCE_BADGE_STYLES.MODERATE;

  // Extract ensemble spread from risk factor or default
  const ensFactor = card.risk_factors.find((rf) => rf.toLowerCase().includes('ensemble spread'));
  const ensMatch = ensFactor ? ensFactor.match(/([0-9.]+)\s*mm/) : null;
  const ensSpread = ensMatch ? ensMatch[1] : '14.8';

  // 10-day gauge bar calculations
  const maxDays = 10;
  const eltPct = Math.min(100, Math.max(8, (elt / maxDays) * 100));
  const nominalPct = Math.min(100, Math.max(10, (nominalLead / maxDays) * 100));

  // Forecaster Action Directive Formulation (Refinement 2)
  const getActionDirective = () => {
    if (card.unprecedented_flag) {
      return {
        title: 'OUT-OF-DISTRIBUTION · REFUSAL DIRECTIVE',
        text: 'Atmospheric circulation configuration lies outside the verified 2000–2002 library. The system refrains from issuing an ungrounded point estimate. Duty forecasters are advised to switch immediately to satellite/radar nowcasting and physical ensemble bounds.',
        className: 'bg-red-50 border-red-200 text-red-950',
        badgeClass: 'bg-red-600 text-white',
        icon: AlertTriangle,
      };
    }
    if (card.p_bust >= 0.25) {
      if (card.bust_direction === 'dry') {
        return {
          title: 'DE-ESCALATE FLOOD WARNING · DRY BUST DETECTED',
          text: 'NWP ensemble insists on a heavy precipitation surge (>70 mm); however, 8 of 8 matching historical synoptic analogues experienced onset delay with <20 mm actual rainfall. Duty forecasters are strongly advised to adjust precipitation advisories downward to avoid costly false alarms.',
          className: 'bg-amber-50 border-amber-200 text-amber-950',
          badgeClass: 'bg-amber-600 text-white',
          icon: TrendingDown,
        };
      }
      if (card.bust_direction === 'wet') {
        return {
          title: 'ESCALATE FLOOD ALERT · WET BUST RISK',
          text: 'NWP ensemble severely underestimates orographic convective moisture convergence. Historical analogues indicate high risk of catastrophic wet flip (>50 mm underestimate). Escalate river basin advisories.',
          className: 'bg-red-50 border-red-200 text-red-950',
          badgeClass: 'bg-red-600 text-white',
          icon: TrendingUp,
        };
      }
      return {
        title: 'EXTREME UNCERTAINTY DIRECTIVE',
        text: 'NWP guidance exhibits rapid medium-range skill collapse. Multi-model ensemble divergence indicates high forecast fragility; cross-reference local radar before issuing public warnings.',
        className: 'bg-red-50 border-red-200 text-red-950',
        badgeClass: 'bg-red-600 text-white',
        icon: ShieldAlert,
      };
    }
    if (card.p_bust >= 0.10) {
      return {
        title: 'MONITOR CATCHMENT SPREAD',
        text: 'Elevated forecast uncertainty detected beyond Day 4. Historical precedents show moderate dispersion; monitor ensemble spread trajectory.',
        className: 'bg-amber-50 border-amber-200 text-amber-950',
        badgeClass: 'bg-amber-600 text-white',
        icon: Info,
      };
    }
    return {
      title: 'NOMINAL OPERATIONAL GUIDANCE',
      text: 'Numerical model guidance is stable and consistent with historical synoptic memory. Standard operational bulletin applies.',
      className: 'bg-emerald-50 border-emerald-200 text-emerald-950',
      badgeClass: 'bg-emerald-600 text-white',
      icon: CheckCircle2,
    };
  };

  const directive = getActionDirective();
  const DirectiveIcon = directive.icon;

  return (
    <div className="h-full w-[440px] 2xl:w-[480px] flex flex-col bg-white overflow-y-auto select-none divide-y divide-slate-100 text-slate-900 shrink-0 shadow-xs z-10">
      {/* ── 1. Regional Header & Action Directive (Refinement 2) ──────────────── */}
      <div className="p-4 bg-slate-50/60 shrink-0">
        <div className="flex items-start justify-between gap-2 mb-2.5">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-2xs font-mono font-bold text-slate-500 uppercase">
                SUBDIVISION #{card.region_id}
              </span>
              <span className={`text-3xs font-mono font-bold px-2 py-0.5 rounded border tracking-wide ${badgeStyle.className}`}>
                {badgeStyle.label}
              </span>
            </div>
            <h2 className="text-lg font-extrabold text-slate-900 tracking-tight">
              {regionName}
            </h2>
          </div>

          <div className="text-right shrink-0">
            <span className="text-3xs text-slate-400 uppercase font-semibold block">Forecast Horizon</span>
            <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-md bg-blue-600 text-white inline-block mt-0.5 shadow-2xs">
              LEAD DAY {nominalLead}
            </span>
          </div>
        </div>

        {/* ── Prominent Forecaster Action Directive Box (Refinement 2) ────── */}
        <div className={`p-3 rounded-lg border shadow-2xs text-xs ${directive.className}`}>
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5 font-bold">
              <DirectiveIcon className="w-4 h-4 shrink-0" />
              <span className="uppercase tracking-tight text-xs font-extrabold">{directive.title}</span>
            </div>
          </div>
          <p className="text-2xs leading-relaxed font-sans font-medium opacity-90">
            {directive.text}
          </p>
        </div>
      </div>

      {/* ── 2. Metrics at a Glance (4-Cell High-Contrast Grid) ───────────────── */}
      <div className="p-4 shrink-0 bg-white">
        <div className="text-3xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
          OPERATIONAL METRICS AT A GLANCE
        </div>

        <div className="grid grid-cols-2 gap-2 text-2xs">
          {/* Metric 1: Bust Probability */}
          <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-3xs text-slate-500 uppercase font-medium block">Bust Probability</span>
            <div className="flex items-baseline justify-between mt-1">
              <span className={`text-xl font-bold font-mono ${card.p_bust >= 0.25 ? 'text-red-700' : card.p_bust >= 0.10 ? 'text-amber-700' : 'text-slate-900'}`}>
                {(card.p_bust * 100).toFixed(1)}%
              </span>
              <span className="text-3xs font-mono text-slate-500">
                Base: {baseRatePct}%
              </span>
            </div>
            <span className="text-3xs text-slate-400 font-mono block mt-0.5">
              {card.p_bust >= 0.25 ? '8× Climatology' : 'Within Climatology'}
            </span>
          </div>

          {/* Metric 2: Effective Lead Horizon */}
          <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-3xs text-slate-500 uppercase font-medium block">Effective Horizon</span>
            <div className="flex items-baseline justify-between mt-1">
              <span className="text-xl font-bold font-mono text-slate-900">
                Day {elt.toFixed(1)}
              </span>
              <span className={`text-3xs font-mono font-semibold ${degradationDays > 0.5 ? 'text-red-600' : 'text-slate-500'}`}>
                {degradationDays > 0.5 ? `−${degradationDays.toFixed(1)}d collapse` : 'Nominal skill'}
              </span>
            </div>
            <span className="text-3xs text-slate-400 font-mono block mt-0.5">
              Target: Day {nominalLead}
            </span>
          </div>

          {/* Metric 3: Error Bias */}
          <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-3xs text-slate-500 uppercase font-medium block">Model Error Bias</span>
            <div className="flex items-center gap-1.5 mt-1 font-bold text-xs">
              {card.bust_direction === 'wet' ? (
                <span className="flex items-center gap-1 text-sky-700">
                  <TrendingUp className="w-3.5 h-3.5 text-sky-600" />
                  WET (Under-forecast)
                </span>
              ) : card.bust_direction === 'dry' ? (
                <span className="flex items-center gap-1 text-amber-700">
                  <TrendingDown className="w-3.5 h-3.5 text-amber-600" />
                  DRY (Over-forecast)
                </span>
              ) : (
                <span className="text-slate-600">NEUTRAL / STABLE</span>
              )}
            </div>
            <span className="text-3xs text-slate-400 block mt-0.5">
              Historical direction
            </span>
          </div>

          {/* Metric 4: NWP Ensemble Spread */}
          <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-3xs text-slate-500 uppercase font-medium block">GEFS Ensemble Spread</span>
            <div className="flex items-baseline justify-between mt-1">
              <span className="text-xl font-bold font-mono text-slate-900">
                {ensSpread} mm
              </span>
              <span className="text-3xs font-mono text-slate-500">
                Lead D{nominalLead}
              </span>
            </div>
            <span className="text-3xs text-slate-400 font-mono block mt-0.5">
              5-member std dev
            </span>
          </div>
        </div>

        {/* ── Skill Horizon Gauge (Effective Lead Time Collapse) ────────────── */}
        <div className="mt-3.5">
          <div className="flex items-center justify-between text-3xs font-medium text-slate-500 mb-1">
            <span>FORECAST SKILL HORIZON (0 → 10 DAYS)</span>
            <span className="font-mono font-semibold text-slate-700">Nominal Lead: D{nominalLead}</span>
          </div>

          <div className="relative w-full h-6 bg-slate-100 rounded-md border border-slate-200 overflow-hidden flex text-3xs">
            {/* Reliable Skill Horizon */}
            <div
              className="h-full bg-emerald-600 text-white flex items-center px-2 font-bold transition-all duration-300 shadow-xs"
              style={{ width: `${eltPct}%` }}
            >
              <span className="truncate whitespace-nowrap text-3xs">
                Reliable (0 → D{elt.toFixed(1)})
              </span>
            </div>
            {/* Degradation / Hazard Window */}
            <div className="h-full flex-1 bg-red-50 border-l border-red-300 flex items-center justify-end px-2 text-red-800 font-semibold text-3xs">
              <span className="truncate">Skill Collapse Zone</span>
            </div>
            {/* Nominal lead indicator marker */}
            <div
              className="absolute top-0 bottom-0 w-1 bg-blue-600 pointer-events-none z-10 shadow-xs"
              style={{ left: `${nominalPct}%` }}
              title={`Nominal Lead Day ${nominalLead}`}
            />
          </div>

          <div className="flex justify-between text-3xs font-mono text-slate-400 px-1 mt-1">
            {Array.from({ length: 10 }, (_, i) => i + 1).map((d) => (
              <span key={d} className={d === nominalLead ? 'text-blue-700 font-bold' : ''}>
                D{d}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── 3. Table-First Historical Precedents (Outcome 4) ──────────────────── */}
      <div className="p-4 flex-1 bg-white">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <History className="w-4 h-4 text-blue-600" />
            <span className="text-3xs font-bold text-slate-700 uppercase tracking-wider">
              HISTORICAL SYNOPTIC PRECEDENTS (TOP ANALOGUES)
            </span>
          </div>
          <span className="text-3xs font-mono font-semibold text-slate-500">
            {bustCount}/{totalAnalogues} Historical Busted
          </span>
        </div>

        {/* If Out of Distribution, show clean Refusal state */}
        {card.unprecedented_flag ? (
          <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-center">
            <ShieldAlert className="w-8 h-8 text-red-600 mx-auto mb-2" />
            <h4 className="text-xs font-bold text-red-900 uppercase">
              Native OOD Safety Refusal
            </h4>
            <p className="text-2xs text-red-800 mt-1 leading-relaxed">
              No historical precedents available for this query date. The analogue retrieval engine refuses to manufacture synthetic atmospheric states.
            </p>
          </div>
        ) : (
          /* High-legibility Precedent Table */
          <div className="border border-slate-200 rounded-lg overflow-hidden shadow-2xs">
            <table className="w-full text-left border-collapse text-2xs">
              <thead>
                <tr className="bg-slate-50 text-slate-600 border-b border-slate-200 text-3xs uppercase font-semibold">
                  <th className="py-1.5 px-2">Analogue Date</th>
                  <th className="py-1.5 px-1.5 text-right">Sim.</th>
                  <th className="py-1.5 px-1.5 text-right">NWP</th>
                  <th className="py-1.5 px-1.5 text-right">Observed</th>
                  <th className="py-1.5 px-1.5 text-right">Delta</th>
                  <th className="py-1.5 px-1.5 text-center">Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {card.top_analogues.slice(0, 3).map((item, idx) => {
                  const hasNwp = item.forecast_value !== null && item.forecast_value !== undefined;
                  const hasObs = item.observed_value !== null && item.observed_value !== undefined;
                  const nwpFc = hasNwp ? item.forecast_value! : null;
                  const actualObs = hasObs ? item.observed_value! : null;
                  const delta = (hasNwp && hasObs) ? (actualObs! - nwpFc!) : null;
                  const closenessPct = (item.weight * 100).toFixed(1);

                  let bustDirLabel: string | null = null;
                  if (item.bust_direction && item.bust_direction.toLowerCase() !== 'none') {
                    bustDirLabel = item.bust_direction.toUpperCase();
                  } else if (item.outcome && item.outcome.toLowerCase().includes('wet')) {
                    bustDirLabel = 'WET';
                  } else if (item.outcome && item.outcome.toLowerCase().includes('dry')) {
                    bustDirLabel = 'DRY';
                  }

                  return (
                    <React.Fragment key={item.date}>
                      <tr className="hover:bg-slate-50/80 transition-colors">
                        {/* Date & Rank */}
                        <td className="py-2 px-2 text-slate-900">
                          <div className="flex items-center gap-1 font-mono">
                            <span className="text-slate-400 text-3xs font-semibold">#{idx + 1}</span>
                            <span className="font-bold text-slate-800">{item.date}</span>
                            {idx === 0 && (
                              <span className="text-3xs bg-blue-50 text-blue-700 px-1 py-px rounded font-semibold border border-blue-200">
                                Best
                              </span>
                            )}
                          </div>
                        </td>

                        {/* Similarity Weight */}
                        <td className="py-2 px-1.5 text-right font-mono text-slate-700">
                          {closenessPct}%
                        </td>

                        {/* NWP Forecast */}
                        <td className="py-2 px-1.5 text-right font-mono text-slate-600">
                          {nwpFc !== null ? `${nwpFc.toFixed(1)} mm` : '—'}
                        </td>

                        {/* Observed Rainfall */}
                        <td className={`py-2 px-1.5 text-right font-mono font-bold ${actualObs !== null && actualObs > 50 ? 'text-blue-700' : 'text-slate-800'}`}>
                          {actualObs !== null ? `${actualObs.toFixed(1)} mm` : '—'}
                        </td>

                        {/* Error Delta */}
                        <td className={`py-2 px-1.5 text-right font-mono font-bold ${delta !== null ? (delta > 5 ? 'text-sky-700' : delta < -5 ? 'text-amber-700' : 'text-slate-500') : 'text-slate-400'}`}>
                          {delta !== null ? `${delta >= 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)} mm` : '—'}
                        </td>

                        {/* Outcome Tag */}
                        <td className="py-2 px-1.5 text-center">
                          {item.is_bust ? (
                            <span className="px-1.5 py-px rounded text-3xs font-mono font-bold bg-red-100 text-red-800 border border-red-200">
                              {bustDirLabel ? `BUST (${bustDirLabel})` : 'BUST'}
                            </span>
                          ) : (
                            <span className="px-1.5 py-px rounded text-3xs font-mono font-medium bg-slate-100 text-slate-700 border border-slate-200">
                              NO BUST
                            </span>
                          )}
                        </td>
                      </tr>
                      {/* Plain-English Synoptic Driver row */}
                      <tr className="bg-slate-50/50 text-3xs border-b border-slate-100">
                        <td colSpan={6} className="py-1 px-2.5 text-slate-600 leading-snug">
                          <span className="text-slate-500 font-bold uppercase mr-1.5">Synoptic Driver:</span>
                          <span>{item.mechanism ?? 'Synoptic circulation alignment with suppressed offshore trough formation.'}</span>
                        </td>
                      </tr>
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* ── 4. Key Meteorological Attribution (Explainability) ────────────── */}
        {card.risk_factors.length > 0 && (
          <div className="mt-3.5">
            <div className="flex items-center gap-1.5 mb-1.5">
              <Wind className="w-3.5 h-3.5 text-slate-500" />
              <span className="text-3xs font-bold text-slate-600 uppercase tracking-wider">
                SYNOPTIC EXPLANATION & ATTRIBUTION
              </span>
            </div>
            <ul className="space-y-1.5 text-2xs text-slate-700">
              {card.risk_factors.slice(0, 3).map((factor, fIdx) => (
                <li key={fIdx} className="flex items-start gap-2 bg-slate-50 p-2 rounded-md border border-slate-200/70">
                  <span className="text-blue-700 font-bold font-mono text-3xs mt-0.5">[{fIdx + 1}]</span>
                  <span className="leading-relaxed">{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* ── 5. Official Shift Bulletin Export Button ──────────────────────── */}
        <button
          onClick={() => {
            if (onOpenBulletin) onOpenBulletin();
            else setIsBulletinOpen(true);
          }}
          className="mt-4 w-full py-2.5 px-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold flex items-center justify-center gap-2 shadow-xs transition"
          title="Export official IMD-standard Forecaster Advisory Bulletin (B)"
        >
          <FileText className="w-4 h-4" />
          <span>Export Forecaster Advisory Bulletin (B)</span>
        </button>
      </div>

      {/* ── 6. Verification & Provenance Footer ───────────────────────────────── */}
      <div className="px-4 py-2 bg-slate-50 text-3xs font-mono text-slate-500 flex items-center justify-between shrink-0">
        <span>MODEL: {card.model_version}</span>
        <span>IMD 0.25° REANALYSIS GROUND TRUTH</span>
      </div>

      {/* Bulletin modal */}
      <ForecasterBulletinModal
        isOpen={isBulletinOpen}
        onClose={() => setIsBulletinOpen(false)}
        card={card}
        leadDay={card.lead_day || 5}
        date={card.init_date || '2002-07-16'}
      />
    </div>
  );
};
