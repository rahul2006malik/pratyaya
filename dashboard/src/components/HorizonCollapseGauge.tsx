import React from 'react';
import {
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  Clock,
  Gauge,
  HelpCircle,
  Info,
  ShieldAlert,
  ShieldCheck,
  TrendingDown,
} from 'lucide-react';
import { ConfidenceBand } from '../api/types';

interface HorizonCollapseGaugeProps {
  nominalLeadDay: number;
  effectiveLeadTime: number;
  regionName: string;
  pBust: number;
  confidenceBand: ConfidenceBand;
  isOOD: boolean;
}

export const HorizonCollapseGauge: React.FC<HorizonCollapseGaugeProps> = ({
  nominalLeadDay,
  effectiveLeadTime,
  regionName,
  pBust,
  confidenceBand,
  isOOD,
}) => {
  // Delta in predictability horizon (how many days earlier did predictability collapse)
  const decayDays = Math.round((effectiveLeadTime - nominalLeadDay) * 10) / 10;
  const isSevereCollapse = effectiveLeadTime >= 8.5 || decayDays >= 2.0 || isOOD;
  const isModerateCollapse = decayDays > 0.5 && !isSevereCollapse;
  const isIntact = !isSevereCollapse && !isModerateCollapse;

  // Gauge percentage (1 to 10 days scaled 0 to 100%)
  const nominalPct = Math.min(100, Math.max(0, (nominalLeadDay / 10) * 100));
  const effectivePct = Math.min(100, Math.max(0, (effectiveLeadTime / 10) * 100));

  // Determine Forecaster Categorical Hedge Advice
  let hedgeDirective = '';
  if (isOOD) {
    hedgeDirective = 'CRITICAL OOD CIRCUIT BREAKER: Circulation vectors exceed 99.9th percentile. Physics models cannot be calibrated via historical analogues. Suppress automated guidance and initiate radar-nowcasting watch.';
  } else if (isSevereCollapse) {
    hedgeDirective = 'CLIFF-EDGE DEGRADATION: Day ' + nominalLeadDay + ' forecast has collapsed to Day ' + effectiveLeadTime.toFixed(1) + ' skill. Do NOT issue single-deterministic reservoir release advisory. Hedge against ±65mm orographic shift.';
  } else if (isModerateCollapse) {
    hedgeDirective = 'ELEVATED SYNOPTIC DRAG: Moderate skill horizon erosion (-' + decayDays + 'd). Cross-reference with regional ensemble clustering before civil disaster sign-off.';
  } else {
    hedgeDirective = 'HIGH PREDICTABILITY STABILITY: Model physics align with historical analogues. Standard IMD medium-range operational guidance is robust.';
  }

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 text-slate-100 flex flex-col justify-between shadow-md">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Gauge className={`w-4 h-4 ${isSevereCollapse ? 'text-red-400' : 'text-cyan-400'}`} />
            <h3 className="font-display font-bold text-sm tracking-tight text-white uppercase">
              Predictability Horizon Collapse
            </h3>
          </div>

          <div className="flex items-center gap-1.5">
            {isOOD ? (
              <span className="px-2 py-0.5 rounded text-3xs font-mono font-extrabold bg-purple-500/20 text-purple-300 border border-purple-500/40">
                OOD ANOMALY
              </span>
            ) : isSevereCollapse ? (
              <span className="px-2 py-0.5 rounded text-3xs font-mono font-extrabold bg-red-500/20 text-red-300 border border-red-500/40 animate-pulse">
                CLIFF EDGE REACHED
              </span>
            ) : isModerateCollapse ? (
              <span className="px-2 py-0.5 rounded text-3xs font-mono font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                HORIZON DECAY
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-3xs font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                STABLE HORIZON
              </span>
            )}
          </div>
        </div>

        {/* ── Dramatized Before vs After Horizon Numbers ───────────────────── */}
        <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3.5 mb-4">
          <div className="flex items-center justify-between gap-2">
            {/* Nominal Horizon */}
            <div className="flex-1">
              <span className="text-3xs uppercase font-mono font-semibold text-slate-400 block">
                Nominal Lead
              </span>
              <div className="text-xl font-mono font-bold text-slate-300 mt-0.5">
                Day {nominalLeadDay}
              </div>
              <span className="text-3xs text-slate-400">Scheduled cycle</span>
            </div>

            {/* Collapse Arrow */}
            <div className="flex flex-col items-center px-2">
              <span className={`text-4xs font-mono font-extrabold px-1.5 py-0.5 rounded mb-1 ${
                isSevereCollapse
                  ? 'bg-red-500/25 text-red-300 border border-red-500/40'
                  : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
              }`}>
                {isOOD ? 'OOD COLLAPSE' : isSevereCollapse ? 'DEGRADED' : 'DECAY'}
              </span>
              <ArrowRight className={`w-5 h-5 ${isSevereCollapse ? 'text-red-400' : 'text-slate-500'}`} />
              <span className="text-4xs font-mono text-slate-400 mt-0.5">
                {decayDays > 0 ? `-${decayDays}d skill` : '0d delta'}
              </span>
            </div>

            {/* Effective Predictability */}
            <div className="flex-1 text-right">
              <span className="text-3xs uppercase font-mono font-semibold text-slate-400 block">
                Effective Horizon (ELT)
              </span>
              <div className={`text-xl font-mono font-bold mt-0.5 ${
                isSevereCollapse ? 'text-red-400' : isModerateCollapse ? 'text-amber-400' : 'text-emerald-400'
              }`}>
                Day {effectiveLeadTime.toFixed(1)}
              </div>
              <span className="text-3xs font-mono text-slate-400">
                {isSevereCollapse ? 'Equivalent uncertainty' : 'Reliable window'}
              </span>
            </div>
          </div>

          {/* ── Two-Tier Linear Gauge Visualization ─────────────────────────── */}
          <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-2">
            <div className="flex justify-between text-3xs font-mono text-slate-400">
              <span>Day 1 (Nowcast)</span>
              <span className="text-amber-400 font-semibold">Day 5 (Cliff Edge)</span>
              <span>Day 10 (Noise)</span>
            </div>

            {/* Bar Track */}
            <div className="relative h-4 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              {/* Reliable Zone (Days 1 to 5) */}
              <div
                className="absolute top-0 bottom-0 left-0 bg-emerald-500/20 border-r border-emerald-500/40"
                style={{ width: '50%' }}
              />

              {/* Cliff Edge Danger Zone (Days 5 to 10) */}
              <div
                className="absolute top-0 bottom-0 right-0 hazard-hatch-pattern opacity-60"
                style={{ width: '50%' }}
              />

              {/* Nominal Lead Marker (White Pin) */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-white shadow-sm z-10"
                style={{ left: `${nominalPct}%` }}
                title={`Nominal Lead: Day ${nominalLeadDay}`}
              />

              {/* Effective Horizon Bar (Colored fill up to ELT) */}
              <div
                className={`absolute top-0 bottom-0 left-0 opacity-80 ${
                  isOOD
                    ? 'bg-purple-600'
                    : isSevereCollapse
                    ? 'bg-red-600'
                    : isModerateCollapse
                    ? 'bg-amber-500'
                    : 'bg-emerald-500'
                }`}
                style={{ width: `${effectivePct}%` }}
              />
            </div>

            <div className="flex items-center justify-between text-4xs font-mono text-slate-400">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-white inline-block" />
                Nominal D+{nominalLeadDay}
              </span>
              <span className="flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full inline-block ${
                  isSevereCollapse ? 'bg-red-500' : 'bg-emerald-500'
                }`} />
                Effective D+{effectiveLeadTime.toFixed(1)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Operational Duty Forecaster Directive ───────────────────────── */}
      <div className={`p-3 rounded-lg border text-2xs leading-relaxed ${
        isSevereCollapse
          ? 'bg-red-950/20 border-red-500/40 text-red-200'
          : 'bg-slate-950/60 border-slate-800 text-slate-300'
      }`}>
        <div className="flex items-start gap-2">
          {isSevereCollapse ? (
            <ShieldAlert className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          ) : (
            <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          )}
          <div>
            <strong className="font-semibold text-white block mb-0.5">
              Duty Forecaster Operational Directive ({regionName}):
            </strong>
            <p className="text-3xs text-slate-300 font-sans">{hedgeDirective}</p>
          </div>
        </div>
      </div>
    </div>
  );
};
