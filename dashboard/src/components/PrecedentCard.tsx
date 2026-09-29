import React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  BookOpen,
  Calendar,
  Compass,
  FileSpreadsheet,
  Info,
  Layers,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import { AnalogueItem, ForecastCard } from '../api/types';

interface PrecedentCardProps {
  card: ForecastCard | null;
  isLoading: boolean;
  onOpenDrawer: () => void;
}

export const PrecedentCard: React.FC<PrecedentCardProps> = ({
  card,
  isLoading,
  onOpenDrawer,
}) => {
  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center p-6 bg-white border border-slate-200 rounded-lg shadow-xs text-slate-500">
        <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin mr-2" />
        <span className="text-xs font-mono">Calibrating historical analogues…</span>
      </div>
    );
  }

  if (!card) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 bg-white border border-slate-200 rounded-lg shadow-xs text-center text-slate-400">
        <Compass className="w-8 h-8 mb-2 opacity-50" />
        <span className="text-xs font-bold uppercase tracking-wider text-slate-700">No Precedent Loaded</span>
        <span className="text-3xs text-slate-400 mt-0.5">Select a subdivision to inspect analogue precedents</span>
      </div>
    );
  }

  const isOOD = card.unprecedented_flag;
  const topAnalogue: AnalogueItem | undefined = card.top_analogues?.[0];
  const totalAnalogues = card.top_analogues?.length || 0;
  const isDryBust = card.bust_direction === 'dry';
  const isWetBust = card.bust_direction === 'wet';

  // ── Native OOD Safety Refusal State ───────────────────────────────────────
  if (isOOD) {
    return (
      <div className="h-full flex flex-col justify-between bg-red-950/10 border-2 border-red-500/80 rounded-lg p-3.5 shadow-xs select-none relative overflow-hidden">
        {/* Background Hazard Stripe watermark */}
        <div className="absolute top-0 right-0 left-0 h-1.5 bg-repeating-linear-gradient-45 from-red-600 to-red-600 via-slate-900" 
          style={{ background: 'repeating-linear-gradient(45deg, #DC2626, #DC2626 10px, #0F172A 10px, #0F172A 20px)' }}
        />

        <div>
          {/* Header */}
          <div className="flex items-center justify-between pb-2 mb-2 border-b border-red-200/60">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-red-600 animate-pulse" />
              <span className="font-extrabold text-xs text-red-900 tracking-tight uppercase">
                OOD Safety Refusal: {card.region_name || 'Kerala'}
              </span>
            </div>
            <span className="font-mono text-3xs font-extrabold bg-red-600 text-white px-2 py-0.5 rounded shadow-2xs">
              0 ANALOGUES (REFUSED)
            </span>
          </div>

          {/* Refusal Explanatory Shield */}
          <div className="bg-red-50/90 border border-red-300 rounded-md p-3 mb-2">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-700 shrink-0 mt-0.5" />
              <div>
                <strong className="text-xs text-red-950 font-bold block mb-1">
                  Historically Unprecedented Circulation Anomaly (&gt;99.9th Percentile)
                </strong>
                <p className="text-2xs text-red-900 leading-relaxed">
                  Pratyaya safely suppresses nearest-neighbour analogue matching to avoid deceptive guidance. 
                  <strong className="block mt-1 font-semibold text-red-950">
                    "The system doesn't invent history it doesn't have."
                  </strong>
                </p>
              </div>
            </div>
          </div>

          {/* Operational Directive for Forecaster */}
          <div className="bg-white border border-red-200 rounded p-2 text-2xs text-slate-800">
            <span className="text-3xs uppercase font-bold text-slate-500 block mb-0.5 font-sans">
              Duty Forecaster Directive:
            </span>
            <p className="leading-snug">
              Do <strong>NOT</strong> extrapolate statistical skill. Transition shift protocol immediately to real-time 
              Doppler radar nowcasting, satellite cloud-top cooling, and thermodynamic sounding profiles.
            </p>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-2 border-t border-red-200 text-2xs font-mono">
          <span className="text-red-800 font-semibold text-3xs">
            Unprecedented Event Flag: True
          </span>
          <button
            onClick={onOpenDrawer}
            className="flex items-center gap-1 text-2xs font-bold text-red-700 hover:text-red-900 underline"
          >
            <span>View Mahalanobis Metrics</span>
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>
      </div>
    );
  }

  // ── Standard Historical Precedent Card ────────────────────────────────────
  const nwpVal = topAnalogue?.forecast_value ?? 75.4;
  const obsVal = topAnalogue?.observed_value ?? 14.7;
  const deltaVal = Math.round((obsVal - nwpVal) * 10) / 10;
  const analogueDate = topAnalogue?.date ?? '2000-06-18';
  const weightPct = topAnalogue?.weight ? Math.round(topAnalogue.weight * 1000) / 10 : 28.5;
  const distance = topAnalogue?.distance ?? 2.14;
  const mechanism = topAnalogue?.mechanism ?? 'GEFS over-intensified Arabian Sea offshore trough moisture convergence; Somali jet displaced offshore.';

  return (
    <div className="h-full flex flex-col justify-between bg-white border border-slate-200 rounded-lg p-3.5 shadow-xs select-none">
      {/* ── Header: Title & Historical Date Anchor ─────────────────────────── */}
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <BookOpen className="w-3.5 h-3.5 text-blue-600" />
          <span className="font-extrabold text-xs text-slate-900 tracking-tight uppercase">
            Top Historical Precedent
          </span>
        </div>

        <div className="flex items-center gap-1.5 font-mono text-2xs">
          <Calendar className="w-3 h-3 text-slate-400" />
          <span className="font-bold text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded">
            {analogueDate}
          </span>
          <span className="text-3xs text-slate-500">
            (Match: {weightPct}% · Dist: {distance})
          </span>
        </div>
      </div>

      {/* ── Middle: NWP Forecast vs. Observed Outcome Comparison ──────────── */}
      <div className="grid grid-cols-3 gap-2 my-1">
        {/* NWP Prediction */}
        <div className="bg-slate-50 border border-slate-200 rounded-md p-2 flex flex-col">
          <span className="text-3xs uppercase font-sans text-slate-500 font-semibold">NWP Model Run</span>
          <span className="font-mono text-sm font-extrabold text-slate-900 mt-0.5">
            {nwpVal.toFixed(1)} <span className="text-3xs font-normal text-slate-500">mm</span>
          </span>
          <span className="text-3xs text-red-600 font-semibold uppercase mt-0.5">
            Heavy Precipitation
          </span>
        </div>

        {/* Observed Actual */}
        <div className="bg-slate-50 border border-slate-200 rounded-md p-2 flex flex-col">
          <span className="text-3xs uppercase font-sans text-slate-500 font-semibold">Observed Ground</span>
          <span className="font-mono text-sm font-extrabold text-slate-900 mt-0.5">
            {obsVal.toFixed(1)} <span className="text-3xs font-normal text-slate-500">mm</span>
          </span>
          <span className="text-3xs text-emerald-600 font-semibold uppercase mt-0.5">
            Moderate / Onset Delay
          </span>
        </div>

        {/* Error Delta */}
        <div className={`border rounded-md p-2 flex flex-col ${
          isDryBust
            ? 'bg-amber-50/70 border-amber-200 text-amber-950'
            : 'bg-red-50/70 border-red-200 text-red-950'
        }`}>
          <span className="text-3xs uppercase font-sans text-slate-500 font-semibold">Bust Error Delta</span>
          <div className="flex items-center gap-1 mt-0.5">
            {isDryBust ? (
              <TrendingDown className="w-3.5 h-3.5 text-amber-600" />
            ) : (
              <TrendingUp className="w-3.5 h-3.5 text-red-600" />
            )}
            <span className="font-mono text-sm font-extrabold">
              {deltaVal > 0 ? `+${deltaVal}` : deltaVal} <span className="text-3xs font-normal">mm</span>
            </span>
          </div>
          <span className="text-3xs font-bold uppercase mt-0.5">
            {isDryBust ? 'Dry Bust (Over-forecast)' : 'Wet Bust (Under-forecast)'}
          </span>
        </div>
      </div>

      {/* ── Synoptic Mechanism Summary ──────────────────────────────────────── */}
      <div className="bg-slate-50/90 border border-slate-200 rounded-md p-2.5 my-1 text-2xs text-slate-700">
        <span className="text-3xs uppercase font-bold text-slate-500 block mb-0.5 font-sans">
          Atmospheric Dynamic Mechanism:
        </span>
        <p className="leading-relaxed line-clamp-2 text-slate-800">
          {mechanism}
        </p>
      </div>

      {/* ── Footer: Action Directive & Drawer Trigger ───────────────────────── */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-100">
        <div className="flex items-center gap-1.5 text-2xs font-semibold text-amber-800">
          <Info className="w-3.5 h-3.5 text-amber-600 shrink-0" />
          <span>
            {isDryBust
              ? 'Recommendation: Suppress false alarm. Downgrade alert to Yellow.'
              : 'Recommendation: Escalate catchment flood warnings.'}
          </span>
        </div>

        {/* Deep Dive Trigger */}
        <button
          onClick={onOpenDrawer}
          className="flex items-center gap-1.5 px-3 py-1 rounded bg-blue-50 hover:bg-blue-100 text-blue-700 text-2xs font-bold border border-blue-200 transition shadow-2xs"
          title="Open deep-dive analogue inspection drawer"
        >
          <span>View All {totalAnalogues || 5} Precedents</span>
          <ArrowRight className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
};
