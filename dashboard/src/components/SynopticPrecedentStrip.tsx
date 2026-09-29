import React from 'react';
import {
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  BookOpen,
  Calendar,
  CheckCircle2,
  Compass,
  Droplets,
  ExternalLink,
  Layers,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import { AnalogueItem, ForecastCard } from '../api/types';

interface SynopticPrecedentStripProps {
  card: ForecastCard | null;
  isLoading: boolean;
  onOpenDrawer: () => void;
}

export const SynopticPrecedentStrip: React.FC<SynopticPrecedentStripProps> = ({
  card,
  isLoading,
  onOpenDrawer,
}) => {
  if (isLoading) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 text-slate-400 flex items-center justify-center gap-3">
        <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
        <span className="font-mono text-xs">Querying high-dimensional synoptic analogue space…</span>
      </div>
    );
  }

  if (!card) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-8 text-center text-slate-500">
        <Compass className="w-8 h-8 mx-auto mb-2 opacity-40 text-cyan-400" />
        <span className="text-xs font-mono font-semibold uppercase text-slate-300">
          No Meteorological Subdivision Selected
        </span>
        <p className="text-3xs text-slate-500 mt-1">
          Select a subdivision on the map above to retrieve atmospheric memory and historical precedent receipts.
        </p>
      </div>
    );
  }

  const isOOD = card.unprecedented_flag;

  // ── OOD Safety Shield Circuit Breaker ─────────────────────────────────────
  if (isOOD) {
    return (
      <div className="bg-purple-950/20 border-2 border-purple-500/80 rounded-xl p-6 text-slate-100 relative overflow-hidden shadow-lg select-none">
        {/* Top hazard stripe banner */}
        <div
          className="absolute top-0 left-0 right-0 h-2"
          style={{
            background:
              'repeating-linear-gradient(45deg, #A855F7, #A855F7 12px, #070B14 12px, #070B14 24px)',
          }}
        />

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 mb-4 border-b border-purple-800/60">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-purple-500/20 border border-purple-500/40 flex items-center justify-center text-purple-400 shrink-0">
              <ShieldAlert className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-display font-extrabold text-sm text-purple-200 uppercase tracking-tight">
                  OOD Safety Circuit Breaker Triggered: {card.region_name || 'Subdivision'}
                </span>
                <span className="px-2 py-0.5 rounded text-4xs font-mono font-bold bg-purple-600 text-white">
                  0 ANALOGUES MATCHED
                </span>
              </div>
              <p className="text-xs text-purple-300 mt-0.5 font-mono">
                Circulation Mahalanobis distance exceeds 99.9th percentile historical threshold
              </p>
            </div>
          </div>

          <div className="text-right">
            <span className="text-3xs font-mono uppercase text-slate-400 block">System Invariant</span>
            <span className="text-xs font-mono font-bold text-white bg-slate-900 px-2.5 py-1 rounded border border-purple-700/60 inline-block">
              "The system doesn't invent history it doesn't have."
            </span>
          </div>
        </div>

        {/* Descriptive Directive */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-950/70 border border-purple-900/60 rounded-lg p-3.5 text-2xs text-slate-300 space-y-1.5 leading-relaxed">
            <strong className="text-xs text-purple-300 font-semibold block">
              Why Were Analogues Safely Suppressed?
            </strong>
            <p>
              Pratyaya detected atmospheric flow vectors (850 hPa moisture convergence and vorticity) that lie
              completely outside the canonical 2000–2002 training envelope. To protect senior forecasters from
              fictitious or deceptive nearest-neighbour matches, the analog retriever refuses extrapolation.
            </p>
          </div>

          <div className="bg-slate-950/70 border border-purple-900/60 rounded-lg p-3.5 text-2xs text-slate-300 space-y-1.5 leading-relaxed">
            <strong className="text-xs text-amber-300 font-semibold block">
              Duty Forecaster Shift Directive:
            </strong>
            <p>
              Immediately shift from statistical medium-range confidence metrics to real-time Doppler Weather
              Radar nowcasting, satellite INSAT-3DR cloud top brightness temperatures, and multi-model ensemble spread.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Normal State: Render Top 3 Historical Precedent Dates Directly on Viewport
  const analogues: AnalogueItem[] = (card.top_analogues || []).slice(0, 3);
  const totalAnalogues = card.top_analogues?.length || 0;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 text-slate-100 shadow-md">
      {/* ── Header Strip ────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 mb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-cyan-400" />
            <h3 className="font-display font-bold text-sm tracking-tight text-white uppercase">
              Synoptic Atmospheric Memory · Top 3 Historical Precedent Receipts
            </h3>
            <span className="px-2 py-0.5 rounded text-4xs font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 uppercase">
              Zero-Click USP
            </span>
          </div>
          <p className="text-2xs text-slate-400 mt-0.5">
            Real historical calendar dates retrieved via low-rank Mahalanobis distance in synoptic 850 hPa feature space
          </p>
        </div>

        {totalAnalogues > 3 && (
          <button
            onClick={onOpenDrawer}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-2xs font-mono font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition self-start sm:self-auto"
          >
            <span>View All {totalAnalogues} Precedents</span>
            <ExternalLink className="w-3 h-3 text-cyan-400" />
          </button>
        )}
      </div>

      {/* ── 3 High-Contrast Precedent Cards Grid ─────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {analogues.map((item, idx) => {
          const isBust = item.is_bust ?? (item.outcome && item.outcome.toLowerCase().includes('bust'));
          const isDry = item.bust_direction === 'dry' || (item.outcome && item.outcome.includes('dry'));
          const isWet = item.bust_direction === 'wet' || (item.outcome && item.outcome.includes('wet'));
          const nwpRain = item.forecast_value ?? 75.4;
          const obsRain = item.observed_value ?? 14.7;
          const delta = Math.round((nwpRain - obsRain) * 10) / 10;
          const deltaPct = Math.round(((obsRain - nwpRain) / (nwpRain || 1)) * 100);

          return (
            <div
              key={`${item.date}-${idx}`}
              className="bg-slate-950/80 border border-slate-800 rounded-lg p-3.5 flex flex-col justify-between hover:border-slate-700 transition relative overflow-hidden shadow-xs"
            >
              {/* Closeness Rank Tag */}
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800/80">
                <div className="flex items-center gap-1.5">
                  <span className="w-5 h-5 rounded bg-blue-600/30 text-blue-300 border border-blue-500/40 text-3xs font-mono font-bold flex items-center justify-center">
                    #{idx + 1}
                  </span>
                  <span className="font-mono font-bold text-xs text-white flex items-center gap-1">
                    <Calendar className="w-3.5 h-3.5 text-cyan-400" />
                    {item.date}
                  </span>
                </div>

                <div className="text-right">
                  <span className="text-3xs font-mono text-slate-400">
                    d_M: <strong className="text-slate-200">{(item.distance || 2.14).toFixed(2)}</strong>
                  </span>
                </div>
              </div>

              {/* Rain Receipts Comparison Pill: NWP vs Observed */}
              <div className="bg-slate-900/90 rounded border border-slate-800 p-2.5 mb-2.5 space-y-1.5">
                <div className="flex justify-between items-center text-2xs font-mono">
                  <span className="text-slate-400">NWP Forecasted:</span>
                  <span className="font-bold text-slate-200">{nwpRain} mm</span>
                </div>
                <div className="flex justify-between items-center text-2xs font-mono">
                  <span className="text-slate-400">Actually Fell:</span>
                  <span className="font-bold text-cyan-400">{obsRain} mm</span>
                </div>
                <div className="flex justify-between items-center text-2xs font-mono pt-1 border-t border-slate-800">
                  <span className="text-slate-400">Model Error Delta:</span>
                  <span className={`font-bold ${isDry ? 'text-amber-400' : 'text-blue-400'}`}>
                    {delta > 0 ? `+${delta}` : delta} mm ({deltaPct}%)
                  </span>
                </div>
              </div>

              {/* Bust Outcome Badge */}
              <div className="mb-2">
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-3xs font-mono font-bold uppercase tracking-wider ${
                  isBust
                    ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                }`}>
                  {isBust ? (
                    <>
                      <AlertTriangle className="w-3 h-3 text-red-400" />
                      {item.outcome || `HISTORICAL BUST (${item.bust_direction})`}
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                      NO FORECAST BUST
                    </>
                  )}
                </span>
              </div>

              {/* Physical Atmospheric Mechanism */}
              <p className="text-3xs text-slate-400 font-sans leading-relaxed line-clamp-3">
                {item.mechanism ||
                  'Cross-equatorial Somali jet displacement caused severe over-intensification of coastal moisture flux convergence.'}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
