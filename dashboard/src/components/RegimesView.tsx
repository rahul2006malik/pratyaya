import React, { useState } from 'react';
import {
  AlertCircle,
  Calendar,
  CheckCircle2,
  Compass,
  Droplets,
  Layers,
  MapPin,
  TrendingUp,
  Wind,
  Sparkles,
} from 'lucide-react';
import { ModelStatus, RegimeClusterSummary, RegimesResponse } from '../api/types';

interface RegimesViewProps {
  regimes: RegimesResponse | null;
  status?: ModelStatus | null;
  isLoading?: boolean;
}

export const RegimesView: React.FC<RegimesViewProps> = ({ regimes, status, isLoading }) => {
  const [selectedClusterId, setSelectedClusterId] = useState<number>(0);

  if (isLoading || !regimes) {
    return (
      <div className="max-w-[1720px] mx-auto px-4 lg:px-6 py-12 flex items-center justify-center">
        <div className="text-center space-y-3 bg-slate-900 border border-slate-800 rounded-xl p-8">
          <Compass className="w-8 h-8 text-cyan-400 animate-spin mx-auto" />
          <p className="text-xs font-mono text-slate-300">Discovering synoptic precursor regimes (Mahalanobis Space)…</p>
        </div>
      </div>
    );
  }

  const activeCluster: RegimeClusterSummary =
    regimes.clusters.find((c) => c.cluster_id === selectedClusterId) || regimes.clusters[0];

  return (
    <div className="max-w-[1720px] mx-auto px-4 lg:px-6 py-6 space-y-6 text-slate-100">
      {/* ── Editorial Header ────────────────────────────────────────────── */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-md relative overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <span className="text-3xs font-semibold px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/40 font-mono uppercase">
                Synoptic Regime Discovery
              </span>
              <span className="text-3xs font-semibold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono">
                Exploratory Physical Archetypes (Decision D-009 · N=13 &lt; 30)
              </span>
              <span className="text-3xs text-slate-400 font-mono">
                Rodwell et al. (2013) Methodology
              </span>
            </div>
            <h1 className="font-display font-extrabold text-2xl tracking-tight text-white">
              Indian Medium-Range Forecast Busts Cluster into Discrete Synoptic Precursor Classes
            </h1>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
              High-burden forecast cycles (&ge; P80 national bust burden at leads 3–7) cluster into physically distinct
              circulation archetypes in Mahalanobis metric space. Adhering to Decision D-009, with N = 13 independent events,
              patterns are reported as exploratory physical archetypes pending a multi-decade archive.
            </p>
          </div>

          {/* Silhouette score ladder */}
          <div className="bg-slate-950/90 px-4 py-3 rounded-lg border border-slate-800 shrink-0 shadow-inner">
            <div className="flex items-center justify-between gap-3 mb-2">
              <span className="text-3xs font-bold text-slate-400 uppercase tracking-wider font-mono">
                Silhouette Optimisation
              </span>
              <span className="text-xs font-mono text-cyan-400 font-bold">
                k = {regimes.optimal_k} (peak: 0.096)
              </span>
            </div>
            <div className="flex items-end gap-1.5 mb-1.5">
              {(Object.keys(regimes.silhouette_scores || {}).length > 0
                ? Object.keys(regimes.silhouette_scores).sort((a, b) => Number(a) - Number(b))
                : ['3', '4', '5', '6', '7', '8', '9', '10']
              ).map((kStr) => {
                const score = regimes.silhouette_scores[kStr] || 0;
                const isOptimal = Number(kStr) === regimes.optimal_k;
                return (
                  <div
                    key={kStr}
                    className={`flex flex-col items-center px-2 py-1 rounded text-center transition ${
                      isOptimal
                        ? 'bg-blue-600 text-white font-bold shadow-xs'
                        : 'bg-slate-900 border border-slate-800 text-slate-400'
                    }`}
                  >
                    <span className="text-3xs font-mono">k={kStr}</span>
                    <span className="text-3xs font-mono">{score.toFixed(3)}</span>
                  </div>
                );
              })}
            </div>
            <div className="text-4xs font-mono text-slate-400 text-right">
              Smallest cluster: <span className="text-white font-bold">N=2</span> cycles · Exploratory per D-009
            </div>
          </div>
        </div>
      </div>

      {/* ── Main Dual Panel Layout (Expansive & Responsive) ─────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Archetype Clusters List (4 cols) */}
        <div className="lg:col-span-5 space-y-2.5">
          <div className="flex items-center justify-between px-1 text-2xs font-mono font-bold uppercase tracking-wider text-slate-400">
            <span>Discovered Archetypes ({regimes.clusters.length})</span>
            <span>Historical Cycles</span>
          </div>

          <div className="space-y-2">
            {regimes.clusters.map((cluster) => {
              const isActive = cluster.cluster_id === selectedClusterId;
              const isWet = cluster.signed_bias === 'wet';
              const isDry = cluster.signed_bias === 'dry';

              return (
                <button
                  key={cluster.cluster_id}
                  onClick={() => setSelectedClusterId(cluster.cluster_id)}
                  className={`w-full text-left p-4 rounded-xl border transition shadow-xs flex flex-col justify-between ${
                    isActive
                      ? 'bg-slate-900 border-cyan-400 ring-1 ring-cyan-400/50 shadow-md'
                      : 'bg-slate-900/70 hover:bg-slate-900 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                        <span className="text-2xs font-mono font-bold text-slate-400">
                          Regime #{cluster.cluster_id + 1}
                        </span>
                        <span
                          className={`text-3xs px-2 py-0.5 rounded border font-mono font-bold uppercase tracking-wider ${
                            isWet
                              ? 'bg-blue-500/20 text-blue-300 border-blue-500/40'
                              : isDry
                              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                              : 'bg-slate-800 text-slate-300 border-slate-700'
                          }`}
                        >
                          {cluster.signed_bias === 'wet' ? 'Wet Bias' : cluster.signed_bias === 'dry' ? 'Dry Bias' : 'Mixed Bias'}
                        </span>
                        {cluster.is_validated && (
                          <span className="flex items-center gap-1 text-3xs text-emerald-400 font-mono">
                            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                            Validated
                          </span>
                        )}
                      </div>

                      <h3 className="font-display font-bold text-sm text-white leading-snug">
                        {cluster.name}
                      </h3>
                      <p className="text-2xs text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                        {cluster.mechanism_summary}
                      </p>
                    </div>

                    <div className="text-right shrink-0">
                      <span className="text-3xs text-slate-500 font-mono block">Cycles</span>
                      <span className="text-base font-mono font-bold text-white">
                        {cluster.n_members}
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right: Selected Cluster Diagnostic Deep-Dive (7 cols) */}
        {activeCluster && (
          <div className="lg:col-span-7 bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-md space-y-6">
            {/* Header info */}
            <div className="pb-4 border-b border-slate-800">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-2xs font-mono font-bold text-slate-400">
                  Regime #{activeCluster.cluster_id + 1} · {activeCluster.n_members} Historical Verification Cycles
                </span>
                {activeCluster.is_validated ? (
                  <span className="flex items-center gap-1 text-3xs font-mono font-bold text-emerald-300 bg-emerald-500/20 border border-emerald-500/40 px-2 py-0.5 rounded">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                    Statistically Validated (Chi² p &lt; 0.05)
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-3xs font-mono font-bold text-amber-300 bg-amber-500/20 border border-amber-500/40 px-2 py-0.5 rounded">
                    <AlertCircle className="w-3 h-3 text-amber-400" />
                    Exploratory Archetype
                  </span>
                )}
              </div>

              <h2 className="font-display font-bold text-xl text-white">
                {activeCluster.name}
              </h2>
              <p className="text-xs text-slate-300 mt-2 leading-relaxed font-sans">
                {activeCluster.mechanism_summary}
              </p>
            </div>

            {/* Core Physical Statistics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3.5 shadow-xs">
                <div className="flex items-center gap-1.5 text-slate-400 mb-1 text-3xs uppercase font-mono font-semibold">
                  <TrendingUp className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Mean Precipitation Error</span>
                </div>
                <div className="text-base font-mono font-bold text-white">
                  {activeCluster.mean_error_mm > 0 ? '+' : ''}{activeCluster.mean_error_mm.toFixed(1)} mm
                </div>
              </div>

              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3.5 shadow-xs">
                <div className="flex items-center gap-1.5 text-slate-400 mb-1 text-3xs uppercase font-mono font-semibold">
                  <Wind className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Dominant Error Bias</span>
                </div>
                <div className="text-base font-mono font-bold text-white capitalize">
                  {activeCluster.signed_bias || 'None'}
                </div>
              </div>

              <div className="bg-slate-950/80 border border-slate-800 rounded-lg p-3.5 shadow-xs">
                <div className="flex items-center gap-1.5 text-slate-400 mb-1 text-3xs uppercase font-mono font-semibold">
                  <Droplets className="w-3.5 h-3.5 text-amber-400" />
                  <span>Vulnerable Lead Days</span>
                </div>
                <div className="text-base font-mono font-bold text-cyan-400">
                  {activeCluster.vulnerable_leads?.map((d) => `D+${d}`).join(', ') || '—'}
                </div>
              </div>
            </div>

            {/* Threatened Meteorological Subdivisions */}
            {activeCluster.threatened_subdivisions?.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-2xs font-mono font-bold text-slate-400 uppercase">
                  <MapPin className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Most-Threatened Meteorological Subdivisions</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                  {activeCluster.threatened_subdivisions.slice(0, 6).map((sub) => (
                    <div
                      key={sub.region_id}
                      className="flex items-center justify-between p-3 rounded-lg bg-slate-950/80 border border-slate-800 shadow-xs"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-3xs font-mono text-slate-500">#{sub.region_id}</span>
                        <span className="text-xs font-semibold text-white font-sans">{sub.name}</span>
                      </div>
                      <div className="flex items-center gap-2 text-2xs font-mono">
                        <span className="text-red-400 font-bold bg-red-950/60 border border-red-800/80 px-1.5 py-0.5 rounded">
                          {(sub.bust_rate * 100).toFixed(0)}% bust
                        </span>
                        <span className={`font-semibold ${sub.mean_signed_error > 0 ? 'text-cyan-400' : 'text-amber-400'}`}>
                          {sub.mean_signed_error > 0 ? '+' : ''}{sub.mean_signed_error.toFixed(1)} mm
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Seasonal Distribution */}
            {activeCluster.seasonality && Object.keys(activeCluster.seasonality).length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-2xs font-mono font-bold text-slate-400 uppercase">
                  <Calendar className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Seasonal Distribution (Monsoon Progression)</span>
                </div>

                <div className="flex items-end gap-2 h-20 bg-slate-950/80 border border-slate-800 rounded-lg p-3">
                  {Object.entries(activeCluster.seasonality)
                    .sort(([a], [b]) => {
                      const order: Record<string, number> = {
                        Jan: 1, Feb: 2, Mar: 3, Apr: 4, May: 5, Jun: 6, June: 6,
                        Jul: 7, July: 7, Aug: 8, August: 8, Sep: 9, Sept: 9, September: 9,
                        Oct: 10, Nov: 11, Dec: 12,
                      };
                      const valA = order[a] ?? Number(a) ?? 99;
                      const valB = order[b] ?? Number(b) ?? 99;
                      return valA - valB;
                    })
                    .map(([month, count]) => {
                      const maxCount = Math.max(...Object.values(activeCluster.seasonality));
                      const heightPct = maxCount > 0 ? (count / maxCount) * 100 : 0;
                      const monthLabels: Record<string, string> = {
                        '1': 'JAN', '2': 'FEB', '3': 'MAR', '4': 'APR', '5': 'MAY', '6': 'JUN',
                        '7': 'JUL', '8': 'AUG', '9': 'SEP', '10': 'OCT', '11': 'NOV', '12': 'DEC',
                        Jan: 'JAN', Feb: 'FEB', Mar: 'MAR', Apr: 'APR', May: 'MAY', Jun: 'JUN', June: 'JUN',
                        Jul: 'JUL', July: 'JUL', Aug: 'AUG', August: 'AUG', Sep: 'SEP', Sept: 'SEP', September: 'SEP',
                        Oct: 'OCT', Nov: 'NOV', Dec: 'DEC',
                      };
                      const displayLabel = monthLabels[month] || month.slice(0, 3).toUpperCase();
                      return (
                        <div
                          key={month}
                          className="flex-1 flex flex-col items-center gap-1 h-full justify-end"
                          title={`${displayLabel}: ${count} cases`}
                        >
                          <div
                            className="w-full rounded-t bg-cyan-500 transition-all"
                            style={{ height: `${Math.max(heightPct, 8)}%` }}
                          />
                          <span className="text-3xs text-slate-300 font-mono font-semibold">
                            {displayLabel}
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}

            {/* Chi-Squared Enrichment Validation */}
            {activeCluster.validation?.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-2xs font-mono font-bold text-slate-400 uppercase">
                  <Layers className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Statistical Enrichment Validation (Catalogue Significance)</span>
                </div>

                <div className="space-y-2">
                  {activeCluster.validation.map((v, vi) => (
                    <div
                      key={vi}
                      className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 text-2xs font-mono"
                    >
                      <span className="text-slate-300 font-medium truncate max-w-[260px]">
                        {v.catalogue_name}
                      </span>
                      <div className="flex items-center gap-3">
                        <span className="text-slate-400">
                          {v.observed_count} obs / {v.expected_count.toFixed(1)} exp
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded font-semibold ${
                            v.is_significant
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {v.is_significant ? 'p < 0.05 (Sig)' : 'n.s.'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
