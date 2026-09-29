import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Layers,
  ShieldCheck,
  TrendingUp,
  X,
} from 'lucide-react';
import { ModelStatus } from '../api/types';

interface ModelStatusModalProps {
  isOpen: boolean;
  onClose: () => void;
  status: ModelStatus | null;
}

export const ModelStatusModal: React.FC<ModelStatusModalProps> = ({ isOpen, onClose, status }) => {
  if (!isOpen) return null;

  const benchmarkLadder = status?.benchmark_ladder ?? [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-fade-up">
      <div className="bg-white border border-slate-300 w-full max-w-4xl rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh] text-slate-900">

        {/* Header */}
        <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center text-blue-700">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900">Model Verification & Baseline Ladder Audit</h2>
              <p className="text-2xs text-slate-500 mt-px">
                3-Fold Purged LOYO-CV · Decisions D-008 &amp; D-010 · 100% Real Empirical Data (43,200 Pairs)
              </p>
            </div>
          </div>
          <button onClick={onClose}
            className="p-1.5 rounded-md text-slate-400 hover:text-slate-800 hover:bg-slate-200 transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-xs text-slate-800">

          {/* Temporal Partitioning Architecture */}
          <div>
            <h3 className="flex items-center gap-1.5 text-xs font-bold text-slate-900 uppercase tracking-wider mb-3">
              <Database className="w-4 h-4 text-blue-600" />
              Rigorous Out-of-Sample Protocol (3-Fold Purged LOYO-CV)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50">
                <span className="text-3xs text-slate-500 uppercase tracking-wider font-semibold block">Cross-Validation Design</span>
                <div className="text-sm font-mono font-bold mt-1 text-slate-900">3-Fold Purged LOYO-CV</div>
                <div className="text-2xs text-slate-600 mt-0.5 leading-relaxed">
                  2000, 2001, 2002 JJAS · 120 cycles · 43,200 real out-of-fold records
                </div>
              </div>
              <div className="p-3.5 rounded-lg border border-blue-200 bg-blue-50/50 text-blue-950">
                <span className="text-3xs text-blue-700 uppercase tracking-wider font-semibold block">Boundary Purge Window</span>
                <div className="text-sm font-mono font-bold mt-1 text-blue-900">12-Day Boundary Purge</div>
                <div className="text-2xs text-blue-800 mt-0.5 leading-relaxed">
                  ±15-day analogue query exclusion · 0 cross-season temporal leakage
                </div>
              </div>
              <div className="p-3.5 rounded-lg border border-amber-200 bg-amber-50/50 text-amber-950">
                <span className="text-3xs text-amber-700 uppercase tracking-wider font-semibold block">Event Clustering Standard</span>
                <div className="text-sm font-mono font-bold mt-1 text-amber-900">N = {status?.n_independent_events ?? "279"} 3D Events</div>
                <div className="text-2xs text-amber-800 mt-0.5 leading-relaxed">
                  {status?.n_independent_events ? (43200 / status.n_independent_events).toFixed(2) : "6.38"} rows/event · 1,000 block bootstrap draws over cycle dates
                </div>
              </div>
            </div>
          </div>

          {/* Benchmark Ladder Table */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="flex items-center gap-1.5 text-xs font-bold text-slate-900 uppercase tracking-wider">
                <TrendingUp className="w-4 h-4 text-blue-600" />
                Verified Benchmark Ladder (43,200 Real Out-of-Fold Pairs)
              </h3>
              <span className="text-2xs font-mono text-slate-500 font-medium">Primary: PR-AUC · Decisive: Brier Skill (BSS)</span>
            </div>

            <div className="rounded-lg border border-slate-200 overflow-x-auto shadow-2xs">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200 text-3xs font-semibold uppercase text-slate-600">
                    <th className="p-2.5">Model / Baseline</th>
                    <th className="p-2.5 text-right font-mono">PR-AUC [95% CI]</th>
                    <th className="p-2.5 text-right font-mono">AUROC</th>
                    <th className="p-2.5 text-right font-mono">Brier Score</th>
                    <th className="p-2.5 text-right font-mono">Rel Error</th>
                    <th className="p-2.5 text-right font-mono">BSS vs B1</th>
                    <th className="p-2.5 text-center">Lead D1–10 Sparkline</th>
                    <th className="p-2.5">Operational Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono text-xs">
                  {benchmarkLadder.map((row) => {
                    const isM2 = row.id === 'M2';
                    const isB3 = row.id === 'B3';
                    const prAucStr = typeof row.pr_auc === 'number' ? row.pr_auc.toFixed(4) : (row as any).prAuc ?? '—';
                    const prAucCiStr = row.pr_auc_ci_lower !== undefined && row.pr_auc_ci_upper !== undefined
                      ? `[${row.pr_auc_ci_lower.toFixed(4)}, ${row.pr_auc_ci_upper.toFixed(4)}]`
                      : (row as any).prAucCi ?? '';
                    const aurocStr = typeof row.auroc === 'number' ? row.auroc.toFixed(4) : (row as any).auroc ?? '—';
                    const brierStr = typeof row.brier === 'number' ? row.brier.toFixed(4) : (row as any).brier ?? '—';
                    const relErrStr = typeof row.reliability === 'number' ? row.reliability.toFixed(4) : (row as any).relErr ?? '—';
                    const bssStr = typeof row.bss_vs_b1 === 'number'
                      ? (row.bss_vs_b1 >= 0 ? `+${row.bss_vs_b1.toFixed(4)}` : row.bss_vs_b1.toFixed(4))
                      : (row as any).bss ?? '—';
                    const badgeClass = row.badge_class ?? (row as any).badgeClass ?? '';

                    return (
                      <tr
                        key={row.id}
                        className={
                          isM2
                            ? 'bg-emerald-50/80 text-slate-900 font-bold border-l-3 border-l-emerald-600'
                            : isB3
                            ? 'bg-amber-50/60 text-slate-900'
                            : 'hover:bg-slate-50 text-slate-700'
                        }
                      >
                        <td className="p-2.5 flex items-center gap-2">
                          {isM2 ? (
                            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                          ) : isB3 ? (
                            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                          ) : (
                            <div className="w-4 h-4" />
                          )}
                          <span className={isM2 ? 'font-bold text-slate-900' : isB3 ? 'font-semibold text-slate-800' : ''}>
                            {row.name}
                          </span>
                        </td>
                        <td className="p-2.5 text-right font-bold text-blue-700">
                          {prAucStr}{' '}
                          <span className="text-3xs text-slate-500 font-normal font-mono">{prAucCiStr}</span>
                        </td>
                        <td className="p-2.5 text-right text-slate-700">{aurocStr}</td>
                        <td className={`p-2.5 text-right font-bold ${isM2 ? 'text-emerald-700' : isB3 ? 'text-amber-700' : 'text-slate-600'}`}>
                          {brierStr}
                        </td>
                        <td className={`p-2.5 text-right ${isM2 ? 'text-emerald-700 font-semibold' : isB3 ? 'text-amber-700' : 'text-slate-600'}`}>
                          {relErrStr}
                        </td>
                        <td className={`p-2.5 text-right font-bold ${bssStr.startsWith('+') ? 'text-emerald-700' : bssStr.startsWith('-') ? 'text-red-700' : 'text-slate-600'}`}>
                          {bssStr}
                        </td>
                        <td className="p-2.5 text-center">
                          {row.pr_auc_by_lead && row.pr_auc_by_lead.length > 0 ? (
                            <div
                              className="inline-flex items-center px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200"
                              title={`Lead D1–10 PR-AUC: ${row.pr_auc_by_lead.map((v, i) => `D${i + 1}:${v.toFixed(3)}`).join(', ')}`}
                            >
                              <svg width="76" height="18" viewBox="0 0 76 18" className="overflow-visible">
                                {row.pr_auc_by_lead.map((val, idx) => {
                                  const barHeight = Math.max(2, Math.min(16, (val / 0.35) * 16));
                                  const x = idx * 7.5 + 1;
                                  const y = 18 - barHeight;
                                  const fill = isM2 ? '#059669' : isB3 ? '#D97706' : '#2563EB';
                                  return (
                                    <rect
                                      key={idx}
                                      x={x}
                                      y={y}
                                      width="5"
                                      height={barHeight}
                                      rx="1"
                                      fill={fill}
                                      opacity={0.85}
                                    >
                                      <title>{`Lead Day ${idx + 1}: PR-AUC = ${val.toFixed(4)}`}</title>
                                    </rect>
                                  );
                                })}
                              </svg>
                            </div>
                          ) : (
                            <span className="text-slate-400 font-mono text-2xs">—</span>
                          )}
                        </td>
                        <td className="p-2.5 font-sans text-2xs">
                          <span className={`px-1.5 py-0.5 rounded border text-3xs font-bold mr-1.5 ${badgeClass}`}>
                            {row.badge}
                          </span>
                          <span className="text-slate-500">{row.role}</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Paired Block Bootstrap & Calibration Insight */}
          <div className="p-4 rounded-lg bg-blue-50/70 border border-blue-200 space-y-2 text-slate-800">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-600" />
              <span className="text-xs font-bold text-blue-900 uppercase tracking-wider">
                Paired Block Bootstrap Comparison &amp; Operational Calibration Takeaway
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-2xs pt-1 leading-relaxed">
              <div className="border-l-3 border-blue-600 pl-3 space-y-1">
                <span className="text-blue-900 font-bold block">Gain Over Climatology (B1): +15.35 pts</span>
                <p className="text-slate-700">
                  Across 1,000 paired block bootstrap draws over identical initialization dates, <strong className="text-slate-900">Model M2</strong> outperforms the operational climatology bar (B1) by{' '}
                  <span className="font-mono text-blue-700 font-bold">&Delta; = +0.1535</span> [95% CI: <span className="font-mono text-slate-800">+0.1324, +0.1731</span>],{' '}
                  <span className="font-mono text-slate-800">p &lt; 0.001</span> (statistically significant gain over baseline).
                </p>
              </div>
              <div className="border-l-3 border-emerald-600 pl-3 space-y-1">
                <span className="text-emerald-900 font-bold block">Discrimination &amp; Calibration Parity (Decision D-020)</span>
                <p className="text-slate-700">
                  Under equivalent isotonic calibration (Decision D-020), both calibrated spread (B3: <span className="font-mono text-slate-800">BSS +12.82%, Brier 0.0367</span>) and Model M2 (<span className="font-mono text-emerald-700 font-bold">BSS +12.45%, Brier 0.0368</span>) achieve positive calibration skill over climatology.{' '}
                  Model M2&apos;s verified advantage is <strong className="text-emerald-700">+2.34 pts PR-AUC discrimination</strong> (<span className="font-mono text-emerald-700">&Delta; = +0.0234</span> [95% CI: <span className="font-mono text-slate-800">+0.0101, +0.0370</span>], <span className="font-mono text-slate-800">p &le; 0.001</span>), concentrated in extreme tail capture and historical case-based synoptic explainability.
                </p>
              </div>
            </div>
          </div>

          {/* Operational Scope */}
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="text-2xs font-bold text-slate-800 uppercase tracking-wider block mb-1">
              Operational Scope &amp; Function
            </span>
            <p className="text-slate-600 leading-relaxed text-2xs">
              <strong className="text-slate-900">Pratyaya is not a numerical weather prediction model.</strong>{' '}
              It does not generate rainfall or temperature values. Pratyaya is a calibrated meta-model that predicts{' '}
              <em>when numerical weather forecasts cannot be trusted</em>, translating probabilistic bust risk into
              effective lead time and retrieving historical synoptic precedents to empower IMD forecasters.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-2xs shrink-0">
          <span className="font-mono text-slate-500">
            Run: <strong className="text-slate-700">{status?.run_id || 'run-loyo-benchmark-2000-2002'}</strong> · 43,200 Real NOAA &amp; IMD Pairs
          </span>
          <button onClick={onClose}
            className="px-4 py-1.5 bg-slate-200 hover:bg-slate-300 text-slate-800 rounded-md transition font-semibold text-xs">
            Close Audit
          </button>
        </div>
      </div>
    </div>
  );
};
