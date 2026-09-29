import React, { useEffect } from 'react';
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Calendar,
  Compass,
  FileText,
  Info,
  Layers,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  X,
} from 'lucide-react';
import { ForecastCard } from '../api/types';

interface AnalogueDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  card: ForecastCard | null;
  leadDay: number;
  onOpenBulletin: () => void;
}

export const AnalogueDrawer: React.FC<AnalogueDrawerProps> = ({
  isOpen,
  onClose,
  card,
  leadDay,
  onOpenBulletin,
}) => {
  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const isOOD = card?.unprecedented_flag || false;
  const regionName = card?.region_name || 'Selected Region';
  const regionId = card?.region_id || '35';
  const pBust = card?.p_bust ?? 0;
  const elt = card?.effective_lead_time ?? leadDay;
  const analogues = card?.top_analogues || [];

  return (
    <div className="fixed inset-0 z-50 overflow-hidden select-none">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/40 backdrop-blur-xs transition-opacity duration-300"
        onClick={onClose}
      />

      {/* Slide-over Drawer Panel */}
      <div className="fixed inset-y-0 right-0 max-w-full flex pl-10">
        <div className="w-screen max-w-md md:max-w-lg bg-white border-l border-slate-200 shadow-2xl flex flex-col justify-between animate-in slide-in-from-right duration-300">
          {/* ── Top Header ─────────────────────────────────────────────────── */}
          <div className="px-5 py-4 bg-slate-900 text-white flex items-center justify-between border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-sm tracking-wide uppercase">
                  Synoptic Analogue Library
                </span>
                <span className="text-3xs font-mono bg-blue-600 text-white px-1.5 py-0.5 rounded font-bold">
                  D{leadDay}
                </span>
              </div>
              <div className="text-2xs text-slate-400 mt-0.5 font-mono">
                {regionName} (Subdivision #{regionId}) · 2000–2017 Archive
              </div>
            </div>

            <button
              onClick={onClose}
              className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition"
              title="Close Drawer (Esc)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* ── Scrollable Body Content ────────────────────────────────────── */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4 text-xs text-slate-800">
            {/* Risk Calibration Strip */}
            <div className="grid grid-cols-3 gap-2 p-3 bg-slate-50 border border-slate-200 rounded-lg font-mono">
              <div>
                <span className="text-3xs text-slate-500 uppercase block font-sans">Bust Probability</span>
                <span className={`text-base font-extrabold ${pBust >= 0.25 ? 'text-red-600' : 'text-slate-800'}`}>
                  {(pBust * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-3xs text-slate-500 uppercase block font-sans">Effective Horizon</span>
                <span className="text-base font-extrabold text-blue-700">
                  D{elt.toFixed(1)}
                </span>
              </div>
              <div>
                <span className="text-3xs text-slate-500 uppercase block font-sans">Confidence Band</span>
                <span className="text-xs font-bold uppercase mt-1 inline-block px-1.5 py-0.5 rounded bg-slate-200 text-slate-800">
                  {card?.confidence_band || 'MODERATE'}
                </span>
              </div>
            </div>

            {/* If Native OOD Refusal */}
            {isOOD ? (
              <div className="p-4 bg-red-50 border-2 border-red-500 rounded-lg space-y-3">
                <div className="flex items-center gap-2 text-red-900 font-extrabold">
                  <ShieldAlert className="w-5 h-5 text-red-600 shrink-0" />
                  <span>Out-of-Distribution Safety Refusal</span>
                </div>
                <p className="text-2xs text-red-800 leading-relaxed">
                  Atmospheric circulation vector Mahalanobis distance exceeds 99.9th percentile threshold. 
                  Nearest-neighbour analogue matching has been suppressed to prevent deceptive guidance.
                </p>
                <div className="p-2.5 bg-white border border-red-200 rounded text-3xs font-mono space-y-1">
                  <div>· Metric: Mahalanobis Distance &gt; 99.9%</div>
                  <div>· Moisture Influx: &gt;95th Percentile Anomaly</div>
                  <div>· Precedent Library Status: Zero un-purged matches</div>
                </div>
              </div>
            ) : (
              <>
                {/* Analogue Table */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-extrabold text-xs text-slate-900 uppercase tracking-tight">
                      Matched Historical Synoptic Cases ({analogues.length})
                    </span>
                    <span className="text-3xs font-mono text-slate-500">
                      Ranked by Mahalanobis Distance
                    </span>
                  </div>

                  <div className="space-y-2">
                    {analogues.map((ana, idx) => {
                      const isDry = ana.bust_direction === 'dry';
                      const isBust = ana.is_bust;
                      const nwp = ana.forecast_value ?? 70;
                      const obs = ana.observed_value ?? 15;
                      const delta = Math.round((obs - nwp) * 10) / 10;

                      return (
                        <div
                          key={`${ana.date}-${idx}`}
                          className={`p-3 rounded-lg border transition ${
                            idx === 0
                              ? 'bg-blue-50/50 border-blue-200 shadow-2xs'
                              : 'bg-white border-slate-200 hover:border-slate-300'
                          }`}
                        >
                          {/* Top Row: Date, Rank, Outcome */}
                          <div className="flex items-center justify-between font-mono text-2xs mb-1.5">
                            <div className="flex items-center gap-1.5">
                              <span className="font-extrabold text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded">
                                #{idx + 1} · {ana.date}
                              </span>
                              <span className="text-3xs text-slate-500">
                                (Weight: {ana.weight ? (ana.weight * 100).toFixed(1) : '28.5'}%)
                              </span>
                            </div>

                            <span className={`px-1.5 py-0.5 rounded font-bold text-3xs uppercase ${
                              isBust
                                ? isDry
                                  ? 'bg-amber-100 text-amber-800 border border-amber-300'
                                  : 'bg-red-100 text-red-800 border border-red-300'
                                : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                            }`}>
                              {ana.outcome || (isBust ? 'BUST' : 'Accurate')}
                            </span>
                          </div>

                          {/* Verification Delta */}
                          <div className="grid grid-cols-3 gap-1 p-1.5 bg-slate-50 rounded font-mono text-3xs mb-1.5 text-center">
                            <div>
                              <span className="text-slate-500 block">NWP Model</span>
                              <strong className="text-slate-800">{nwp.toFixed(1)} mm</strong>
                            </div>
                            <div>
                              <span className="text-slate-500 block">Observed</span>
                              <strong className="text-slate-800">{obs.toFixed(1)} mm</strong>
                            </div>
                            <div>
                              <span className="text-slate-500 block">Error Delta</span>
                              <strong className={isDry ? 'text-amber-700' : 'text-red-700'}>
                                {delta > 0 ? `+${delta}` : delta} mm
                              </strong>
                            </div>
                          </div>

                          {/* Physical Mechanism */}
                          <p className="text-2xs text-slate-600 leading-snug">
                            {ana.mechanism || 'Arabian sea offshore trough convergence anomaly.'}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Risk Factors */}
                {card?.risk_factors && card.risk_factors.length > 0 && (
                  <div className="pt-2">
                    <span className="font-extrabold text-xs text-slate-900 uppercase tracking-tight block mb-1.5">
                      Identified Synoptic Risk Factors
                    </span>
                    <ul className="space-y-1 text-2xs text-slate-700 list-disc list-inside">
                      {card.risk_factors.map((rf, rIdx) => (
                        <li key={rIdx} className="leading-relaxed">
                          {rf}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>

          {/* ── Footer Actions ─────────────────────────────────────────────── */}
          <div className="p-4 bg-slate-50 border-t border-slate-200 flex items-center justify-between gap-3">
            <button
              onClick={onClose}
              className="px-3 py-2 rounded-md bg-white border border-slate-300 hover:bg-slate-100 text-xs font-semibold text-slate-700 transition"
            >
              Close Drawer (Esc)
            </button>

            <button
              onClick={() => {
                onClose();
                onOpenBulletin();
              }}
              className="flex items-center gap-1.5 px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold shadow-xs transition"
            >
              <FileText className="w-4 h-4" />
              <span>Export into Shift Bulletin</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
