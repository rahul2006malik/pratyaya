import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Calendar,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock,
  Compass,
  FastForward,
  Film,
  History,
  Layers,
  MapPin,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import {
  ConfidenceBand,
  ReplayFrame,
  ReplayScenario,
  ScenarioSummary,
} from '../api/types';

interface ReplayViewProps {
  scenario: ReplayScenario | null;
  scenarios: ScenarioSummary[];
  selectedScenarioId: string;
  onSelectScenario: (scenarioId: string) => void;
  currentLeadDay: number;
  onLeadDayChange: (lead: number) => void;
  isLoading: boolean;
  onSelectRegion?: (regionId: string) => void;
}

export const ReplayView: React.FC<ReplayViewProps> = ({
  scenario,
  scenarios,
  selectedScenarioId,
  onSelectScenario,
  currentLeadDay,
  onLeadDayChange,
  isLoading,
  onSelectRegion,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const timerRef = useRef<number | null>(null);

  const frames = scenario?.frames || [];
  const maxLead = frames.length > 0 ? frames[frames.length - 1].lead_day : 7;
  const currentFrame: ReplayFrame | undefined =
    frames.find((f) => f.lead_day === currentLeadDay) || frames[0];

  // Playback timer loop
  useEffect(() => {
    if (isPlaying) {
      const intervalMs = playbackSpeed === 2 ? 800 : playbackSpeed === 5 ? 400 : 1600;
      timerRef.current = window.setInterval(() => {
        onLeadDayChange(currentLeadDay >= maxLead ? 1 : currentLeadDay + 1);
      }, intervalMs);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, currentLeadDay, maxLead, playbackSpeed, onLeadDayChange]);

  const topSub = currentFrame?.subdivisions?.find(
    (s) => s.region_id === currentFrame.top_risk_region_id
  );

  return (
    <div className="max-w-[1720px] mx-auto px-4 lg:px-6 py-6 space-y-6 text-slate-100">
      {/* ── Top Header & Scenario Controller ────────────────────────────── */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-md relative overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 rounded text-3xs font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 uppercase flex items-center gap-1">
                <Film className="w-3 h-3 text-indigo-400" />
                Cinematic Time-Machine
              </span>
              <span className="px-2 py-0.5 rounded text-3xs font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                Lead-by-Lead Simulation
              </span>
            </div>
            <h1 className="font-display font-extrabold text-2xl tracking-tight text-white">
              Historical Replay: Progressive Forecast Evolution
            </h1>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
              Step backwards and forwards in time to observe how Pratyaya detected the atmospheric precursor signals
              up to 7 days before standard numerical weather prediction models recognized the forecast bust.
            </p>
          </div>

          {/* Scenario Selector Dropdown */}
          <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-3 min-w-[320px] shadow-inner">
            <span className="text-3xs font-mono font-semibold uppercase text-slate-400 block mb-1.5">
              Select Canonical Scenario:
            </span>
            <select
              value={selectedScenarioId}
              onChange={(e) => onSelectScenario(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 text-xs font-mono text-slate-200 rounded px-3 py-2 outline-none cursor-pointer hover:border-slate-500 transition"
            >
              {scenarios.map((s) => (
                <option key={s.id} value={s.id} className="bg-slate-900 text-slate-200">
                  {s.title}
                </option>
              ))}
            </select>
            <div className="text-3xs text-slate-400 font-mono mt-1.5 flex items-center justify-between">
              <span>{scenario?.title}</span>
              <span className="text-indigo-400 font-bold">120 Cycles</span>
            </div>
          </div>
        </div>

        {/* ── Transport Player Bar ────────────────────────────────────────── */}
        <div className="mt-6 pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-4">
          {/* Left Transport Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setIsPlaying(false);
                onLeadDayChange(1);
              }}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              title="Reset to Day 1"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            <button
              onClick={() => {
                setIsPlaying(false);
                onLeadDayChange(Math.max(1, currentLeadDay - 1));
              }}
              disabled={currentLeadDay <= 1}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 transition"
              title="Previous Lead Day"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-mono font-bold text-xs shadow-md transition ${
                isPlaying
                  ? 'bg-amber-500 hover:bg-amber-400 text-slate-950'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white'
              }`}
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4 fill-current" />
                  <span>PAUSE PLAYBACK</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>RUN REPLAY</span>
                </>
              )}
            </button>

            <button
              onClick={() => {
                setIsPlaying(false);
                onLeadDayChange(Math.min(maxLead, currentLeadDay + 1));
              }}
              disabled={currentLeadDay >= maxLead}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 transition"
              title="Next Lead Day"
            >
              <ChevronRight className="w-4 h-4" />
            </button>

            {/* Speed Multipliers */}
            <div className="flex items-center bg-slate-950 rounded-lg p-1 border border-slate-800 ml-2">
              {[1, 2, 5].map((speed) => (
                <button
                  key={`speed-${speed}`}
                  onClick={() => setPlaybackSpeed(speed)}
                  className={`px-2 py-1 rounded text-3xs font-mono transition ${
                    playbackSpeed === speed
                      ? 'bg-indigo-600 text-white font-bold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>

          {/* Active Frame Context Indicator */}
          <div className="flex items-center gap-3 font-mono text-xs">
            <span className="text-slate-400">
              FRAME:{' '}
              <strong className="text-white">
                LEAD DAY {currentLeadDay} OF {maxLead}
              </strong>
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400">
              INIT CYCLE: <strong className="text-indigo-400">{currentFrame?.cycle_date || scenario?.cycle_date || '2002-07-07'}</strong>
            </span>
            <span className="text-slate-600">→</span>
            <span className="text-slate-400">
              TARGET HORIZON: <strong className="text-cyan-400">{currentFrame?.valid_date || '—'} (D+{currentLeadDay})</strong>
            </span>
          </div>
        </div>

        {/* ── Visual Scrubber Bar ─────────────────────────────────────────── */}
        <div className="mt-4 pt-2">
          <div className="flex items-center justify-between text-3xs font-mono text-slate-400 mb-1.5">
            <span>D+1 (Initialization)</span>
            <span className="text-amber-400">D+4 (Precursor Detection)</span>
            <span>D+{maxLead} (Target Horizon)</span>
          </div>

          <div className="grid grid-cols-7 lg:grid-cols-10 gap-1.5">
            {Array.from({ length: maxLead }, (_, i) => i + 1).map((lead) => {
              const isCurrent = lead === currentLeadDay;
              const frameForLead = frames.find((f) => f.lead_day === lead);
              const maxP = frameForLead?.max_p_bust ?? 0.1;
              const isHigh = maxP >= 0.3;

              return (
                <button
                  key={`scrub-lead-${lead}`}
                  onClick={() => {
                    setIsPlaying(false);
                    onLeadDayChange(lead);
                  }}
                  className={`py-2 px-1 rounded-lg border text-center transition flex flex-col items-center justify-between ${
                    isCurrent
                      ? 'bg-indigo-600 text-white border-indigo-400 ring-2 ring-indigo-400/50 shadow-md'
                      : isHigh
                      ? 'bg-red-950/40 text-red-200 border-red-800/80 hover:bg-red-900/50'
                      : 'bg-slate-950 text-slate-300 border-slate-800 hover:bg-slate-800'
                  }`}
                >
                  <span className="text-xs font-mono font-bold">D+{lead}</span>
                  <span className="text-3xs font-mono opacity-80 mt-0.5">
                    {Math.round(maxP * 100)}% risk
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* ── Live Forecaster Narrative Briefing Ticker ────────────────────── */}
      <div className="bg-gradient-to-r from-blue-950/40 via-indigo-950/40 to-slate-900 border-2 border-indigo-500/40 rounded-xl p-5 shadow-lg relative overflow-hidden">
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg bg-indigo-500/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shrink-0 mt-0.5">
            <Sparkles className="w-5 h-5 animate-pulse" />
          </div>

          <div className="flex-1 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-3xs font-mono uppercase tracking-wider text-indigo-300 font-bold">
                Duty Forecaster Synoptic Briefing · Day {currentLeadDay} Evolution
              </span>
              <span className="text-3xs font-mono text-slate-400">
                Simulation Cycle: <strong className="text-indigo-300">{currentFrame?.cycle_date || scenario?.cycle_date || '2002-07-07'}</strong> (Verifying D+{currentLeadDay})
              </span>
            </div>

            <p className="text-sm font-sans text-slate-100 font-medium leading-relaxed">
              {currentFrame?.caption_strip ||
                'Synoptic analogues establish persistent moisture flux divergence over Western Ghats; models show initial under-estimation of break-monsoon conditions.'}
            </p>
          </div>
        </div>
      </div>

      {/* ── Synchronized Replay Diagnostics Grid ─────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Top Risk Subdivision */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-md">
          <span className="text-3xs font-mono uppercase text-slate-400 block mb-1">
            Primary Vulnerable Catchment
          </span>
          <div className="flex items-center justify-between">
            <h3 className="font-display font-bold text-lg text-white">
              {currentFrame?.top_risk_region_name || 'Kerala'}
            </h3>
            <span className="px-2 py-0.5 rounded text-3xs font-mono font-bold bg-red-500/20 text-red-300 border border-red-500/30">
              #{currentFrame?.top_risk_region_id || '35'}
            </span>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 space-y-2 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-slate-400">Maximum Bust Probability:</span>
              <span className="font-bold text-red-400">
                {currentFrame ? (currentFrame.max_p_bust * 100).toFixed(1) : 32.4}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Mean Horizon (ELT):</span>
              <span className="font-bold text-amber-400">
                {currentFrame?.mean_effective_lead_time?.toFixed(1) || 8.8} Days
              </span>
            </div>
            {topSub && (
              <div className="flex justify-between pt-1 border-t border-slate-800/80">
                <span className="text-slate-400">NWP vs Obs Rain:</span>
                <span className="font-bold text-white">
                  {topSub.nwp_rainfall} mm / {topSub.obs_rainfall ?? '—'} mm
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Card 2: Ranked Triage Queue for This Frame */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-md md:col-span-2">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-800">
            <span className="text-3xs font-mono uppercase font-bold text-slate-400">
              Ranked Triage Queue · Day {currentLeadDay} State
            </span>
            <span className="text-3xs font-mono text-slate-400">
              Sorted by Priority Score (p_bust × decision_weight)
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-3xs text-slate-500 uppercase">
                  <th className="pb-2">Rank</th>
                  <th className="pb-2">Subdivision</th>
                  <th className="pb-2 text-right">p_bust</th>
                  <th className="pb-2 text-right">ELT</th>
                  <th className="pb-2 text-center">Bias</th>
                  <th className="pb-2">Top Risk Factor</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {(currentFrame?.triage_queue || []).slice(0, 4).map((q) => (
                  <tr
                    key={`replay-q-${q.region_id}`}
                    className="hover:bg-slate-800/40 cursor-pointer transition"
                    onClick={() => onSelectRegion?.(q.region_id)}
                  >
                    <td className="py-2 text-slate-400">#{q.rank}</td>
                    <td className="py-2 font-sans font-semibold text-slate-100">
                      {q.region_name}
                    </td>
                    <td className="py-2 text-right font-bold text-red-400">
                      {(q.p_bust * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 text-right text-slate-300">
                      {q.effective_lead_time.toFixed(1)}d
                    </td>
                    <td className="py-2 text-center">
                      <span className={`px-1.5 py-0.2 rounded text-3xs uppercase ${
                        q.bust_direction === 'dry' ? 'bg-amber-950/60 text-amber-300' : 'bg-blue-950/60 text-blue-300'
                      }`}>
                        {q.bust_direction}
                      </span>
                    </td>
                    <td className="py-2 text-3xs font-sans text-slate-400 truncate max-w-[200px]">
                      {q.top_risk_factor || 'Orographic moisture convergence collapse'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
