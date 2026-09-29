import React, { useEffect, useRef, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Film,
  Pause,
  Play,
  RotateCcw,
} from 'lucide-react';
import {
  ConfidenceBand,
  ReplayFrame,
  ReplayScenario,
  ScenarioSummary,
} from '../api/types';

interface ReplayControlProps {
  scenario: ReplayScenario | null;
  scenarios: ScenarioSummary[];
  selectedScenarioId: string;
  onSelectScenario: (scenarioId: string) => void;
  currentLeadDay: number;
  onLeadDayChange: (lead: number) => void;
  isLoading: boolean;
}

const BAND_BADGE: Record<ConfidenceBand, string> = {
  CRITICAL: 'bg-red-100 text-red-800 border-red-300 font-bold',
  LOW:      'bg-orange-100 text-orange-800 border-orange-300 font-bold',
  MODERATE: 'bg-amber-100 text-amber-800 border-amber-300 font-medium',
  HIGH:     'bg-emerald-100 text-emerald-800 border-emerald-300 font-medium',
};

export const ReplayControl: React.FC<ReplayControlProps> = ({
  scenario,
  scenarios,
  selectedScenarioId,
  onSelectScenario,
  currentLeadDay,
  onLeadDayChange,
  isLoading,
}) => {
  const [isPlaying,     setIsPlaying]     = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const timerRef = useRef<number | null>(null);

  const frames    = scenario?.frames || [];
  const maxLead   = frames.length > 0 ? frames[frames.length - 1].lead_day : 7;
  const currentFrame: ReplayFrame | undefined =
    frames.find((f) => f.lead_day === currentLeadDay) || frames[0];

  // Playback loop
  useEffect(() => {
    if (isPlaying) {
      const ms = playbackSpeed === 2 ? 900 : 1800;
      timerRef.current = window.setInterval(() => {
        onLeadDayChange(currentLeadDay >= maxLead ? 1 : currentLeadDay + 1);
      }, ms);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isPlaying, currentLeadDay, maxLead, playbackSpeed, onLeadDayChange]);

  const topSubBand = currentFrame?.subdivisions.find(
    (s) => s.region_id === currentFrame.top_risk_region_id
  )?.confidence_band;

  return (
    <div className="w-full bg-white border-b border-slate-200 flex flex-col select-none z-20 shrink-0 shadow-xs">
      {/* ── Transport bar ──────────────────────────────────────────────── */}
      <div className="px-4 py-2 flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 bg-slate-50/50">
        {/* Left: label + scenario select */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-50 border border-amber-200 text-amber-800 text-xs font-semibold">
            <Film className="w-3.5 h-3.5 text-amber-600" />
            <span>Historical Replay Simulator</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-medium">Scenario:</span>
            <select
              value={selectedScenarioId}
              onChange={(e) => onSelectScenario(e.target.value)}
              className="bg-white border border-slate-300 text-xs text-slate-800 rounded-md px-2.5 py-1 font-medium shadow-2xs focus:outline-none focus:border-blue-600 transition cursor-pointer"
            >
              {scenarios.map((s) => (
                <option key={s.id} value={s.id}>{s.title}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Center: transport controls */}
        <div className="flex items-center gap-1.5 bg-slate-100 px-2 py-1 rounded-lg border border-slate-200">
          <button onClick={() => { setIsPlaying(false); onLeadDayChange(1); }}
            className="p-1 rounded text-slate-600 hover:text-slate-900 hover:bg-slate-200 transition" title="Reset to Day 1">
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => { setIsPlaying(false); onLeadDayChange(Math.max(1, currentLeadDay - 1)); }}
            disabled={currentLeadDay <= 1}
            className="p-1 rounded text-slate-600 hover:text-slate-900 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed transition">
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setIsPlaying(!isPlaying)}
            className={`px-3 py-1 rounded-md text-xs font-bold flex items-center gap-1.5 transition shadow-2xs ${
              isPlaying
                ? 'bg-amber-600 text-white hover:bg-amber-700'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isPlaying ? <><Pause className="w-3.5 h-3.5 fill-current" /><span>Pause</span></>
                       : <><Play  className="w-3.5 h-3.5 fill-current" /><span>Play</span></>}
          </button>
          <button onClick={() => { setIsPlaying(false); onLeadDayChange(Math.min(maxLead, currentLeadDay + 1)); }}
            disabled={currentLeadDay >= maxLead}
            className="p-1 rounded text-slate-600 hover:text-slate-900 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed transition">
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => setPlaybackSpeed(playbackSpeed === 1 ? 2 : 1)}
            className="px-2 py-0.5 rounded text-xs font-mono font-bold text-blue-700 bg-blue-50 border border-blue-200 hover:bg-blue-100 transition ml-0.5">
            {playbackSpeed}×
          </button>
        </div>

        {/* Right: lead day pills */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-slate-500 font-medium">Lead:</span>
          <div className="flex gap-1">
            {Array.from({ length: maxLead }, (_, i) => i + 1).map((ld) => (
              <button key={ld}
                onClick={() => { setIsPlaying(false); onLeadDayChange(ld); }}
                className={`w-7 h-7 rounded-md text-xs font-mono font-bold transition flex items-center justify-center ${
                  currentLeadDay === ld
                    ? 'bg-blue-600 text-white shadow-2xs scale-105'
                    : 'bg-white text-slate-700 hover:bg-slate-100 border border-slate-200'
                }`}
              >
                D{ld}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Caption strip ──────────────────────────────────────────────── */}
      <div className="px-4 py-2.5 flex items-start gap-3 bg-white">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-2xs font-mono font-semibold text-slate-500 uppercase tracking-wider">
              Lead D{currentLeadDay} · {currentFrame?.valid_date || '—'}
            </span>
            {currentFrame && (
              <span className={`text-2xs font-mono px-2 py-0.5 rounded-full border ${
                BAND_BADGE[topSubBand || 'HIGH']
              }`}>
                Peak Risk: {currentFrame.top_risk_region_name} ({Math.round(currentFrame.max_p_bust * 100)}%)
              </span>
            )}
            {isLoading && (
              <span className="text-2xs text-blue-600 font-semibold animate-pulse">Loading frame…</span>
            )}
          </div>
          <p className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-md px-3 py-1.5 leading-relaxed font-sans font-medium">
            {currentFrame?.caption_strip || 'Scoring historical analogue flow across 36 meteorological subdivisions…'}
          </p>
        </div>
      </div>
    </div>
  );
};
