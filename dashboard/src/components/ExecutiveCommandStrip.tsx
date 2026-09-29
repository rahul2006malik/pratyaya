import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Calendar,
  Clock,
  Compass,
  Database,
  FileText,
  History,
  Maximize2,
  Minimize2,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { ForecastCard } from '../api/types';

export interface TopThreatItem {
  rank: number;
  region_id: string;
  region_name: string;
  lead_day: number;
  p_bust: number;
  bust_direction: string;
  unprecedented_flag?: boolean;
}

interface ExecutiveCommandStripProps {
  cycleDate: string;
  leadDay: number;
  criticalCount: number;
  topThreats: TopThreatItem[];
  selectedRegionId: string;
  onSelectRegion: (regionId: string) => void;
  activeForecastCard: ForecastCard | null;
  isPresentationMode: boolean;
  onTogglePresentationMode: () => void;
  onOpenBulletin: () => void;
  onOpenModelStatus: () => void;
  onOpenJuryTour: () => void;
  viewMode: 'live' | 'replay' | 'regimes';
  onSelectViewMode: (mode: 'live' | 'replay' | 'regimes') => void;
  isLive: boolean;
  forceOffline: boolean;
  onToggleForceOffline: () => void;
  onCycleDateSelect?: (date: string) => void;
}

export const ExecutiveCommandStrip: React.FC<ExecutiveCommandStripProps> = ({
  cycleDate,
  leadDay,
  criticalCount,
  topThreats,
  selectedRegionId,
  onSelectRegion,
  activeForecastCard,
  isPresentationMode,
  onTogglePresentationMode,
  onOpenBulletin,
  onOpenModelStatus,
  onOpenJuryTour,
  viewMode,
  onSelectViewMode,
  isLive,
  forceOffline,
  onToggleForceOffline,
  onCycleDateSelect,
}) => {
  // Synchronized operational clock
  const [time, setTime] = useState<Date>(new Date());
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const pad = (n: number) => String(n).padStart(2, '0');
  const utc = `${pad(time.getUTCHours())}:${pad(time.getUTCMinutes())}:${pad(time.getUTCSeconds())} UTC`;
  const istMs = time.getTime() + 5.5 * 3600 * 1000;
  const istDate = new Date(istMs);
  const ist = `${pad(istDate.getUTCHours())}:${pad(istDate.getUTCMinutes())}:${pad(istDate.getUTCSeconds())} IST`;

  // Determine Directive Headline based on active forecast card and flags
  const isOOD = activeForecastCard?.unprecedented_flag || false;
  const pBust = activeForecastCard?.p_bust ?? 0;
  const bustDir = activeForecastCard?.bust_direction ?? 'none';
  const regionName = activeForecastCard?.region_name || 'Selected Region';

  let directiveHeadline = 'MODEL SPREAD NORMAL · SYNOPTIC CONFIDENCE HIGH ACROSS METEOROLOGICAL SUBDIVISIONS';
  let directiveCategory: 'critical-refusal' | 'dry-bust' | 'wet-bust' | 'watch' | 'normal' = 'normal';

  if (isOOD) {
    directiveHeadline = `CRITICAL SAFETY REFUSAL: UNPRECEDENTED CIRCULATION PATTERN OVER ${regionName.toUpperCase()} (OOD)`;
    directiveCategory = 'critical-refusal';
  } else if (pBust >= 0.25) {
    if (bustDir === 'dry') {
      directiveHeadline = `DIRECTIVE: SUPPRESS FALSE FLOOD ALARM IN ${regionName.toUpperCase()} (LEAD ${leadDay}) · DRY BUST DETECTED`;
      directiveCategory = 'dry-bust';
    } else if (bustDir === 'wet') {
      directiveHeadline = `DIRECTIVE: ESCALATE FLOOD ALERT IN ${regionName.toUpperCase()} (LEAD ${leadDay}) · WET BUST DETECTED`;
      directiveCategory = 'wet-bust';
    } else {
      directiveHeadline = `DIRECTIVE: RAPID MODEL SKILL DEGRADATION IN ${regionName.toUpperCase()} (LEAD ${leadDay}) · MONITOR CLOSELY`;
      directiveCategory = 'watch';
    }
  } else if (pBust >= 0.10) {
    directiveHeadline = `ELEVATED WATCH: MODERATE ENSEMBLE SPREAD IN ${regionName.toUpperCase()} (LEAD ${leadDay})`;
    directiveCategory = 'watch';
  }

  // Pre-configured quick cycle test cases
  const quickTestDates = [
    { label: 'Case A: June 2002 (Dry Bust)', date: '2002-06-07', lead: 6, reg: '35' },
    { label: 'Case B: August 2018 (OOD Refusal)', date: '2018-08-16', lead: 6, reg: '35' },
    { label: 'July 2002 (Monsoon Break)', date: '2002-07-16', lead: 4, reg: '35' },
  ];

  return (
    <div className={`w-full transition-all duration-300 select-none ${
      isPresentationMode ? 'bg-slate-950 border-b border-slate-800' : 'bg-slate-900 border-b border-slate-800'
    } shadow-md`}>
      {/* ── Level 1: Institutional & Triage Status Bar ──────────────────────── */}
      {!isPresentationMode && (
        <div className="h-10 px-4 border-b border-slate-800/80 flex items-center justify-between text-xs text-slate-300">
          {/* Institutional Badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-sm tracking-wider text-white font-sans">PRATYAYA</span>
              <span className="text-2xs text-slate-400 font-mono px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700">
                ATMOSPHERIC RESEARCH · NCMRWF/IMD
              </span>
              <span className="text-3xs font-mono px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold">
                v1.0.0 LOCKED
              </span>
            </div>

            <div className="hidden lg:flex items-center gap-1.5 text-2xs text-slate-400 border-l border-slate-800 pl-3">
              <span>Synoptic Analogue Diagnostic Engine & Forecast Bust Early Warning</span>
            </div>
          </div>

          {/* Center: Synchronized Clocks & Navigation Tabs */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 font-mono text-2xs text-slate-400 bg-slate-950 px-2.5 py-1 rounded border border-slate-800">
              <Clock className="w-3 h-3 text-slate-400" />
              <span className="text-slate-200 font-bold">{ist}</span>
              <span className="text-slate-600">/</span>
              <span className="text-slate-400">{utc}</span>
            </div>

            {/* View Mode Tabs */}
            <div className="flex items-center bg-slate-950 p-0.5 rounded-md border border-slate-800 text-2xs">
              <button
                onClick={() => onSelectViewMode('live')}
                className={`px-2.5 py-1 rounded transition font-medium flex items-center gap-1.5 ${
                  viewMode === 'live'
                    ? 'bg-blue-600 text-white font-bold shadow-xs'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
                title="Operational HUD (1)"
              >
                <span>Operational HUD</span>
                <kbd className="text-3xs opacity-60">1</kbd>
              </button>

              <button
                onClick={() => onSelectViewMode('replay')}
                className={`px-2.5 py-1 rounded transition font-medium flex items-center gap-1.5 ${
                  viewMode === 'replay'
                    ? 'bg-amber-600 text-white font-bold shadow-xs'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
                title="Historical Replay Simulator (2)"
              >
                <History className="w-3 h-3" />
                <span>Replay</span>
                <kbd className="text-3xs opacity-60">2</kbd>
              </button>

              <button
                onClick={() => onSelectViewMode('regimes')}
                className={`px-2.5 py-1 rounded transition font-medium flex items-center gap-1.5 ${
                  viewMode === 'regimes'
                    ? 'bg-indigo-600 text-white font-bold shadow-xs'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
                title="Synoptic Regimes Discovery (3)"
              >
                <Compass className="w-3 h-3" />
                <span>Regimes</span>
                <kbd className="text-3xs opacity-60">3</kbd>
              </button>

              <button
                onClick={onOpenModelStatus}
                className="px-2.5 py-1 rounded transition font-medium text-slate-400 hover:text-white hover:bg-slate-800 flex items-center gap-1.5"
                title="Scientific Benchmark & LOYO-CV Audit"
              >
                <Database className="w-3 h-3 text-slate-400" />
                <span>Audit</span>
                <span className="text-3xs bg-slate-800 text-slate-300 px-1 py-px rounded font-mono">LOYO</span>
              </button>
            </div>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-2">
            {/* Guided Tour */}
            <button
              onClick={onOpenJuryTour}
              className="hidden md:flex items-center gap-1 px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-2xs font-medium border border-slate-700 transition"
              title="60-Second Guided Tour (T)"
            >
              <Sparkles className="w-3 h-3 text-amber-400" />
              <span>Tour (T)</span>
            </button>

            {/* Shift Bulletin Export */}
            <button
              onClick={onOpenBulletin}
              className="flex items-center gap-1.5 px-3 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white text-2xs font-bold shadow-xs transition"
              title="Export Forecaster Advisory Bulletin (B)"
            >
              <FileText className="w-3 h-3" />
              <span>Bulletin (B)</span>
            </button>

            {/* Offline/Mock indicator */}
            <button
              onClick={onToggleForceOffline}
              className={`flex items-center gap-1 px-2 py-1 rounded text-2xs font-mono font-bold transition border ${
                !forceOffline && isLive
                  ? 'bg-emerald-950 text-emerald-300 border-emerald-800'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
              title="API status"
            >
              {!forceOffline && isLive ? (
                <>
                  <Wifi className="w-3 h-3 text-emerald-400" />
                  <span>LIVE</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3 h-3 text-slate-400" />
                  <span>MOCK</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* ── Level 2: The Core Executive Command HUD Strip (Persistent) ───────── */}
      <div className="px-4 py-2 flex flex-col xl:flex-row items-start xl:items-center justify-between gap-3 text-xs">
        {/* Left: Cycle Selector & Threat Level Badge */}
        <div className="flex items-center gap-2.5 shrink-0 flex-wrap">
          {/* Cycle Timestamp */}
          <div className="flex items-center gap-1.5 bg-slate-950 px-2.5 py-1.5 rounded border border-slate-800 font-mono">
            <Calendar className="w-3.5 h-3.5 text-blue-400 shrink-0" />
            <span className="text-3xs uppercase tracking-wider text-slate-400 font-sans">CYCLE:</span>
            <span className="font-bold text-white text-xs">{cycleDate} 00Z</span>
          </div>

          {/* Quick Benchmark Switcher Dropdown / Pills */}
          {onCycleDateSelect && (
            <div className="hidden lg:flex items-center gap-1 bg-slate-950/70 p-0.5 rounded border border-slate-800 text-3xs font-mono">
              <span className="px-1 text-slate-500 uppercase">Test Case:</span>
              {quickTestDates.map((tc) => (
                <button
                  key={tc.date}
                  onClick={() => onCycleDateSelect(tc.date)}
                  className={`px-1.5 py-0.5 rounded transition ${
                    cycleDate.startsWith(tc.date)
                      ? 'bg-blue-600 text-white font-bold'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800'
                  }`}
                  title={`Switch to ${tc.label}`}
                >
                  {tc.date === '2002-06-07' ? 'Case A (Dry)' : tc.date === '2018-08-16' ? 'Case B (OOD)' : 'Break'}
                </button>
              ))}
            </div>
          )}

          {/* Threat Count Badge */}
          {criticalCount > 0 ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-red-950/90 border border-red-800 text-red-200 font-bold font-mono text-2xs animate-pulse">
              <ShieldAlert className="w-3.5 h-3.5 text-red-400 shrink-0" />
              <span>{criticalCount} CRITICAL BUST THREATS FLAGGED</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-emerald-950/80 border border-emerald-800 text-emerald-300 font-mono text-2xs">
              <span>0 CRITICAL THREATS</span>
            </div>
          )}
        </div>

        {/* Center: The Hero Operational Directive */}
        <div className="flex-1 w-full xl:w-auto xl:mx-3">
          <div className={`w-full px-3.5 py-1.5 rounded-md border flex items-center justify-between gap-3 shadow-inner ${
            directiveCategory === 'critical-refusal'
              ? 'bg-red-950/90 border-red-500 text-red-100 ring-1 ring-red-500'
              : directiveCategory === 'dry-bust'
              ? 'bg-amber-950/90 border-amber-500/80 text-amber-100'
              : directiveCategory === 'wet-bust'
              ? 'bg-red-950/80 border-red-600/80 text-red-100'
              : directiveCategory === 'watch'
              ? 'bg-amber-950/50 border-amber-700/60 text-amber-200'
              : 'bg-slate-950 border-slate-800 text-slate-300'
          }`}>
            <div className="flex items-center gap-2 overflow-hidden">
              {directiveCategory === 'critical-refusal' ? (
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 animate-bounce" />
              ) : directiveCategory === 'dry-bust' ? (
                <TrendingDown className="w-4 h-4 text-amber-400 shrink-0" />
              ) : directiveCategory === 'wet-bust' ? (
                <TrendingUp className="w-4 h-4 text-red-400 shrink-0" />
              ) : (
                <ShieldAlert className="w-4 h-4 text-blue-400 shrink-0" />
              )}
              <span className="font-mono text-xs font-extrabold tracking-tight truncate uppercase">
                {directiveHeadline}
              </span>
            </div>

            <div className="hidden 2xl:flex items-center gap-1.5 text-3xs font-mono text-slate-400 shrink-0">
              <span>LEAD D{leadDay}</span>
              <span>·</span>
              <span className="text-slate-300 font-bold">{regionName}</span>
            </div>
          </div>
        </div>

        {/* Right: Quick-Jump Threat Pills & Presentation Mode Toggle */}
        <div className="flex items-center gap-2 shrink-0 flex-wrap">
          {/* Quick-Jump Pills for Top Ranked Subdivisions */}
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-md border border-slate-800 text-2xs font-mono">
            <span className="text-3xs uppercase tracking-wider text-slate-500 font-sans px-1 hidden md:inline">
              Top Threats:
            </span>
            {topThreats.slice(0, 3).map((threat) => {
              const isSelected = threat.region_id === selectedRegionId;
              const isDry = threat.bust_direction === 'dry';
              return (
                <button
                  key={`${threat.rank}-${threat.region_id}`}
                  onClick={() => onSelectRegion(threat.region_id)}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded transition font-bold ${
                    isSelected
                      ? 'bg-blue-600 text-white ring-1 ring-white/30'
                      : 'bg-slate-900 text-slate-300 hover:bg-slate-800 hover:text-white border border-slate-800'
                  }`}
                  title={`Focus on ${threat.region_name} (Lead ${threat.lead_day}, ${(threat.p_bust * 100).toFixed(1)}% bust)`}
                >
                  <span className="opacity-70 text-3xs">#{threat.rank}</span>
                  <span className="truncate max-w-[80px]">{threat.region_name}</span>
                  <span className={isDry ? 'text-amber-400' : 'text-red-400'}>
                    {(threat.p_bust * 100).toFixed(0)}%{isDry ? '▼' : '▲'}
                  </span>
                </button>
              );
            })}
          </div>

          {/* 1-Click 16:9 Presentation Mode Button */}
          <button
            onClick={onTogglePresentationMode}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-sans text-xs font-bold transition shadow-sm ${
              isPresentationMode
                ? 'bg-amber-500 hover:bg-amber-400 text-slate-950 ring-2 ring-amber-400'
                : 'bg-slate-800 hover:bg-slate-700 text-white border border-slate-700'
            }`}
            title="Toggle 16:9 Fullscreen Briefing Slide Mode (F)"
          >
            {isPresentationMode ? (
              <>
                <Minimize2 className="w-3.5 h-3.5" />
                <span>Exit Slide</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-3.5 h-3.5 text-blue-400" />
                <span>⛶ Present Slide</span>
                <kbd className="font-mono text-3xs bg-slate-900 px-1 py-px rounded ml-0.5 text-slate-400">F</kbd>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
