import React, { useEffect, useState } from 'react';
import {
  Activity,
  Calendar,
  Clock,
  Compass,
  Database,
  FileText,
  History,
  Info,
  MapPin,
  RotateCcw,
  Wifi,
  WifiOff,
} from 'lucide-react';

interface HeaderProps {
  leadDay: number;
  onLeadDayChange: (lead: number) => void;
  variable: string;
  onVariableChange: (variable: string) => void;
  date: string;
  onDateChange?: (date: string) => void;
  isLive: boolean;
  forceOffline: boolean;
  onToggleForceOffline: () => void;
  onOpenModelStatus: () => void;
  isReplayMode: boolean;
  onToggleReplayMode: () => void;
  viewMode?: 'live' | 'replay' | 'regimes';
  onSelectViewMode?: (mode: 'live' | 'replay' | 'regimes') => void;
  criticalCount?: number;
  onOpenBulletin?: () => void;
  onOpenJuryTour?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  leadDay,
  onLeadDayChange,
  variable,
  onVariableChange,
  date,
  onDateChange,
  isLive,
  forceOffline,
  onToggleForceOffline,
  onOpenModelStatus,
  isReplayMode,
  onToggleReplayMode,
  viewMode,
  onSelectViewMode,
  criticalCount = 0,
  onOpenBulletin,
  onOpenJuryTour,
}) => {
  const currentMode = viewMode || (isReplayMode ? 'replay' : 'live');

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

  const tabBase =
    'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all select-none';
  const tabActive =
    'bg-blue-50 text-blue-800 font-semibold border border-blue-200 shadow-2xs';
  const tabInactive =
    'text-slate-600 hover:text-slate-900 hover:bg-slate-100 border border-transparent';

  return (
    <header className="bg-white border-b border-slate-200 px-4 h-14 min-h-[56px] flex items-center justify-between select-none shrink-0 z-30 shadow-2xs">
      {/* ── Left: Institutional Identity ───────────────────────────────────── */}
      <div className="flex items-center gap-3">
        {/* Institutional Icon */}
        <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-xs">
          <Activity className="w-5 h-5" />
        </div>

        <div>
          <div className="flex items-center gap-2">
            <span className="font-extrabold text-base text-slate-900 tracking-tight">
              PRATYAYA
            </span>
            <span className="text-xs text-slate-500 font-medium hidden sm:inline">
              प्रत्यय
            </span>
            <span className="text-2xs font-mono font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
              OPERATIONAL
            </span>
            <span className="text-2xs font-mono font-medium px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
              v1.0.0 LOCKED
            </span>
          </div>
          <div className="text-2xs text-slate-500 font-medium hidden lg:block">
            NCMRWF / IMD · Medium-Range Forecast Bust Detection & Synoptic Memory
          </div>
        </div>
      </div>

      {/* ── Center: Clocks, Modes, and Verification ─────────────────────────── */}
      <div className="flex items-center gap-3">
        {/* Synchronized Operational Clocks */}
        <div className="hidden md:flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-md px-2.5 py-1 text-xs font-mono text-slate-600">
          <Clock className="w-3.5 h-3.5 text-slate-400" />
          <span className="font-bold text-slate-800">{ist}</span>
          <span className="text-slate-300">|</span>
          <span className="text-slate-500">{utc}</span>
        </div>

        {/* View Mode Navigation Tabs */}
        <div className="flex items-center bg-slate-100 rounded-lg p-0.5 border border-slate-200">
          <button
            onClick={() => onSelectViewMode ? onSelectViewMode('live') : (isReplayMode && onToggleReplayMode())}
            className={`${tabBase} ${currentMode === 'live' ? tabActive : tabInactive}`}
            title="Operational Guidance & Bust Detection (1)"
          >
            <Activity className="w-3.5 h-3.5 text-blue-600" />
            <span>Operational Guidance</span>
            <kbd className="hidden xl:inline font-mono text-3xs opacity-60 ml-0.5 px-1 py-px rounded bg-slate-200/60">1</kbd>
          </button>

          <button
            onClick={() => onSelectViewMode ? onSelectViewMode('replay') : (!isReplayMode && onToggleReplayMode())}
            className={`${tabBase} ${currentMode === 'replay' ? tabActive : tabInactive}`}
            title="Historical Replay Simulator (2)"
          >
            <History className="w-3.5 h-3.5 text-amber-600" />
            <span>Replay Simulator</span>
            <kbd className="hidden xl:inline font-mono text-3xs opacity-60 ml-0.5 px-1 py-px rounded bg-slate-200/60">2</kbd>
          </button>

          <button
            onClick={() => onSelectViewMode && onSelectViewMode('regimes')}
            className={`${tabBase} ${currentMode === 'regimes' ? tabActive : tabInactive}`}
            title="Synoptic Regimes Discovery (3)"
          >
            <Compass className="w-3.5 h-3.5 text-indigo-600" />
            <span>Synoptic Regimes</span>
            <kbd className="hidden xl:inline font-mono text-3xs opacity-60 ml-0.5 px-1 py-px rounded bg-slate-200/60">3</kbd>
          </button>

          <div className="w-px h-4 bg-slate-300 mx-1" />

          {/* Model Status Modal Trigger */}
          <button
            onClick={onOpenModelStatus}
            className={`${tabBase} ${tabInactive}`}
            title="LOYO-CV Benchmark & Scientific Audit"
          >
            <Database className="w-3.5 h-3.5 text-slate-500" />
            <span>LOYO-CV Audit</span>
            <span className="font-mono text-3xs bg-slate-200 text-slate-800 font-bold px-1.5 py-px rounded ml-0.5">
              M2
            </span>
          </button>
        </div>

        {/* Forecast Cycle Selector Badge */}
        <div className="hidden xl:flex items-center gap-1.5 bg-slate-50 border border-slate-200 rounded-md px-2.5 py-1 text-xs text-slate-700">
          <Calendar className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-2xs uppercase text-slate-500 font-semibold">Cycle:</span>
          <strong className="font-mono font-bold text-slate-900">{date} 00Z</strong>
        </div>
      </div>

      {/* ── Right: Executive Actions & Connectivity ────────────────────────── */}
      <div className="flex items-center gap-2">
        {/* 1-Click Forecaster Bulletin Action (High Contrast Blue) */}
        {onOpenBulletin && (
          <button
            onClick={onOpenBulletin}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-xs transition"
            title="Export Forecaster Advisory Bulletin (B)"
          >
            <FileText className="w-4 h-4" />
            <span>Export Shift Bulletin</span>
            <kbd className="font-mono text-3xs bg-blue-700/80 px-1 py-px rounded ml-0.5 hidden sm:inline">B</kbd>
          </button>
        )}

        {/* 60-Second Guided Tour */}
        {onOpenJuryTour && (
          <button
            onClick={onOpenJuryTour}
            className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-medium transition"
            title="60-Second Guided Jury Tour (T)"
          >
            <Info className="w-3.5 h-3.5 text-slate-500" />
            <span>Tour</span>
            <kbd className="font-mono text-3xs bg-slate-100 text-slate-500 px-1 py-px rounded ml-0.5">T</kbd>
          </button>
        )}

        {/* Live / Offline Toggle Indicator */}
        <button
          onClick={onToggleForceOffline}
          className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-mono font-bold border transition ${
            !forceOffline && isLive
              ? 'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100'
              : 'bg-slate-100 text-slate-600 border-slate-200 hover:bg-slate-200'
          }`}
          title="Toggle backend API mode (Mock fallback available)"
        >
          {!forceOffline && isLive ? (
            <>
              <Wifi className="w-3.5 h-3.5 text-emerald-600" />
              <span>LIVE</span>
            </>
          ) : (
            <>
              <WifiOff className="w-3.5 h-3.5 text-slate-400" />
              <span>MOCK</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
