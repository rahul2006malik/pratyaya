/**
 * TopNav — Phase 2 clean-slate rebuild per spec Section 2.2
 * Persistent top strip with brand, 5 primary tabs, cycle selector,
 * subdivision search, export button, live clock (IST), and health heartbeat.
 */
import React, { useEffect, useState } from 'react';
import {
  Activity,
  BookOpen,
  CheckCircle2,
  Clock,
  Compass,
  Download,
  History,
  MapPin,
  Moon,
  Radio,
  Scale,
  Search,
  Sun,
  Wifi,
  WifiOff,
} from 'lucide-react';

export type NavTabId = 'operations' | 'triad' | 'replay' | 'regimes' | 'verification';

export interface ActiveCycle {
  date: string;
  leadDay: number;
}

export interface ReplayFrameState {
  cycleDate: string;
  leadDay: number;
  scenarioTitle?: string;
}

interface TopNavProps {
  activeTab: NavTabId;
  onSelectTab: (tab: NavTabId) => void;
  activeCycle: ActiveCycle;
  onSelectCycleDate: (date: string) => void;
  replayFrame?: ReplayFrameState;
  criticalCount: number;
  isLive: boolean;
  forceOffline: boolean;
  onToggleForceOffline: () => void;
  onOpenBulletin: () => void;
  onOpenJuryTour?: () => void;
  selectedRegionName?: string;
  onSearchSubdivision?: (query: string) => void;
  isDarkMode?: boolean;
  onToggleDarkMode?: () => void;
}

export const CANONICAL_CYCLES = [
  { id: '2002-06-07', label: '2002-06-07' },
  { id: '2002-07-16', label: '2002-07-16' },
  { id: '2001-07-10', label: '2001-07-10' },
  { id: '2018-08-16', label: '2018-08-16' },
];

/** Returns current IST time as HH:MM:SS string */
function getISTTime(): string {
  const now = new Date();
  const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
  return ist.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

export const TopNav: React.FC<TopNavProps> = ({
  activeTab,
  onSelectTab,
  activeCycle,
  onSelectCycleDate,
  replayFrame,
  criticalCount,
  isLive,
  forceOffline,
  onToggleForceOffline,
  onOpenBulletin,
  onOpenJuryTour,
  selectedRegionName,
  onSearchSubdivision,
  isDarkMode = true,
  onToggleDarkMode,
}) => {
  const [time, setTime] = useState(getISTTime());
  const [searchVal, setSearchVal] = useState('');
  const [showSearch, setShowSearch] = useState(false);

  // Live IST clock tick
  useEffect(() => {
    const id = setInterval(() => setTime(getISTTime()), 1000);
    return () => clearInterval(id);
  }, []);

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchVal(e.target.value);
    onSearchSubdivision?.(e.target.value);
  };

  const navTabs: Array<{
    id: NavTabId;
    label: string;
    shortLabel: string;
    icon: React.FC<{ className?: string }>;
    accentColor: string;
    badge?: string;
  }> = [
    {
      id: 'operations',
      label: 'Operational Stage',
      shortLabel: 'Stage',
      icon: Activity,
      accentColor: '#4FB8E0',
      badge: criticalCount > 0 ? String(criticalCount) : undefined,
    },
    {
      id: 'triad',
      label: 'Bust Triad Room',
      shortLabel: 'Triad',
      icon: Scale,
      accentColor: '#C99A3A',
    },
    {
      id: 'replay',
      label: 'Time Machine',
      shortLabel: 'Replay',
      icon: History,
      accentColor: '#7C5CBF',
    },
    {
      id: 'regimes',
      label: 'Regime Atlas',
      shortLabel: 'Regimes',
      icon: Compass,
      accentColor: '#4C8C6D',
    },
    {
      id: 'verification',
      label: 'Verification Hub',
      shortLabel: 'Verify',
      icon: CheckCircle2,
      accentColor: '#4C8C6D',
    },
  ];

  return (
    <header
      className="sticky top-0 z-50 w-full select-none"
      style={{ background: 'rgba(11, 18, 32, 0.97)', backdropFilter: 'blur(12px)', borderBottom: '1px solid #1E2D42' }}
    >
      {/* ── Row 1: Brand · Controls · Status ──────────────────────────────── */}
      <div className="max-w-[1920px] mx-auto px-4 lg:px-6 h-12 flex items-center justify-between gap-3">

        {/* Left: Brand */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Logo glyph */}
          <div
            className="flex items-center justify-center w-8 h-8 rounded"
            style={{ background: 'linear-gradient(135deg, #1E2D42 0%, #0B1220 100%)', border: '1px solid #4FB8E0' }}
          >
            <Radio className="w-4 h-4" style={{ color: '#4FB8E0' }} />
          </div>

          <div>
            <div className="flex items-center gap-2">
              {/* Brand title in Fraunces */}
              <span className="font-serif font-semibold text-sm tracking-tight" style={{ color: '#E9EDF5' }}>
                PRATYAYA
              </span>
              <span style={{ color: '#8792A8' }}>·</span>
              <span className="font-serif text-sm" style={{ color: '#8792A8' }}>प्रत्यय</span>
              <span
                className="hidden sm:inline font-mono text-[0.55rem] px-1.5 py-0.5 rounded uppercase tracking-wider"
                style={{ background: 'rgba(79,184,224,0.1)', color: '#4FB8E0', border: '1px solid rgba(79,184,224,0.3)' }}
              >
                RESEARCH LAB · V1.0
              </span>
            </div>
            <p className="hidden md:block font-sans text-[0.625rem] leading-none mt-0.5" style={{ color: '#8792A8' }}>
              Synoptic Analogue Diagnostic Engine & Forecast Bust Early Warning
            </p>
          </div>
        </div>

        {/* Center: Forecast Cycle Selector vs Replay Frame */}
        {activeTab === 'replay' ? (
          <div
            className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded transition-all"
            style={{
              background: 'rgba(124,92,191,0.18)',
              border: '1px solid #7C5CBF',
            }}
          >
            <History className="w-3.5 h-3.5 shrink-0" style={{ color: '#A78BFA' }} />
            <span
              className="font-mono text-[0.625rem] uppercase tracking-widest font-semibold"
              style={{ color: '#C4B5FD' }}
            >
              Replay Frame
            </span>
            <span className="font-mono text-xs font-bold" style={{ color: '#E9EDF5' }}>
              {replayFrame?.cycleDate || '2002-07-07'}
            </span>
            <span style={{ width: 1, height: 14, background: 'rgba(124,92,191,0.4)' }} />
            <span className="font-mono text-[0.625rem] font-bold" style={{ color: '#A78BFA' }}>
              D+{replayFrame?.leadDay ?? 6}
            </span>
            <span
              className="px-1.5 py-0.5 rounded text-[0.5rem] font-mono font-bold uppercase tracking-wider"
              style={{ background: 'rgba(124,92,191,0.3)', color: '#E9D5FF', border: '1px solid rgba(167,139,250,0.5)' }}
            >
              SIMULATION
            </span>
          </div>
        ) : (
          <div
            className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded transition-all"
            style={{
              background: '#0B1220',
              border: '1px solid #1E2D42',
            }}
          >
            <BookOpen className="w-3.5 h-3.5 shrink-0" style={{ color: '#8792A8' }} />
            <span
              className="font-mono text-[0.625rem] uppercase tracking-widest font-semibold"
              style={{ color: '#8792A8' }}
            >
              Cycle
            </span>
            <select
              value={activeCycle.date}
              onChange={(e) => onSelectCycleDate(e.target.value)}
              className="bg-transparent font-mono text-xs outline-none cursor-pointer"
              style={{ color: '#E9EDF5' }}
            >
              {CANONICAL_CYCLES.map((c) => (
                <option key={c.id} value={c.id} style={{ background: '#121B2E', color: '#E9EDF5' }}>
                  {c.label}
                </option>
              ))}
            </select>
            <span style={{ width: 1, height: 14, background: '#1E2D42' }} />
            <span className="font-mono text-[0.625rem]" style={{ color: '#4FB8E0' }}>
              D+{activeCycle.leadDay}
            </span>
          </div>
        )}

        {/* Right: Search + Actions + Status */}
        <div className="flex items-center gap-2">

          {/* Subdivision Search */}
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded" style={{ background: '#0B1220', border: '1px solid #1E2D42' }}>
            <Search className="w-3 h-3" style={{ color: '#8792A8' }} />
            <input
              type="text"
              value={searchVal}
              onChange={handleSearch}
              placeholder="Jump to subdivision…"
              className="bg-transparent font-sans text-xs outline-none w-36"
              style={{ color: '#E9EDF5' }}
            />
          </div>

          {/* Export A4 Bulletin */}
          <button
            onClick={onOpenBulletin}
            className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-sans font-semibold transition-colors"
            style={{ background: 'rgba(79,184,224,0.1)', color: '#4FB8E0', border: '1px solid rgba(79,184,224,0.3)' }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(79,184,224,0.2)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(79,184,224,0.1)')}
            title="Export A4 Shift Bulletin"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export A4 ⬇</span>
          </button>

          {/* Dark/Light toggle */}
          {onToggleDarkMode && (
            <button
              onClick={onToggleDarkMode}
              className="p-1.5 rounded transition-colors"
              style={{ color: '#8792A8', border: '1px solid #1E2D42' }}
              title="Toggle dark/light mode"
            >
              {isDarkMode ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
            </button>
          )}

          {/* Health heartbeat */}
          <button
            onClick={onToggleForceOffline}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded font-mono text-[0.625rem] font-semibold uppercase tracking-wider transition-colors"
            style={{
              background: isLive && !forceOffline ? 'rgba(76,140,109,0.12)' : 'rgba(201,154,58,0.12)',
              color:      isLive && !forceOffline ? '#4C8C6D'               : '#C99A3A',
              border:     `1px solid ${isLive && !forceOffline ? 'rgba(76,140,109,0.35)' : 'rgba(201,154,58,0.35)'}`,
            }}
            title="Toggle live/offline mode"
          >
            {isLive && !forceOffline ? (
              <><Wifi className="w-3 h-3" /><span className="hidden md:inline">Live</span></>
            ) : (
              <><WifiOff className="w-3 h-3" /><span className="hidden md:inline">Demo</span></>
            )}
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: isLive && !forceOffline ? '#4C8C6D' : '#C99A3A',
                animation: isLive && !forceOffline ? 'pulse 2s infinite' : 'none',
              }}
            />
          </button>

          {/* IST Clock */}
          <div className="hidden lg:flex items-center gap-1.5 font-mono text-[0.625rem]" style={{ color: '#8792A8' }}>
            <Clock className="w-3 h-3" />
            <span style={{ color: '#E9EDF5' }}>{time}</span>
            <span style={{ color: '#8792A8' }}>IST</span>
          </div>
        </div>
      </div>

      {/* ── Row 2: Navigation Tabs ─────────────────────────────────────────── */}
      <div
        className="max-w-[1920px] mx-auto px-4 lg:px-6 flex items-center gap-0.5 overflow-x-auto no-scrollbar"
        style={{ borderTop: '1px solid #1E2D42', height: '38px' }}
      >
        {navTabs.map(({ id, label, shortLabel, icon: Icon, accentColor, badge }) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => onSelectTab(id)}
              className="relative flex items-center gap-1.5 px-3 py-1.5 rounded font-sans text-xs font-semibold transition-colors whitespace-nowrap"
              style={{
                color:      isActive ? '#E9EDF5' : '#8792A8',
                background: isActive ? 'rgba(255,255,255,0.06)' : 'transparent',
              }}
              onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.color = '#E9EDF5'; }}
              onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.color = '#8792A8'; }}
            >
              <span style={{ color: isActive ? accentColor : 'inherit' }} className="flex shrink-0">
                <Icon className="w-3.5 h-3.5" />
              </span>
              <span className="hidden sm:inline">{label}</span>
              <span className="sm:hidden">{shortLabel}</span>
              {badge && (
                <span
                  className="font-mono text-[0.5rem] px-1 py-px rounded-full font-bold"
                  style={{ background: 'rgba(184,32,58,0.2)', color: '#B8203A', border: '1px solid rgba(184,32,58,0.4)' }}
                >
                  {badge}
                </span>
              )}
              {/* Active underline */}
              {isActive && (
                <span
                  className="absolute bottom-0 left-2 right-2 h-0.5 rounded-full"
                  style={{ background: accentColor }}
                />
              )}
            </button>
          );
        })}

        {/* Spacer + active region label */}
        <div className="ml-auto flex items-center gap-2 shrink-0 pl-4">
          {selectedRegionName && (
            <div className="hidden xl:flex items-center gap-1.5 font-sans text-[0.625rem]" style={{ color: '#8792A8' }}>
              <MapPin className="w-3 h-3" style={{ color: '#4FB8E0' }} />
              <span style={{ color: '#E9EDF5' }}>{selectedRegionName}</span>
            </div>
          )}
          {onOpenJuryTour && (
            <button
              onClick={onOpenJuryTour}
              className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 rounded font-mono text-[0.55rem] font-bold uppercase tracking-wider transition-colors"
              style={{ background: 'rgba(201,154,58,0.1)', color: '#C99A3A', border: '1px solid rgba(201,154,58,0.3)' }}
              title="60-second guided jury tour"
            >
              60s Tour
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
