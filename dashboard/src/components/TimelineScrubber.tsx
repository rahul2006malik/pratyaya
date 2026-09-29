import React from 'react';
import { ChevronLeft, ChevronRight, CloudRain, Thermometer, ShieldAlert, Sparkles, Layers } from 'lucide-react';

interface TimelineScrubberProps {
  leadDay: number;
  onLeadDayChange: (lead: number) => void;
  variable: string;
  onVariableChange: (variable: string) => void;
  date: string;
  criticalCount?: number;
}

interface HorizonGroup {
  id: 'short' | 'medium' | 'extended';
  label: string;
  sublabel: string;
  leads: number[];
  isCoreWindow?: boolean;
}

const HORIZON_GROUPS: HorizonGroup[] = [
  {
    id: 'short',
    label: 'Short-Range',
    sublabel: 'Deterministic (D1–D3)',
    leads: [1, 2, 3],
  },
  {
    id: 'medium',
    label: 'Medium-Range',
    sublabel: 'Core Bust Detection Zone (D4–D7)',
    leads: [4, 5, 6, 7],
    isCoreWindow: true,
  },
  {
    id: 'extended',
    label: 'Extended-Range',
    sublabel: 'Probabilistic Decay (D8–D10)',
    leads: [8, 9, 10],
  },
];

export const TimelineScrubber: React.FC<TimelineScrubberProps> = ({
  leadDay,
  onLeadDayChange,
  variable,
  onVariableChange,
  date,
  criticalCount = 0,
}) => {
  return (
    <div className="w-full bg-white border-b border-slate-200 px-4 py-2 flex flex-col md:flex-row items-center justify-between gap-3 select-none shrink-0 shadow-xs z-20">
      {/* ── Left: Meteorological Horizons Grouping ────────────────────────── */}
      <div className="flex items-center gap-2 overflow-x-auto w-full md:w-auto pb-1 md:pb-0">
        <div className="flex items-center gap-1 shrink-0 text-slate-500 mr-1">
          <Layers className="w-4 h-4 text-slate-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-600">Horizon:</span>
        </div>

        {/* Step backward */}
        <button
          onClick={() => onLeadDayChange(Math.max(1, leadDay - 1))}
          disabled={leadDay <= 1}
          className="p-1 rounded text-slate-500 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition shrink-0"
          title="Previous Lead Day (Left Arrow)"
          aria-label="Previous Lead Day"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>

        {/* Horizon Segment Groups */}
        <div className="flex items-center gap-2.5">
          {HORIZON_GROUPS.map((group) => {
            const isGroupActive = group.leads.includes(leadDay);
            return (
              <div
                key={group.id}
                className={`flex items-center gap-1.5 px-2 py-1 rounded-md border transition-all ${
                  group.isCoreWindow
                    ? isGroupActive
                      ? 'bg-blue-50/70 border-blue-200'
                      : 'bg-slate-50/90 border-slate-200 hover:border-slate-300'
                    : isGroupActive
                    ? 'bg-slate-100 border-slate-300'
                    : 'bg-slate-50/50 border-slate-200'
                }`}
              >
                {/* Horizon Label & Status */}
                <div className="flex flex-col pr-1 border-r border-slate-200">
                  <div className="flex items-center gap-1">
                    <span className={`text-2xs font-bold uppercase tracking-tight ${
                      group.isCoreWindow ? 'text-blue-700' : 'text-slate-700'
                    }`}>
                      {group.label}
                    </span>
                    {group.isCoreWindow && (
                      <span className="text-3xs font-semibold px-1 py-px rounded bg-blue-100 text-blue-800 border border-blue-200">
                        CORE
                      </span>
                    )}
                  </div>
                  <span className="text-3xs text-slate-400 hidden xl:inline">
                    {group.sublabel}
                  </span>
                </div>

                {/* Day Buttons */}
                <div className="flex items-center gap-1">
                  {group.leads.map((day) => {
                    const isSelected = day === leadDay;
                    return (
                      <button
                        key={day}
                        onClick={() => onLeadDayChange(day)}
                        className={`h-7 px-2.5 rounded text-xs font-mono font-bold transition-all flex items-center justify-center gap-1 ${
                          isSelected
                            ? 'bg-blue-600 text-white shadow-xs ring-2 ring-blue-500/20 scale-105'
                            : 'bg-white text-slate-700 border border-slate-200 hover:bg-slate-100 hover:text-slate-900'
                        }`}
                        title={`Forecast Lead Day ${day}`}
                      >
                        <span>D{day}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Step forward */}
        <button
          onClick={() => onLeadDayChange(Math.min(10, leadDay + 1))}
          disabled={leadDay >= 10}
          className="p-1 rounded text-slate-500 hover:text-slate-900 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition shrink-0"
          title="Next Lead Day (Right Arrow)"
          aria-label="Next Lead Day"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {/* ── Right: Target Meteorological Variable & National Status ────── */}
      <div className="flex items-center gap-3 shrink-0">
        {/* National Risk Indicator */}
        <div className="hidden lg:flex items-center gap-2">
          {criticalCount > 0 ? (
            <span className="flex items-center gap-1.5 text-xs font-semibold text-red-700 bg-red-50 border border-red-200 px-2.5 py-1 rounded-full">
              <ShieldAlert className="w-3.5 h-3.5 text-red-600" />
              <span>{criticalCount} Subdivisions in High Bust Risk</span>
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full">
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              <span>NWP Guidance Stable Across India</span>
            </span>
          )}
        </div>

        {/* Meteorological Target Variable Segmented Control */}
        <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200">
          <button
            onClick={() => onVariableChange('rainfall')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
              variable === 'rainfall'
                ? 'bg-white text-slate-900 font-bold shadow-xs border border-slate-200/80'
                : 'text-slate-600 hover:text-slate-900'
            }`}
            title="Precipitation Forecast Bust Probability (Official PS Parameter)"
          >
            <CloudRain className="w-3.5 h-3.5 text-blue-600" />
            <span>Precipitation (mm)</span>
          </button>
          <button
            onClick={() => onVariableChange('t2m')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
              variable === 't2m'
                ? 'bg-white text-slate-900 font-bold shadow-xs border border-slate-200/80'
                : 'text-slate-600 hover:text-slate-900'
            }`}
            title="Surface 2m Temperature (Secondary Parameter)"
          >
            <Thermometer className="w-3.5 h-3.5 text-amber-600" />
            <span>2m Temperature</span>
          </button>
        </div>
      </div>
    </div>
  );
};
