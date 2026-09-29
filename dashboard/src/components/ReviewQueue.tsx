import React from 'react';
import {
  ArrowDownRight,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Filter,
  Info,
  ListOrdered,
  ShieldAlert,
} from 'lucide-react';
import { ConfidenceBand, QueueItem } from '../api/types';

interface ReviewQueueProps {
  queue: QueueItem[];
  selectedRegionId?: string;
  onSelectRegion: (regionId: string, leadDay?: number) => void;
  currentLeadDay: number;
  filterByCurrentLead: boolean;
  onToggleFilterByLead: () => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

const ACTION_BAND_BADGES: Record<ConfidenceBand, { label: string; className: string }> = {
  CRITICAL: {
    label: 'CRITICAL',
    className: 'bg-red-100 text-red-800 border-red-300 font-bold',
  },
  LOW: {
    label: 'HIGH RISK',
    className: 'bg-orange-100 text-orange-800 border-orange-300 font-bold',
  },
  MODERATE: {
    label: 'WATCH',
    className: 'bg-amber-100 text-amber-800 border-amber-300 font-medium',
  },
  HIGH: {
    label: 'NORMAL',
    className: 'bg-emerald-100 text-emerald-800 border-emerald-300 font-medium',
  },
};

export const ReviewQueue: React.FC<ReviewQueueProps> = ({
  queue,
  selectedRegionId,
  onSelectRegion,
  currentLeadDay,
  filterByCurrentLead,
  onToggleFilterByLead,
  isCollapsed = false,
  onToggleCollapse,
}) => {
  const filteredQueue = filterByCurrentLead
    ? queue.filter((item) => item.lead_day === currentLeadDay)
    : queue;

  // ── Collapsed state (Compact Rail - Refinement 3) ─────────────────────────
  if (isCollapsed) {
    return (
      <div className="h-full w-12 bg-white border-r border-slate-200 flex flex-col items-center py-3 select-none shrink-0 shadow-xs z-10 justify-between">
        <div className="flex flex-col items-center gap-3">
          <button
            onClick={onToggleCollapse}
            className="p-1.5 rounded-md hover:bg-slate-100 text-slate-600 hover:text-slate-900 border border-slate-200 transition"
            title="Expand Operational Triage Rail («)"
            aria-label="Expand Triage Rail"
          >
            <ChevronRight className="w-4 h-4 text-blue-600" />
          </button>

          {/* Critical count indicator */}
          <div className="flex flex-col items-center gap-1">
            <span className="w-6 h-6 rounded-full bg-red-100 text-red-700 border border-red-200 font-mono text-xs font-bold flex items-center justify-center">
              {filteredQueue.length}
            </span>
          </div>

          {/* Vertical rotated text */}
          <div className="mt-4 [writing-mode:vertical-rl] rotate-180 text-xs font-bold tracking-wider text-slate-500 uppercase">
            Operational Triage Queue
          </div>
        </div>

        <div className="p-1">
          <button
            onClick={onToggleCollapse}
            className="p-1.5 rounded-md hover:bg-slate-100 text-slate-400 hover:text-slate-600"
            title="Expand Triage Rail"
          >
            <ListOrdered className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  // ── Expanded state (Full 320px Workstation Rail) ──────────────────────────
  return (
    <div className="h-full w-[320px] 2xl:w-[340px] flex flex-col bg-white border-r border-slate-200 select-none overflow-hidden text-slate-900 shrink-0 shadow-xs z-10">
      {/* ── Header: Table title & operational controls ─────────────────────────── */}
      <div className="px-3.5 py-2.5 border-b border-slate-200 shrink-0 bg-slate-50/70">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ListOrdered className="w-4 h-4 text-blue-600" />
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wide">
              Operational Triage Queue
            </h2>
            <span className="text-2xs font-mono font-bold px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
              {filteredQueue.length}
            </span>
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={onToggleFilterByLead}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-2xs font-medium transition border ${
                filterByCurrentLead
                  ? 'bg-blue-600 border-blue-600 text-white font-bold'
                  : 'bg-white border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
              title="Filter table by current lead day"
            >
              <Filter className="w-2.5 h-2.5" />
              <span>D{currentLeadDay} only</span>
            </button>

            {onToggleCollapse && (
              <button
                onClick={onToggleCollapse}
                className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-200 transition ml-0.5"
                title="Collapse Triage Rail (Shift view to full map)"
                aria-label="Collapse Triage Rail"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        <p className="text-3xs text-slate-500 mt-1 font-sans">
          Ranked strictly by Priority Score = <strong className="font-mono text-slate-700">p_bust × Vulnerability Wt</strong>
        </p>
      </div>

      {/* ── Table: High-density Operational Triage Grid ──────────────────────── */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {filteredQueue.length === 0 ? (
          <div className="p-8 text-center text-slate-400 text-xs">
            <p>No subdivisions exceeding risk threshold for Lead Day {currentLeadDay}.</p>
          </div>
        ) : (
          <table className="w-full table-fixed text-left border-collapse text-2xs">
            <thead>
              <tr className="bg-slate-100/80 text-slate-600 border-b border-slate-200 sticky top-0 z-10 text-3xs uppercase font-semibold">
                <th className="py-1.5 px-1 text-center w-6">#</th>
                <th className="py-1.5 px-1.5">Subdivision</th>
                <th className="py-1.5 px-1 text-center w-8">Lead</th>
                <th className="py-1.5 px-1 text-right w-13">p_bust</th>
                <th className="py-1.5 px-1 text-right w-7">Wt</th>
                <th className="py-1.5 px-1 text-right w-11 text-blue-700">Score</th>
                <th className="py-1.5 px-1 text-center w-14">Band</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredQueue.map((item, idx) => {
                const isSelected = item.region_id === selectedRegionId;
                const badge = ACTION_BAND_BADGES[item.confidence_band] || ACTION_BAND_BADGES.MODERATE;
                const bustPct = (item.p_bust * 100).toFixed(1);
                const rankDisplay = String(item.rank || idx + 1).padStart(2, '0');

                return (
                  <tr
                    key={`${item.region_id}-${item.lead_day}`}
                    onClick={() => onSelectRegion(item.region_id, item.lead_day)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-blue-50/90 text-slate-900 font-semibold border-l-3 border-l-blue-600 shadow-2xs'
                        : 'hover:bg-slate-50 text-slate-700'
                    }`}
                  >
                    {/* Rank */}
                    <td className="py-2 px-1 text-center font-mono text-slate-400 font-semibold">
                      {rankDisplay}
                    </td>

                    {/* Subdivision */}
                    <td className="py-2 px-1.5 min-w-0">
                      <div className="flex items-center gap-1 truncate">
                        <span className="truncate text-slate-900 font-medium">
                          {item.region_name}
                        </span>
                        <span className="text-slate-400 font-mono text-3xs shrink-0">
                          #{item.region_id}
                        </span>
                      </div>
                      {/* Secondary factor line */}
                      {item.top_risk_factor && (
                        <div className="text-3xs text-slate-500 truncate mt-0.5 leading-tight">
                          {item.top_risk_factor}
                        </div>
                      )}
                    </td>

                    {/* Lead */}
                    <td className="py-2 px-1 text-center font-mono text-slate-800 font-bold">
                      D{item.lead_day}
                    </td>

                    {/* p_bust */}
                    <td className="py-2 px-1 text-right font-mono font-bold text-slate-900">
                      <div className="flex items-center justify-end gap-0.5">
                        <span className={item.p_bust >= 0.25 ? 'text-red-700 font-extrabold' : item.p_bust >= 0.10 ? 'text-amber-700' : 'text-slate-800'}>
                          {bustPct}%
                        </span>
                        {item.bust_direction === 'wet' ? (
                          <span title="Wet Bias (Model under-predicting rain)" className="shrink-0">
                            <ArrowUpRight className="w-3 h-3 text-sky-600" />
                          </span>
                        ) : item.bust_direction === 'dry' ? (
                          <span title="Dry Bias (Model over-predicting rain)" className="shrink-0">
                            <ArrowDownRight className="w-3 h-3 text-amber-600" />
                          </span>
                        ) : null}
                      </div>
                    </td>

                    {/* Decision Weight */}
                    <td className="py-2 px-1 text-right font-mono text-slate-500">
                      {item.decision_weight.toFixed(1)}×
                    </td>

                    {/* Priority Score */}
                    <td className="py-2 px-1 text-right font-mono font-bold text-blue-700">
                      {item.priority_score.toFixed(3)}
                    </td>

                    {/* Action / Band Badge */}
                    <td className="py-2 px-1 text-center">
                      <span className={`inline-block px-1.5 py-px rounded text-3xs font-mono border tracking-tight ${badge.className}`}>
                        {badge.label}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Footer: Operational summary ───────────────────────────────────────── */}
      <div className="px-3 py-1.5 bg-slate-50 border-t border-slate-200 text-3xs font-mono text-slate-500 flex items-center justify-between shrink-0">
        <span>SORT: PRIORITY DESC</span>
        <span>TOP {filteredQueue.length} SUBDIVISIONS</span>
      </div>
    </div>
  );
};
