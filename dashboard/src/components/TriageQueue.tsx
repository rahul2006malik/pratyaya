/**
 * TriageQueue — Phase 5 clean-slate rebuild per spec Sections 1.2, 3.1
 *
 * Right-rail triage queue (~36% width on Operational Stage).
 * Ranked by priority_score = p_bust × decision_weight.
 * Macro Pulse Header shows "9 of 36 subdivisions LOW / CRITICAL for Day 6".
 * Each row: rank, name, IMD alert badge, direction arrow, P(bust).
 */
import React from 'react';
import { ConfidenceBand, QueueItem } from '../api/types';

const IMD_BAND_FILL: Record<ConfidenceBand, string> = {
  HIGH:     '#4C8C6D',
  MODERATE: '#C99A3A',
  LOW:      '#B8551A',
  CRITICAL: '#B8203A',
};
const ACCENT = '#4FB8E0';

interface TriageQueueProps {
  queue: QueueItem[];
  totalSubdivisions?: number;
  selectedRegionId?: string;
  onSelectRegion: (regionId: string, leadDay?: number) => void;
  currentLeadDay: number;
  onViewAll?: () => void;
  nationalLowCriticalCount?: number;
}

export const TriageQueue: React.FC<TriageQueueProps> = ({
  queue,
  totalSubdivisions = 36,
  selectedRegionId,
  onSelectRegion,
  currentLeadDay,
  onViewAll,
  nationalLowCriticalCount,
}) => {
  // Use unified nationalLowCriticalCount if passed from App state; fallback to filtering queue
  const highRiskCount = nationalLowCriticalCount !== undefined
    ? nationalLowCriticalCount
    : queue.filter(
        (q) => q.confidence_band === 'LOW' || q.confidence_band === 'CRITICAL'
      ).length;

  return (
    <div
      className="h-full flex flex-col overflow-hidden font-sans"
      style={{ background: '#121B2E', borderLeft: '1px solid #1E2D42' }}
    >
      {/* ── Macro Pulse Header (Tier 0 "The Glance") ─────────────────────── */}
      <div
        className="px-4 py-3 shrink-0"
        style={{ borderBottom: '1px solid #1E2D42', background: '#0B1220' }}
      >
        <div className="flex items-baseline gap-2">
          <span
            className="font-mono font-bold"
            style={{
              fontSize: '2.25rem',
              lineHeight: 1,
              color: highRiskCount > 0 ? '#B8203A' : '#4C8C6D',
              letterSpacing: '-0.02em',
            }}
          >
            {highRiskCount}
          </span>
          <span className="font-sans text-xs" style={{ color: '#8792A8' }}>
            of {totalSubdivisions} subdivisions
            <br />
            <span style={{ color: '#E9EDF5', fontWeight: 600 }}>LOW / CRITICAL</span>
            {' '}for Day {currentLeadDay}
          </span>
        </div>

        <div className="flex items-center justify-between mt-2">
          <span className="font-sans text-[0.625rem] uppercase tracking-widest" style={{ color: '#8792A8' }}>
            TRIAGE QUEUE
          </span>
          <span className="font-mono text-[0.55rem]" style={{ color: '#3D536E' }}>
            sort: priority ▾
          </span>
        </div>
      </div>

      {/* ── Ranked Queue Rows ─────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {queue.length === 0 ? (
          <div className="p-6 text-center font-sans text-xs" style={{ color: '#8792A8' }}>
            No subdivisions flagged for Day {currentLeadDay}
          </div>
        ) : (
          <div>
            {queue.map((item, idx) => {
              const isSelected = item.region_id === selectedRegionId;
              const bandColor  = IMD_BAND_FILL[item.confidence_band] ?? '#3D536E';
              const bustPct    = (item.p_bust * 100).toFixed(1);

              return (
                <button
                  key={`${item.region_id}-${item.lead_day}`}
                  onClick={() => onSelectRegion(item.region_id, item.lead_day)}
                  className="w-full text-left px-4 py-2.5 transition-colors"
                  style={{
                    background: isSelected ? 'rgba(79,184,224,0.08)' : 'transparent',
                    borderBottom: '1px solid #1E2D42',
                    borderLeft: `3px solid ${isSelected ? ACCENT : 'transparent'}`,
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'rgba(255,255,255,0.03)';
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <div className="flex items-center gap-2">
                    {/* Rank */}
                    <span
                      className="font-mono text-[0.625rem] font-semibold shrink-0 w-5 text-right"
                      style={{ color: '#3D536E' }}
                    >
                      {item.rank || idx + 1}
                    </span>

                    {/* IMD alert color swatch */}
                    <span
                      className="shrink-0 w-2 h-2 rounded-sm"
                      style={{ background: bandColor }}
                      title={item.confidence_band}
                    />

                    {/* Subdivision name */}
                    <span
                      className="flex-1 font-sans text-xs font-semibold truncate"
                      style={{ color: isSelected ? '#E9EDF5' : '#B8C5D6' }}
                    >
                      {item.region_name}
                    </span>

                    {/* Direction arrow */}
                    {item.bust_direction && item.bust_direction !== 'none' && (
                      <span
                        className="font-mono text-xs font-bold shrink-0"
                        style={{ color: item.bust_direction === 'wet' ? '#4C8C6D' : '#C99A3A' }}
                        title={`${item.bust_direction.toUpperCase()} bias`}
                      >
                        {item.bust_direction === 'wet' ? '↗' : '↘'}
                      </span>
                    )}

                    {/* P(bust) */}
                    <span
                      className="font-mono text-xs font-bold shrink-0 tabular-nums"
                      style={{ color: bandColor }}
                    >
                      {bustPct}%
                    </span>
                  </div>

                  {/* Sub-line: top risk factor (if available) */}
                  {item.top_risk_factor && (
                    <div
                      className="mt-0.5 pl-9 font-sans text-[0.55rem] truncate"
                      style={{ color: '#3D536E' }}
                    >
                      {item.top_risk_factor}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Footer: View All link → Bust Triad Room ─────────────────────── */}
      {onViewAll && queue.length > 0 && (
        <div
          className="px-4 py-2.5 shrink-0"
          style={{ borderTop: '1px solid #1E2D42', background: '#0B1220' }}
        >
          <button
            onClick={onViewAll}
            className="w-full font-sans text-xs text-center transition-colors"
            style={{ color: '#4FB8E0' }}
            onMouseEnter={(e) => (e.currentTarget.style.color = '#E9EDF5')}
            onMouseLeave={(e) => (e.currentTarget.style.color = '#4FB8E0')}
          >
            [ view all {queue.length} → Bust Triad Room ]
          </button>
        </div>
      )}
    </div>
  );
};
