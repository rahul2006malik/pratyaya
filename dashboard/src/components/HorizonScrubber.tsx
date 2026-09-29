/**
 * HorizonScrubber — Phase 4 clean-slate rebuild per spec Section 3.2
 *
 * Physical ruler D1–D10 with:
 * - Continuous click/drag cross-fades map choropleth
 * - Risk-density ribbon: fraction of LOW/CRITICAL subdivisions per lead
 * - ELT Ghost Pin animation: slides from nominal to effective lead on selection
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ConfidenceBand, SubdivisionForecast } from '../api/types';

const IMD_BAND_FILL: Record<ConfidenceBand, string> = {
  HIGH:     '#4C8C6D',
  MODERATE: '#C99A3A',
  LOW:      '#B8551A',
  CRITICAL: '#B8203A',
};
const ACCENT = '#4FB8E0';

interface HorizonScrubberProps {
  leadDay: number;
  onSelectLeadDay: (lead: number) => void;
  /** Selected region's nominal lead day (usually == leadDay) */
  nominalLeadDay?: number;
  /** Selected region's effective lead time from API */
  effectiveLeadTime?: number;
  /** Region name for Ghost Pin annotation */
  selectedRegionName?: string;
  /** Per-lead risk density data (optional; derived from regions if not provided) */
  riskDensityByLead?: Record<number, number>; // lead -> fraction 0..1 in LOW/CRITICAL
  /** Full regions array so we can compute risk density locally */
  regions?: SubdivisionForecast[];
}

/** Compute fraction of subdivisions in LOW or CRITICAL at synthetic lead estimate */
function syntheticRiskDensity(regions: SubdivisionForecast[], lead: number, nominalLead: number): number {
  if (regions.length === 0) return 0.1;
  // Interpolate: earlier leads are safer, later leads are riskier
  const scale = Math.pow(lead / nominalLead, 1.4);
  const atNominal = regions.filter(
    (r) => r.confidence_band === 'LOW' || r.confidence_band === 'CRITICAL'
  ).length / regions.length;
  return Math.min(0.95, atNominal * scale);
}

export const HorizonScrubber: React.FC<HorizonScrubberProps> = ({
  leadDay,
  onSelectLeadDay,
  nominalLeadDay,
  effectiveLeadTime,
  selectedRegionName,
  regions = [],
}) => {
  const rulerRef   = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  // Whether the ghost pin has been animated for the current selection
  const [ghostAnimated, setGhostAnimated] = useState(false);
  const prevNominal = useRef<number | undefined>(undefined);

  const nominal = nominalLeadDay ?? leadDay;
  const elt     = effectiveLeadTime;
  const hasPinDelta = elt !== undefined && elt > nominal + 0.2;
  const pinDelta = hasPinDelta ? +(elt! - nominal).toFixed(1) : 0;

  // Trigger ghost pin animation only when nominal lead changes (one hero moment per spec)
  useEffect(() => {
    if (nominal !== prevNominal.current) {
      setGhostAnimated(false);
      const id = setTimeout(() => setGhostAnimated(true), 80);
      prevNominal.current = nominal;
      return () => clearTimeout(id);
    }
  }, [nominal]);

  // Drag-to-scrub
  const getLeadFromPointer = useCallback((clientX: number): number => {
    const ruler = rulerRef.current;
    if (!ruler) return leadDay;
    const { left, width } = ruler.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (clientX - left) / width));
    return Math.max(1, Math.min(10, Math.round(fraction * 9) + 1));
  }, [leadDay]);

  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    onSelectLeadDay(getLeadFromPointer(e.clientX));
  };

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      onSelectLeadDay(getLeadFromPointer(e.clientX));
    };
    const onUp = () => { isDragging.current = false; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [getLeadFromPointer, onSelectLeadDay]);

  // Risk density per lead (0..1 fraction in LOW/CRITICAL)
  const riskDensities = Array.from({ length: 10 }, (_, i) => {
    const lead = i + 1;
    return syntheticRiskDensity(regions, lead, nominal);
  });

  // Ruler geometry helpers
  const isOverflow  = elt !== undefined && elt > 10;
  const dayFraction = (day: number) => (day - 1) / 9; // 0..1
  const nominalPct  = dayFraction(nominal) * 100;
  const eltPct      = elt ? (isOverflow ? 100 : dayFraction(elt) * 100) : nominalPct;

  return (
    <div
      className="w-full select-none font-sans"
      data-testid="horizon-scrubber"
      style={{ background: '#0B1220', borderTop: '1px solid #1E2D42' }}
    >
      {/* ── ELT Ghost Pin Annotation ─────────────────────────────────────── */}
      {hasPinDelta && selectedRegionName && (
        <div
          className="px-4 py-1 font-sans text-[0.625rem] flex items-center gap-2"
          style={{ borderBottom: '1px solid #1E2D42', color: '#8792A8' }}
        >
          <span style={{ color: '#E9EDF5' }}>{selectedRegionName}:</span>
          <span>
            Nominal D{nominal}
          </span>
          <span style={{ color: '#4FB8E0' }}>➔</span>
          <span style={{ color: ACCENT, fontFamily: 'IBM Plex Mono, monospace', fontWeight: 600 }}>
            Effective {isOverflow ? `D10+ (D${elt?.toFixed(1)})` : `D${elt?.toFixed(1)}`}
          </span>
          <span style={{ color: '#B8551A', fontFamily: 'IBM Plex Mono, monospace' }}>
            (Predictability collapsed +{pinDelta} days{isOverflow ? ' — beyond 10-day NWP horizon' : ''})
          </span>
        </div>
      )}

      {/* ── Risk Density Ribbon (H4 widened contrast gradient) ───────────── */}
      <div className="relative flex h-5" style={{ borderBottom: '1px solid #1E2D42' }}>
        {riskDensities.map((density, i) => {
          const lead = i + 1;
          // Color: green→amber→rust→red based on density
          let ribbonColor = IMD_BAND_FILL.HIGH;
          if (density > 0.40)      ribbonColor = IMD_BAND_FILL.CRITICAL;
          else if (density > 0.22) ribbonColor = IMD_BAND_FILL.LOW;
          else if (density > 0.08) ribbonColor = IMD_BAND_FILL.MODERATE;

          return (
            <div
              key={lead}
              className="flex-1 relative cursor-pointer"
              onClick={() => onSelectLeadDay(lead)}
              title={`D${lead}: ${(density * 100).toFixed(0)}% LOW/CRITICAL`}
            >
              <div
                className="absolute inset-0 transition-opacity duration-200"
                style={{
                  background: ribbonColor,
                  opacity: Math.max(0.32, Math.min(0.95, 0.28 + Math.pow(density, 0.7) * 0.70)),
                  borderRight: lead < 10 ? '1px solid rgba(30,45,66,0.4)' : 'none',
                }}
              />
              {/* Selected day highlight */}
              {lead === leadDay && (
                <div className="absolute inset-0" style={{ background: ACCENT, opacity: 0.16 }} />
              )}
            </div>
          );
        })}
        <div
          className="absolute left-2 top-0 bottom-0 flex items-center font-mono text-[0.5rem] pointer-events-none"
          style={{ color: '#8792A8' }}
        >
          RISK DENSITY
        </div>
      </div>

      {/* ── Main Ruler ───────────────────────────────────────────────────── */}
      <div className="relative px-4 pb-2 pt-1.5">
        {/* Clickable / draggable ruler track */}
        <div
          ref={rulerRef}
          className="relative flex items-center h-8 scrubber-handle"
          onMouseDown={handleMouseDown}
        >
          {/* Track background */}
          <div
            className="absolute inset-y-2 inset-x-0 rounded"
            style={{ background: '#162238' }}
          />

          {/* Day ticks + labels */}
          {Array.from({ length: 10 }, (_, i) => {
            const day    = i + 1;
            const isActive = day === leadDay;
            const pct    = dayFraction(day) * 100;

            return (
              <button
                key={day}
                className="absolute flex flex-col items-center transform -translate-x-1/2 z-10"
                style={{ left: `${pct}%` }}
                onClick={(e) => { e.stopPropagation(); onSelectLeadDay(day); }}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowLeft') onSelectLeadDay(Math.max(1, day - 1));
                  if (e.key === 'ArrowRight') onSelectLeadDay(Math.min(10, day + 1));
                }}
                aria-label={`Lead Day ${day}`}
                tabIndex={0}
              >
                {/* Tick mark */}
                <div
                  className="w-0.5 rounded-full transition-all duration-150"
                  style={{
                    height: isActive ? 16 : 8,
                    background: isActive ? ACCENT : '#3D536E',
                    marginTop: isActive ? 0 : 4,
                  }}
                />
                {/* Day label */}
                <span
                  className="font-mono text-[0.6rem] font-semibold mt-0.5 transition-colors"
                  style={{ color: isActive ? '#E9EDF5' : '#8792A8' }}
                >
                  D{day}
                </span>
              </button>
            );
          })}

          {/* Selected day "solid pin" handle */}
          <div
            className="absolute z-20 transform -translate-x-1/2 -translate-y-0.5 transition-all duration-100"
            style={{ left: `${nominalPct}%`, top: '50%' }}
            title={`Selected: Day ${nominal}`}
          >
            <div
              className="w-4 h-4 rounded-full border-2 shadow-lg"
              style={{
                background: ACCENT,
                borderColor: '#0B1220',
                boxShadow: `0 0 8px ${ACCENT}80`,
              }}
            />
          </div>

          {/* ELT Ghost Pin — animates outward from solid pin */}
          {hasPinDelta && ghostAnimated && (
            <div
              className="absolute z-20 transform -translate-x-1/2 -translate-y-0.5 ghost-pin-animated"
              style={{
                left: `${nominalPct}%`,
                top: '50%',
                '--ghost-offset': `${(eltPct - nominalPct) * rulerRef.current?.offsetWidth! / 100}px`,
              } as React.CSSProperties}
              title={`Effective Lead: ${isOverflow ? `D10+ (D${elt?.toFixed(1)}) — Horizon Collapsed` : `Day ${elt?.toFixed(1)}`}`}
            >
              {/* Ghost pin visual (half-filled circle = ◐) */}
              <div
                className="relative w-4 h-4 rounded-full border-2 flex items-center justify-center"
                style={{
                  borderColor: ACCENT,
                  background: 'transparent',
                  overflow: 'hidden',
                  boxShadow: `0 0 6px ${ACCENT}50`,
                }}
              >
                {/* Left half filled */}
                <div
                  className="absolute inset-y-0 left-0 w-1/2"
                  style={{ background: ACCENT }}
                />
                {isOverflow && (
                  <span className="relative z-10 font-mono text-[0.45rem] font-bold text-[#E9EDF5] ml-1">
                    +
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Dashed connector: nominal → elt */}
          {hasPinDelta && (
            <svg
              className="absolute inset-0 pointer-events-none"
              width="100%"
              height="100%"
              style={{ overflow: 'visible' }}
            >
              <line
                x1={`${nominalPct}%`}
                y1="50%"
                x2={`${eltPct}%`}
                y2="50%"
                stroke={ACCENT}
                strokeWidth="1.5"
                strokeDasharray="4 3"
                opacity="0.5"
              />
            </svg>
          )}
        </div>

        {/* ELT annotation line */}
        {hasPinDelta && (
          <div className="relative h-4 mt-0.5">
            <span
              className="absolute font-mono text-[0.55rem] font-semibold transform -translate-x-1/2"
              style={{ left: `${nominalPct}%`, color: '#E9EDF5' }}
            >
              ●D{nominal}
            </span>
            <span
              className="absolute font-mono text-[0.55rem] font-semibold transform -translate-x-1/2"
              style={{ left: `${eltPct}%`, color: ACCENT }}
            >
              ◐ {isOverflow ? `D10+ (D${elt?.toFixed(1)})` : `D${elt?.toFixed(1)}`}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
