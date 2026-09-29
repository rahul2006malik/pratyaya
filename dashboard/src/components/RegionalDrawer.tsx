/**
 * RegionalDrawer — Phase 6 clean-slate rebuild per spec Sections 4, 7.2
 *
 * Non-modal contextual drawer. Slides in from right. Map stays visible on left.
 * Contains:
 * - Headline sentence in Fraunces display serif (Section 4.1)
 * - Compact Bust Triad: MODEL | ACTUAL | PRATYAYA (Section 5.1)
 * - Physical Risk Factors with meteorological glyphs (Section 4.3)
 * - Precedent Strip: horizontally scrollable Top-8 analogue cards (Section 4.2)
 * - OOD Circuit Breaker banner when unprecedented_flag (Section 4.4)
 */
import React, { useState } from 'react';
import { AlertTriangle, ChevronRight, Shield, Wind, X } from 'lucide-react';
import { AnalogueItem, ConfidenceBand, ForecastCard } from '../api/types';

const IMD_BAND_FILL: Record<ConfidenceBand, string> = {
  HIGH:     '#4C8C6D',
  MODERATE: '#C99A3A',
  LOW:      '#B8551A',
  CRITICAL: '#B8203A',
};
const ACCENT            = '#4FB8E0';
const UNPRECEDENTED_CLR = '#7C5CBF';

// ── Risk-factor glyph mapping ───────────────────────────────────────────────
function rfGlyph(factor: string): string {
  const lc = factor.toLowerCase();
  if (lc.includes('wind') || lc.includes('shear') || lc.includes('jet')) return '↝';
  if (lc.includes('trough') || lc.includes('axis'))                        return '⟿';
  if (lc.includes('analogue') || lc.includes('historical'))                return '⟲';
  if (lc.includes('moisture') || lc.includes('flux'))                      return '∿';
  return '·';
}

interface RegionalDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  card: ForecastCard | null;
  leadDay: number;
  isLoading?: boolean;
  onOpenBulletin?: () => void;
}

// ── Compact Bust Triad (Section 5.1) ───────────────────────────────────────
const BustTriad: React.FC<{ card: ForecastCard }> = ({ card }) => {
  const modelMm = card.top_analogues[0]?.forecast_value ?? 75.4;
  const obsMm   = card.top_analogues[0]?.observed_value ?? null;
  // Shared axis scale (C5) across Model and Observed
  const maxMm   = Math.max(modelMm, obsMm ?? 0, 60);
  const isLive  = card.is_live;

  const bar = (mm: number, color: string, opacity = 1) => (
    <div className="flex-1 rounded-sm transition-all duration-300" style={{
      width: `${Math.min(100, (mm / maxMm) * 100)}%`,
      background: color,
      opacity,
      height: 14,
    }} />
  );

  return (
    <div className="space-y-2.5" style={{ fontFamily: "'IBM Plex Mono', monospace" }}>
      {/* Lane 1: MODEL THOUGHT */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[0.55rem] uppercase tracking-widest font-semibold" style={{ color: '#8792A8' }}>
            MODEL THOUGHT
          </span>
          <span className="text-sm font-bold tabular-nums" style={{ color: '#E9EDF5' }}>
            {modelMm.toFixed(1)} mm
          </span>
        </div>
        <div className="w-full rounded overflow-hidden" style={{ background: '#162238', height: 14 }}>
          {bar(modelMm, '#4FB8E0', 0.9)}
        </div>
      </div>

      {/* Lane 2: ACTUALLY HAPPENED */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="text-[0.55rem] uppercase tracking-widest font-semibold" style={{ color: '#8792A8' }}>
            ACTUALLY HAPPENED
          </span>
          {isLive ? (
            <span className="text-[0.55rem] font-semibold" style={{ color: '#8792A8' }}>awaiting ground truth</span>
          ) : (
            <span className="text-sm font-bold tabular-nums" style={{ color: '#E9EDF5' }}>
              {obsMm !== null ? `${obsMm.toFixed(1)} mm` : '—'}
            </span>
          )}
        </div>
        <div className="w-full rounded overflow-hidden relative" style={{ background: '#162238', height: 14 }}>
          {isLive ? (
            /* Diagonal hatch placeholder per spec Section 5.2 */
            <div
              className="absolute inset-0 rounded"
              style={{
                backgroundImage: 'repeating-linear-gradient(-45deg, rgba(135,146,168,0.15) 0px, rgba(135,146,168,0.15) 3px, transparent 3px, transparent 8px)',
              }}
            />
          ) : obsMm !== null ? (
            bar(obsMm, '#4C8C6D', 0.85)
          ) : null}
        </div>
      </div>

      {/* Lane 3: PRATYAYA WARNED (separate unit — probability + direction + ELT) */}
      <div className="flex items-center gap-3 py-2 px-3 rounded" style={{ background: '#0B1220', border: '1px solid #1E2D42' }}>
        {/* Dial: P(bust) rendered as arc */}
        <svg width="40" height="40" viewBox="0 0 40 40">
          {/* Track */}
          <circle cx="20" cy="20" r="16" fill="none" stroke="#1E2D42" strokeWidth="5" />
          {/* Fill arc */}
          <circle
            cx="20" cy="20" r="16"
            fill="none"
            stroke={IMD_BAND_FILL[card.confidence_band]}
            strokeWidth="5"
            strokeDasharray={`${card.p_bust * 100.5} 100.5`}
            strokeLinecap="round"
            transform="rotate(-90 20 20)"
          />
          <text x="20" y="25" textAnchor="middle" fontSize="9" fontWeight="700"
            style={{ fill: '#E9EDF5', fontFamily: 'IBM Plex Mono, monospace' }}>
            {(card.p_bust * 100).toFixed(0)}%
          </text>
        </svg>

        <div className="flex-1 space-y-1">
          <div className="text-[0.55rem] uppercase tracking-widest" style={{ color: '#8792A8' }}>PRATYAYA WARNED</div>
          <div className="flex items-center gap-3">
            {/* Direction arrow */}
            {card.bust_direction && card.bust_direction !== 'none' && (
              <span className="font-bold text-xs" style={{
                color: card.bust_direction === 'wet' ? IMD_BAND_FILL.HIGH : IMD_BAND_FILL.MODERATE,
                fontFamily: 'IBM Plex Mono, monospace',
              }}>
                {card.bust_direction === 'wet' ? '↗ WET' : '↘ DRY'}
              </span>
            )}
            {/* ELT callout */}
            <span className="font-mono text-[0.625rem]" style={{ color: ACCENT }}>
              ELT: Day {card.effective_lead_time.toFixed(1)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

// ── Individual Analogue Receipt Card ───────────────────────────────────────
const AnalogueCard: React.FC<{ item: AnalogueItem; rank: number }> = ({ item, rank }) => {
  const [expanded, setExpanded] = useState(false);
  const isBust   = item.is_bust;
  const isDry    = item.bust_direction === 'dry';
  const isWet    = item.bust_direction === 'wet';
  const nwp      = item.forecast_value ?? 75.4;
  const obs      = item.observed_value ?? 14.7;
  const dist     = item.distance ?? item.weight * 10;
  const maxMm    = Math.max(nwp, obs, 10);

  const outcomeColor = isBust
    ? (isDry ? '#C99A3A' : '#B8203A')
    : '#4C8C6D';

  return (
    <div
      className="shrink-0 rounded overflow-hidden cursor-pointer transition-all"
      style={{
        width: 140,
        background: '#162238',
        border: `1px solid ${expanded ? ACCENT : '#1E2D42'}`,
      }}
      onClick={() => setExpanded(!expanded)}
      title={expanded ? 'Click to collapse' : 'Click to see synoptic mechanism'}
    >
      {/* Card header */}
      <div className="px-2.5 pt-2 pb-1">
        <div className="flex items-start justify-between">
          <div>
            <div className="font-mono text-[0.6rem] font-bold" style={{ color: '#E9EDF5' }}>
              #{rank} {item.date}
            </div>
            <div className="font-mono text-[0.55rem] mt-0.5" style={{ color: '#8792A8' }}>
              d = {dist.toFixed(2)}
            </div>
          </div>
          {/* Outcome badge */}
          <span
            className="font-mono text-[0.5rem] font-bold px-1 py-px rounded"
            style={{
              background: `${outcomeColor}20`,
              color: outcomeColor,
              border: `1px solid ${outcomeColor}40`,
            }}
          >
            {isBust ? (isDry ? 'BUST dry' : isWet ? 'BUST wet' : 'BUST') : 'no bust'}
          </span>
        </div>
      </div>

      <div style={{ borderTop: '1px solid #1E2D42', padding: '6px 10px' }}>
        {/* Paired Model vs Observed bars */}
        <div className="space-y-1.5">
          <div>
            <div className="flex items-center justify-between font-mono text-[0.5rem] mb-0.5" style={{ color: '#8792A8' }}>
              <span>MODEL</span>
              <span style={{ color: '#E9EDF5' }}>{nwp.toFixed(1)}</span>
            </div>
            <div className="rounded" style={{ background: '#0B1220', height: 5 }}>
              <div
                className="rounded"
                style={{ width: `${(nwp / maxMm) * 100}%`, background: '#4FB8E0', height: 5 }}
              />
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between font-mono text-[0.5rem] mb-0.5" style={{ color: '#8792A8' }}>
              <span>OBS</span>
              <span style={{ color: '#E9EDF5' }}>{obs.toFixed(1)}</span>
            </div>
            <div className="rounded" style={{ background: '#0B1220', height: 5 }}>
              <div
                className="rounded"
                style={{ width: `${(obs / maxMm) * 100}%`, background: IMD_BAND_FILL.HIGH, height: 5 }}
              />
            </div>
          </div>
        </div>

        {/* Expanded mechanism (spec: only prose, gated behind 1 click) */}
        {expanded && item.mechanism && (
          <p className="mt-2 font-sans text-[0.55rem] leading-relaxed" style={{ color: '#8792A8' }}>
            {item.mechanism}
          </p>
        )}
        {!expanded && (
          <div className="mt-1.5 flex items-center gap-0.5 font-sans text-[0.5rem]" style={{ color: '#3D536E' }}>
            <ChevronRight className="w-2.5 h-2.5" />
            <span>mechanism</span>
          </div>
        )}
      </div>
    </div>
  );
};

// ── OOD Circuit Breaker banner ─────────────────────────────────────────────
const OODBanner: React.FC = () => (
  <div className="ood-banner rounded p-4 animate-unprecedented">
    <div className="flex items-start gap-3">
      <Shield className="w-6 h-6 shrink-0 mt-0.5" style={{ color: UNPRECEDENTED_CLR }} />
      <div>
        <div
          className="font-sans font-bold text-sm uppercase tracking-wide mb-2"
          style={{ color: UNPRECEDENTED_CLR }}
        >
          UNPRECEDENTED — NO HISTORICAL PRECEDENT FOUND
        </div>
        <p className="font-sans text-xs leading-relaxed" style={{ color: '#B8C5D6' }}>
          Current atmospheric circulation exceeds the 99th-percentile distance across 40 years (1982–2022) of
          IMD gridded reanalysis record. Pratyaya does not invent history it doesn't have.
        </p>
        <p className="font-sans text-xs mt-2" style={{ color: '#8792A8' }}>
          → Rely on live radar and nowcasting for this subdivision.
        </p>
      </div>
    </div>
  </div>
);

// ── Main RegionalDrawer component ──────────────────────────────────────────
export const RegionalDrawer: React.FC<RegionalDrawerProps> = ({
  isOpen,
  onClose,
  card,
  leadDay,
  isLoading = false,
  onOpenBulletin,
}) => {
  if (!isOpen) return null;

  const regionName = card?.region_name || 'Selected Subdivision';
  const elt        = card?.effective_lead_time ?? leadDay;
  const pBust      = card?.p_bust ?? 0;
  const baseRate   = card?.observed_bust_rate_in_band ?? 0.06;
  const isOOD      = card?.unprecedented_flag ?? false;

  // Headline sentence construction (spec Section 4.1)
  const headlinePct    = (pBust * 100).toFixed(0);
  const baseRatePct    = (baseRate * 100).toFixed(0);
  const directionWord  = card?.bust_direction && card.bust_direction !== 'none'
    ? `direction: ${card.bust_direction}` : '';

  const rawHeadline = card?.headline ||
    `Day ${leadDay} forecast behaving like Day ${elt.toFixed(1)} (${headlinePct}% bust probability vs ${baseRatePct}% base rate${directionWord ? '; ' + directionWord : ''})`;
  const headline = rawHeadline
    .replace(/none bias/gi, 'balanced error profile')
    .replace(/none error bias/gi, 'balanced error profile');

  return (
    <>
      {/* Drawer panel — slides in from right, sits above scrubber ruler (C6) */}
      <div
        className="fixed right-0 z-40 overflow-hidden drawer-slide-in"
        style={{
          top: '86px',
          bottom: '96px',
          width: 'min(440px, 90vw)',
          background: '#121B2E',
          borderLeft: '1px solid #1E2D42',
          borderTop: '1px solid #1E2D42',
          borderBottom: '1px solid #1E2D42',
          borderRadius: '4px 0 0 4px',
          boxShadow: '-4px 0 32px rgba(0,0,0,0.5)',
        }}
      >
        <div className="h-full flex flex-col overflow-hidden">
          {/* ── Header ─────────────────────────────────────────────────── */}
          <div
            className="px-4 py-3 flex items-center justify-between shrink-0"
            style={{ borderBottom: '1px solid #1E2D42', background: '#0B1220' }}
          >
            <div>
              <div className="flex items-center gap-2">
                <span className="font-sans font-semibold text-sm" style={{ color: '#E9EDF5' }}>
                  {regionName}
                </span>
                {card?.region_id && (
                  <span className="font-mono text-[0.55rem]" style={{ color: '#8792A8' }}>
                    #{card.region_id}
                  </span>
                )}
              </div>
              <div className="font-mono text-[0.625rem] mt-0.5" style={{ color: '#8792A8' }}>
                Day {leadDay} → Effective Day {elt.toFixed(1)}
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded transition-colors"
              style={{ color: '#8792A8', border: '1px solid #1E2D42' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = '#E9EDF5'; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = '#8792A8'; }}
              title="Close drawer (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* ── Scrollable body ────────────────────────────────────────── */}
          <div className="flex-1 overflow-y-auto">
            {isLoading ? (
              <div className="flex items-center justify-center h-40">
                <div className="w-6 h-6 rounded-full border-2 border-t-transparent animate-spin"
                  style={{ borderColor: ACCENT, borderTopColor: 'transparent' }} />
              </div>
            ) : !card ? (
              <div className="p-6 text-center font-sans text-xs" style={{ color: '#8792A8' }}>
                Select a subdivision to view diagnostics
              </div>
            ) : (
              <div className="p-4 space-y-5">
                {/* ── Headline Sentence (Section 4.1) — Fraunces display serif ── */}
                <div
                  className="font-serif leading-snug"
                  style={{ fontSize: '1.1rem', fontWeight: 600, color: '#E9EDF5', maxWidth: '42ch' }}
                >
                  {headline}
                </div>

                {/* ── Compact Bust Triad (Section 5.1) ─────────────────── */}
                <div>
                  <div
                    className="font-sans text-[0.55rem] uppercase tracking-widest mb-2 font-semibold"
                    style={{ color: '#8792A8' }}
                  >
                    Forecast Bust Triad
                  </div>
                  <BustTriad card={card} />
                </div>

                {/* ── Physical Risk Factors (Section 4.3) ──────────────── */}
                {!isOOD && card.risk_factors.length > 0 && (
                  <div>
                    <div
                      className="font-sans text-[0.55rem] uppercase tracking-widest mb-2 font-semibold flex items-center gap-1.5"
                      style={{ color: '#8792A8' }}
                    >
                      <Wind className="w-3 h-3" />
                      Why Low Confidence
                    </div>
                    <ul className="space-y-1.5">
                      {card.risk_factors.slice(0, 4).map((rf, i) => (
                        <li
                          key={i}
                          className="flex items-start gap-2 font-sans text-xs"
                          style={{ color: '#B8C5D6' }}
                        >
                          <span className="shrink-0 font-mono text-base leading-none" style={{ color: ACCENT }}>
                            {rfGlyph(rf)}
                          </span>
                          {rf}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* ── OOD Banner or Precedent Strip ────────────────────── */}
                {isOOD ? (
                  <OODBanner />
                ) : (
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div
                        className="font-sans text-[0.55rem] uppercase tracking-widest font-semibold"
                        style={{ color: '#8792A8' }}
                      >
                        Atmospheric Memory — Top {card.top_analogues.length} Historical Analogues
                      </div>
                      <div className="font-mono text-[0.5rem]" style={{ color: '#3D536E' }}>
                        closest ◀──▶ furthest
                      </div>
                    </div>

                    {/* Horizontally scrollable strip — ZERO-CLICK USP (spec Section 4.2) */}
                    <div
                      className="flex gap-2.5 overflow-x-auto pb-2"
                      style={{ scrollbarWidth: 'thin', scrollbarColor: '#1E2D42 #0B1220' }}
                    >
                      {card.top_analogues.map((analogue, i) => (
                        <AnalogueCard key={`${analogue.date}-${i}`} item={analogue} rank={i + 1} />
                      ))}
                    </div>
                  </div>
                )}

                {/* ── Provenance footer ────────────────────────────────── */}
                <div
                  className="font-mono text-[0.55rem] pt-3 flex items-center justify-between"
                  style={{ borderTop: '1px solid #1E2D42', color: '#3D536E' }}
                >
                  <span>MODEL: {card.model_version}</span>
                  <span>{card.data_as_of ?? 'IMD 0.25° Archive'}</span>
                </div>
              </div>
            )}
          </div>

          {/* ── Footer Actions ─────────────────────────────────────────── */}
          <div
            className="px-4 py-3 flex items-center justify-between gap-3 shrink-0"
            style={{ borderTop: '1px solid #1E2D42', background: '#0B1220' }}
          >
            <button
              onClick={onClose}
              className="px-3 py-2 rounded font-sans text-xs font-semibold transition-colors"
              style={{ background: '#162238', color: '#8792A8', border: '1px solid #1E2D42' }}
            >
              Close (Esc)
            </button>
            {onOpenBulletin && (
              <button
                onClick={() => { onClose(); onOpenBulletin(); }}
                className="flex-1 py-2 rounded font-sans text-xs font-semibold transition-colors text-center"
                style={{ background: ACCENT, color: '#0B1220' }}
                onMouseEnter={(e) => { e.currentTarget.style.background = '#6EC8E8'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = ACCENT; }}
              >
                Export into Shift Bulletin ↗
              </button>
            )}
          </div>
        </div>
      </div>

      {/* No backdrop scrim — map must stay visible and un-dimmed (spec Section 2.3) */}
    </>
  );
};
