/**
 * A4BulletinModal — Phase 7 clean-slate rebuild per spec Section 7.6
 *
 * Grayscale-safe printable A4 shift bulletin.
 * bust_direction is carried by texture glyphs (not hue) so it survives B&W print.
 * Includes: date header, national triage table, model version footer.
 */
import React, { useRef } from 'react';
import { Download, Printer, X } from 'lucide-react';
import { ForecastCard, QueueItem } from '../api/types';
import { CANONICAL_MODEL_VERSION } from '../data/mock_data';
import { ActiveCycle } from './TopNav';

interface A4BulletinModalProps {
  isOpen: boolean;
  onClose: () => void;
  card: ForecastCard | null;
  activeCycle: ActiveCycle;
  criticalQueue?: QueueItem[];
}

const BAND_WORD: Record<string, string> = {
  HIGH:     'HIGH CONFIDENCE',
  MODERATE: 'MODERATE WATCH',
  LOW:      'LOW CONFIDENCE',
  CRITICAL: 'CRITICAL RISK',
};

const DIRECTION_GLYPH: Record<string, string> = {
  wet:  '↗ WET',
  dry:  '↘ DRY',
  none: '— BALANCED',
};

export const A4BulletinModal: React.FC<A4BulletinModalProps> = ({
  isOpen,
  onClose,
  card,
  activeCycle,
  criticalQueue = [],
}) => {
  const bulletinRef = useRef<HTMLDivElement>(null);

  if (!isOpen) return null;

  const date = activeCycle.date;
  const leadDay = activeCycle.leadDay;

  const handlePrint = () => window.print();

  const regionName = card?.region_name || 'National Summary';
  const pBust      = card?.p_bust ?? 0;
  const elt        = card?.effective_lead_time ?? leadDay;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.75)' }}>
      <div
        className="relative flex flex-col"
        style={{
          width: '794px',
          maxWidth: '95vw',
          maxHeight: '90vh',
          background: '#FAFAF8',
          borderRadius: 6,
          boxShadow: '0 24px 64px rgba(0,0,0,0.6)',
          overflow: 'hidden',
        }}
      >
        {/* ── Modal chrome (not printed) ──────────────────────────────── */}
        <div
          className="no-print flex items-center justify-between px-4 py-3 shrink-0"
          style={{ background: '#121B2E', borderBottom: '1px solid #1E2D42' }}
        >
          <span className="font-sans text-sm font-semibold text-white">
            A4 Shift Bulletin Preview
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded font-sans text-xs font-semibold"
              style={{ background: '#4FB8E0', color: '#0B1220' }}
            >
              <Printer className="w-3.5 h-3.5" />
              Print / Save PDF
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded"
              style={{ color: '#8792A8', border: '1px solid #1E2D42' }}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* ── A4 content area (printed) ───────────────────────────────── */}
        <div className="flex-1 overflow-y-auto">
          <div
            ref={bulletinRef}
            className="print-bulletin-container"
            style={{
              width: '100%',
              padding: '32px 40px',
              background: '#FFFFFF',
              color: '#0F172A',
              fontFamily: "'Mukta', sans-serif",
              fontSize: '11px',
              lineHeight: 1.5,
            }}
          >
            {/* Header block */}
            <div style={{ borderBottom: '2px solid #0F172A', paddingBottom: 12, marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'Fraunces, Georgia, serif', letterSpacing: '-0.01em' }}>
                    PRATYAYA · प्रत्यय
                  </div>
                  <div style={{ fontSize: 10, color: '#5B6274', marginTop: 2 }}>
                    Synoptic Analogue Diagnostic Engine · Operational Forecast Risk Bulletin
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 13, fontWeight: 700 }}>
                    {date}
                  </div>
                  <div style={{ fontSize: 10, color: '#5B6274', marginTop: 1 }}>
                    Forecast Cycle · Lead D+{leadDay}
                  </div>
                </div>
              </div>
            </div>

            {/* National headline numeral */}
            <div style={{ marginBottom: 20, padding: '12px 16px', border: '1px solid #E2E8F0', borderRadius: 6 }}>
              <div style={{ fontSize: 10, color: '#5B6274', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>
                NATIONAL SITUATION
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span style={{ fontSize: 32, fontWeight: 700, fontFamily: 'IBM Plex Mono, monospace', color: criticalQueue.length > 0 ? '#B8203A' : '#4C8C6D' }}>
                  {criticalQueue.length}
                </span>
                <span style={{ fontSize: 13, color: '#14171F' }}>
                  of 36 subdivisions flagged LOW / CRITICAL for Day {leadDay}
                </span>
              </div>
            </div>

            {/* Triage table */}
            {criticalQueue.length > 0 && (
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: '#14171F', marginBottom: 8, borderBottom: '1px solid #E2E8F0', paddingBottom: 4 }}>
                  PRIORITY TRIAGE QUEUE — RANKED BY RISK WEIGHT
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #0F172A', background: '#F8FAFC' }}>
                      {['Rank', 'Subdivision', 'Band', 'Direction', 'P(bust)', 'ELT', 'Priority Score'].map((h) => (
                        <th key={h} style={{ padding: '4px 8px', textAlign: 'left', fontWeight: 600, fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#5B6274' }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {criticalQueue.slice(0, 12).map((item, i) => (
                      <tr key={item.region_id} style={{ borderBottom: '1px solid #E2E8F0', background: i % 2 === 0 ? '#FFFFFF' : '#F8FAFC' }}>
                        <td style={{ padding: '4px 8px', fontFamily: 'IBM Plex Mono, monospace', fontWeight: 700, color: '#0F172A' }}>
                          #{i + 1}
                          {item.rank && item.rank !== i + 1 && (
                            <span style={{ fontSize: 8, color: '#8792A8', marginLeft: 4 }}>
                              (Nat #{item.rank})
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '4px 8px', fontWeight: 600, color: '#14171F' }}>
                          {item.region_name}
                        </td>
                        <td style={{ padding: '4px 8px', color: '#14171F' }}>
                          {BAND_WORD[item.confidence_band] || item.confidence_band}
                        </td>
                        <td style={{ padding: '4px 8px', fontFamily: 'IBM Plex Mono, monospace', fontWeight: 700 }}>
                          {DIRECTION_GLYPH[item.bust_direction] || '—'}
                        </td>
                        <td style={{ padding: '4px 8px', fontFamily: 'IBM Plex Mono, monospace', fontWeight: 700 }}>
                          {(item.p_bust * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '4px 8px', fontFamily: 'IBM Plex Mono, monospace' }}>
                          D{item.effective_lead_time.toFixed(1)}
                        </td>
                        <td style={{ padding: '4px 8px', fontFamily: 'IBM Plex Mono, monospace', fontWeight: 700 }}>
                          {item.priority_score.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: '8.5px', color: '#64748B', marginTop: 6, fontStyle: 'italic', lineHeight: 1.4 }}>
                  * Rank indicates priority within this shift's action queue. (Nat #) denotes national all-subdivision ranking across all 36 meteorological regions based on decision-weighted bust risk.
                </div>
              </div>
            )}

            {/* Selected region deep-dive */}
            {card && !card.unprecedented_flag && (
              <div style={{ marginBottom: 16, padding: 12, border: '1px solid #E2E8F0', borderRadius: 6 }}>
                <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8, color: '#14171F' }}>
                  REGIONAL DETAIL — {(card.region_name || '').toUpperCase()}
                </div>
                <p style={{ fontSize: 12, fontFamily: 'Fraunces, Georgia, serif', fontWeight: 500, lineHeight: 1.5, marginBottom: 8, color: '#14171F', maxWidth: '65ch' }}>
                  {(card.headline || `Day ${leadDay} forecast behaving like Day ${card.effective_lead_time.toFixed(1)} — ${(card.p_bust * 100).toFixed(0)}% bust probability.`).replace(/none bias/gi, 'balanced error profile').replace(/none error bias/gi, 'balanced error profile')}
                </p>
                {card.risk_factors.length > 0 && (
                  <ul style={{ paddingLeft: 16, fontSize: 10, color: '#5B6274', lineHeight: 1.6 }}>
                    {card.risk_factors.slice(0, 3).map((rf, i) => (
                      <li key={i}>{rf}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {/* Footer */}
            <div style={{ borderTop: '1px solid #E2E8F0', paddingTop: 8, display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#8792A8', fontFamily: 'IBM Plex Mono, monospace' }}>
              <span>MODEL: {card?.model_version || CANONICAL_MODEL_VERSION}</span>
              <span>DATA AS OF: {card?.data_as_of || `${date}T00:00:00Z`}</span>
              <span>FOR OFFICIAL USE — DUTY FORECASTER SHIFT HANDOVER</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
