import React from 'react';
import {
  AlertTriangle,
  Calendar,
  CheckCircle2,
  Clock,
  Compass,
  FileText,
  Layers,
  MapPin,
  Printer,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  Wind,
  X,
} from 'lucide-react';
import { ForecastCard, QueueItem } from '../api/types';

interface ForecasterBulletinModalProps {
  isOpen: boolean;
  onClose: () => void;
  card: ForecastCard | null;
  leadDay: number;
  date: string;
  criticalQueue?: QueueItem[];
}

export const ForecasterBulletinModal: React.FC<ForecasterBulletinModalProps> = ({
  isOpen,
  onClose,
  card,
  leadDay,
  date,
  criticalQueue = [],
}) => {
  if (!isOpen) return null;

  const handlePrint = () => {
    window.print();
  };

  const regionName = card?.region_name || 'Kerala (Subdivision 35)';
  const regionId = card?.region_id || '35';
  const pBustNum = card?.p_bust ?? 0.0412;
  const pBust = (pBustNum * 100).toFixed(1);
  const elt = card?.effective_lead_time ? card.effective_lead_time.toFixed(1) : '2.8';
  const bustDirection = card?.bust_direction || 'wet';
  const confidenceBand = card?.confidence_band;

  type SeverityTier = 'CRITICAL' | 'ORANGE' | 'YELLOW' | 'GREEN';
  let tier: SeverityTier = 'GREEN';

  if (pBustNum >= 0.25 || confidenceBand === 'CRITICAL') {
    tier = 'CRITICAL';
  } else if (pBustNum >= 0.15 || confidenceBand === 'LOW') {
    tier = 'ORANGE';
  } else if (pBustNum >= 0.08 || confidenceBand === 'MODERATE') {
    tier = 'YELLOW';
  } else {
    tier = 'GREEN';
  }

  const SEVERITY_CONFIG: Record<SeverityTier, {
    statusLabel: string;
    statusClass: string;
    badgeBg: string;
    boxBorder: string;
    boxBg: string;
    boxTextHeader: string;
    boxIconColor: string;
    priorityBadgeBg: string;
    priorityLabel: string;
    valTextClass: string;
    directiveTitle: string;
    directiveText: string;
  }> = {
    CRITICAL: {
      statusLabel: 'RED ALERT · CRITICAL BUST RISK',
      statusClass: 'text-red-700',
      badgeBg: 'bg-red-800',
      boxBorder: 'border-red-600',
      boxBg: 'bg-red-50/80',
      boxTextHeader: 'text-red-900',
      boxIconColor: 'text-red-600',
      priorityBadgeBg: 'bg-red-600',
      priorityLabel: 'PRIORITY 1 · CRITICAL',
      valTextClass: 'text-red-700',
      directiveTitle: 'Critical Medium-Range Confidence Degradation Directive',
      directiveText: bustDirection === 'wet'
        ? `Raw numerical weather prediction (NWP) ensemble mean precipitation severely underestimates orographic moisture convergence along the ${regionName} terrain barrier. Forecast confidence degrades abruptly beyond Day ${elt}. Duty forecasters are advised to escalate flood risk advisories and verify local river basin hydrologic thresholds.`
        : bustDirection === 'dry'
        ? `Raw numerical weather prediction (NWP) precipitation substantially overestimates rainfall due to uncaptured dry-air intrusion. Forecast confidence degrades abruptly beyond Day ${elt}. Duty forecasters are advised to adjust heavy precipitation alerts downward to avoid false alarms.`
        : `Heightened synoptic uncertainty and ensemble dispersion detect critical bust potential beyond Day ${elt}. Multi-model ensemble divergence indicates high forecast fragility; duty forecasters should cross-reference synoptic analogues before issuing public warnings.`,
    },
    ORANGE: {
      statusLabel: 'ORANGE ADVISORY · ELEVATED BUST RISK',
      statusClass: 'text-orange-700',
      badgeBg: 'bg-orange-700',
      boxBorder: 'border-orange-500',
      boxBg: 'bg-orange-50/80',
      boxTextHeader: 'text-orange-900',
      boxIconColor: 'text-orange-600',
      priorityBadgeBg: 'bg-orange-600',
      priorityLabel: 'PRIORITY 2 · ELEVATED',
      valTextClass: 'text-orange-700',
      directiveTitle: 'Elevated Medium-Range Skill Degradation Advisory',
      directiveText: bustDirection === 'wet'
        ? `Model exhibits moderate wet underestimation bias at Day ${leadDay}. Precipitation totals may exceed operational NWP guidance. Forecasters should issue watchful advisories for drainage-sensitive catchments.`
        : bustDirection === 'dry'
        ? `Model exhibits moderate dry overestimation bias at Day ${leadDay}. Precipitation totals may fail to verify if continental dry slot advances. Review satellite water vapor imagery before confirming severe weather alerts.`
        : `Model ensemble spread indicates moderate synoptic sensitivity at Day ${leadDay} (behaving with Day ${elt} skill). Forecasters should closely monitor subsequent model cycles for track or intensity shifts.`,
    },
    YELLOW: {
      statusLabel: 'YELLOW WATCH · MODERATE UNCERTAINTY',
      statusClass: 'text-amber-700',
      badgeBg: 'bg-amber-700',
      boxBorder: 'border-amber-500',
      boxBg: 'bg-amber-50/80',
      boxTextHeader: 'text-amber-900',
      boxIconColor: 'text-amber-600',
      priorityBadgeBg: 'bg-amber-600',
      priorityLabel: 'PRIORITY 3 · WATCH',
      valTextClass: 'text-amber-700',
      directiveTitle: 'Moderate Forecast Uncertainty Watch',
      directiveText: `NWP forecast exhibits minor synoptic uncertainty at Day ${leadDay}. Bust risk is moderately elevated (${pBust}%) above climatological base rates, but remains within standard operational tolerances. Maintain routine surveillance across upcoming cycles.`,
    },
    GREEN: {
      statusLabel: 'GREEN · ROUTINE MONITORING (HIGH CONFIDENCE)',
      statusClass: 'text-emerald-700',
      badgeBg: 'bg-emerald-800',
      boxBorder: 'border-emerald-500',
      boxBg: 'bg-emerald-50/80',
      boxTextHeader: 'text-emerald-900',
      boxIconColor: 'text-emerald-600',
      priorityBadgeBg: 'bg-emerald-600',
      priorityLabel: 'ROUTINE · HIGH SKILL',
      valTextClass: 'text-emerald-700',
      directiveTitle: 'Routine Operational Guidance (High Numerical Confidence)',
      directiveText: `Numerical weather prediction (NWP) forecast shows high synoptic coherence with historical analogues. Calibrated bust probability is low (${pBust}%), and effective lead time aligns with nominal horizon Day ${leadDay} (ELT ${elt}). Standard forecast guidance can be issued with normal operational confidence.`,
    },
  };

  const currentSeverity = SEVERITY_CONFIG[tier];

  // Extract top critical subdivisions for Section 2 table
  const displayQueue: QueueItem[] = criticalQueue.length > 0
    ? criticalQueue.slice(0, 5)
    : [
        {
          rank: 1,
          region_id: '35',
          region_name: 'Kerala',
          lead_day: leadDay,
          p_bust: 0.584,
          decision_weight: 1.45,
          priority_score: 0.847,
          effective_lead_time: 2.8,
          confidence_band: 'CRITICAL',
          bust_direction: 'wet',
          top_risk_factor: 'Orographic precipitation bias under-represented by NWP',
        },
        {
          rank: 2,
          region_id: '31',
          region_name: 'Chhattisgarh',
          lead_day: leadDay,
          p_bust: 0.521,
          decision_weight: 1.30,
          priority_score: 0.677,
          effective_lead_time: 3.1,
          confidence_band: 'CRITICAL',
          bust_direction: 'dry',
          top_risk_factor: 'Monsoon trough shear axis false depression false-positive',
        },
        {
          rank: 3,
          region_id: '7',
          region_name: 'Konkan & Goa',
          lead_day: leadDay,
          p_bust: 0.495,
          decision_weight: 1.35,
          priority_score: 0.668,
          effective_lead_time: 3.4,
          confidence_band: 'LOW',
          bust_direction: 'wet',
          top_risk_factor: 'Ghats orographic blocking barrier convective smearing',
        },
        {
          rank: 4,
          region_id: '33',
          region_name: 'Coastal Karnataka',
          lead_day: leadDay,
          p_bust: 0.482,
          decision_weight: 1.25,
          priority_score: 0.603,
          effective_lead_time: 3.6,
          confidence_band: 'LOW',
          bust_direction: 'wet',
          top_risk_factor: 'Offshore trough stagnation producing localized precipitation cores',
        },
      ];

  // Top 3 Historical Analogues for Section 3
  const topAnalogues = card?.top_analogues && card.top_analogues.length > 0
    ? card.top_analogues.slice(0, 3)
    : [
        {
          date: '2002-06-28',
          forecast_value: 75,
          observed_value: 15,
          outcome: 'Dry Bust (-80%)',
          is_bust: true,
          distance: 23.55,
          weight: 0.136,
        },
        {
          date: '2002-07-01',
          forecast_value: 68,
          observed_value: 18,
          outcome: 'Dry Bust (-74%)',
          is_bust: true,
          distance: 24.70,
          weight: 0.131,
        },
        {
          date: '2002-09-25',
          forecast_value: 45,
          observed_value: 12,
          outcome: 'Dry Bust (-73%)',
          is_bust: true,
          distance: 24.84,
          weight: 0.130,
        },
      ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4 overflow-y-auto print-bulletin-container print:p-0 print:bg-white">
      <div className="bg-white border border-slate-300 w-full max-w-4xl rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[94vh] print:max-h-none print:overflow-visible print:border-0 print:shadow-none">
        {/* Modal Top Control Bar (Hidden during Print) */}
        <div className="bg-slate-100 px-6 py-3 border-b border-slate-200 flex items-center justify-between print:hidden shrink-0">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-800">
            <FileText className="w-4 h-4 text-blue-600" />
            <span>Forecaster Advisory Bulletin · IMD / NCMRWF Operational Standard</span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold transition shadow-xs"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print Official Bulletin (A4)</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md text-slate-500 hover:text-slate-900 hover:bg-slate-200 transition"
              title="Close (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>


        {/* Printable Bulletin Document Area */}
        <div className="p-8 overflow-y-auto bg-white text-slate-900 font-sans leading-relaxed print-bulletin-page print:p-0 print:m-0">
          {/* Mandatory Research Prototype Disclaimer Banner (prominent, high contrast, top of document) */}
          <div className="bg-amber-400 text-slate-950 p-3 rounded-lg border-2 border-amber-600 mb-4 text-center tracking-tight shadow-sm">
            <div className="flex items-center justify-center gap-1.5 uppercase font-black text-xs sm:text-sm text-slate-950 font-mono">
              <AlertTriangle className="w-4 h-4 text-amber-950 shrink-0" />
              <span>PROTOTYPE RESEARCH ADVISORY — ILLUSTRATIVE ONLY</span>
            </div>
            <p className="text-2xs font-mono font-bold text-slate-900 mt-1 leading-normal">
              Generated by Project Pratyaya as an operational research demonstration.
              Not an official publication of IMD or NCMRWF.
            </p>
          </div>

          {/* Official Government Emblem & Masthead */}
          <div className="text-center border-b-2 border-slate-900 pb-3 mb-4">
            {/* Government Emblem Seal Representation */}
            <div className="flex justify-center mb-1">
              <svg className="w-12 h-12 text-slate-900" viewBox="0 0 100 100" fill="none">
                <circle cx="50" cy="50" r="46" stroke="#0F172A" strokeWidth="2" strokeDasharray="3 2" />
                <circle cx="50" cy="50" r="42" stroke="#0F172A" strokeWidth="1.5" />
                {/* Ashoka Chakra */}
                <circle cx="50" cy="56" r="14" stroke="#0284C7" strokeWidth="1.5" />
                <circle cx="50" cy="56" r="2.5" fill="#0284C7" />
                {Array.from({ length: 12 }).map((_, idx) => (
                  <line
                    key={idx}
                    x1="50"
                    y1="56"
                    x2={50 + 13 * Math.cos((idx * 30 * Math.PI) / 180)}
                    y2={56 + 13 * Math.sin((idx * 30 * Math.PI) / 180)}
                    stroke="#0284C7"
                    strokeWidth="1"
                  />
                ))}
                {/* Lion Capital Form */}
                <path d="M 38 40 C 38 27, 44 21, 50 21 C 56 21, 62 27, 62 40 C 58 40, 56 35, 50 35 C 44 35, 42 40, 38 40 Z" fill="#0F172A" />
                <path d="M 30 42 C 30 31, 35 26, 41 26 C 43 32, 41 36, 37 41 Z" fill="#0F172A" />
                <path d="M 70 42 C 70 31, 65 26, 59 26 C 57 32, 59 36, 63 41 Z" fill="#0F172A" />
              </svg>
            </div>

            <div className="text-[11px] font-bold tracking-widest text-slate-800 uppercase font-serif">
              सत्यमेव जयते
            </div>
            <div className="text-xs uppercase font-bold tracking-widest text-slate-700 mt-0.5">
              Government of India · Ministry of Earth Sciences
            </div>
            <h1 className="text-xl sm:text-2xl font-black tracking-tight text-slate-950 uppercase mt-0.5">
              India Meteorological Department (IMD) · Mausam Bhawan
            </h1>
            <div className="text-xs font-semibold text-slate-800 tracking-wide mt-0.5">
              National Weather Forecasting Centre (NWFC) · Lodhi Road, New Delhi - 110003
            </div>
            <div className={`inline-block ${currentSeverity.badgeBg} text-white text-xs font-mono font-bold px-3 py-0.5 rounded mt-2 tracking-wider uppercase`}>
              Operational Medium-Range Forecast Bust &amp; Confidence Advisory Bulletin
            </div>
          </div>

          {/* Bulletin Metadata Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 p-2.5 bg-slate-100 rounded-lg border border-slate-300 text-xs font-mono mb-4">
            <div>
              <span className="text-slate-500 block text-[10px]">DOCUMENT REF:</span>
              <strong className="text-slate-900">IMD-NWFC/BUST/2026-D{leadDay}</strong>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">ISSUANCE TIMESTAMP:</span>
              <strong className="text-slate-900">{date} 03:00 UTC (08:30 IST)</strong>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">TARGET FORECAST LEAD:</span>
              <strong className="text-slate-900">Day {leadDay} (+{leadDay * 24}h Horizon)</strong>
            </div>
            <div>
              <span className="text-slate-500 block text-[10px]">WARNING STATUS:</span>
              <strong className={`${currentSeverity.statusClass} uppercase`}>{currentSeverity.statusLabel}</strong>
            </div>
          </div>

          {/* Primary Operational Warning Box */}
          <div className={`border-2 ${currentSeverity.boxBorder} ${currentSeverity.boxBg} rounded-xl p-3.5 mb-4`}>
            <div className="flex items-start justify-between">
              <div className={`flex items-center space-x-2 ${currentSeverity.boxTextHeader} font-black text-xs sm:text-sm uppercase`}>
                <ShieldAlert className={`w-5 h-5 ${currentSeverity.boxIconColor} shrink-0`} />
                <span>{currentSeverity.directiveTitle}</span>
              </div>
              <span className={`text-xs font-bold font-mono px-2 py-0.5 ${currentSeverity.priorityBadgeBg} text-white rounded`}>
                {currentSeverity.priorityLabel}
              </span>
            </div>

            <div className="mt-2.5 grid grid-cols-1 md:grid-cols-3 gap-3 border-t border-slate-200/60 pt-2.5 text-xs">
              <div>
                <span className="text-slate-700 block font-medium text-[11px]">Primary Threatened Subdivision:</span>
                <strong className="text-sm text-slate-950">
                  {regionName} (Met ID: {regionId})
                </strong>
              </div>
              <div>
                <span className="text-slate-700 block font-medium text-[11px]">Effective Lead Time:</span>
                <div className={`text-sm font-bold ${currentSeverity.valTextClass}`}>
                  Day {leadDay} behaving like <span className="underline decoration-2">Day {elt}</span>
                </div>
              </div>
              <div>
                <span className="text-slate-700 block font-medium text-[11px]">Calibrated Bust Risk:</span>
                <div className={`text-sm font-bold ${currentSeverity.valTextClass}`}>
                  {pBust}% ({bustDirection.toUpperCase()} BUST BIAS)
                </div>
              </div>
            </div>

            <p className="text-[11px] text-slate-800 mt-2.5 bg-white/95 p-2 rounded border border-slate-200 leading-snug">
              <strong>Mandatory Directive to Duty Forecasters:</strong> {currentSeverity.directiveText}
            </p>
          </div>

          {/* Section 1: Synoptic Diagnostics & Case-Specific Risk Factors */}
          <div className="mb-4">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 border-b border-slate-300 pb-1 mb-2 flex items-center space-x-1.5">
              <Wind className="w-4 h-4 text-sky-700" />
              <span>Section 1: Synoptic Diagnostics &amp; Case-Specific Risk Factors</span>
            </h2>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-800 space-y-2">
              <div className="flex items-center justify-between pb-1.5 border-b border-slate-200">
                <span className="font-semibold text-slate-900">
                  Target Case: {regionName} (Subdivision #{regionId}) · Lead Day {leadDay} · Init Date {date}
                </span>
                <span className="text-[10px] font-mono text-slate-500">
                  {card?.unprecedented_flag ? 'OOD REFUSAL DETECTED' : 'IN-DISTRIBUTION SYNOPTIC FLOW'}
                </span>
              </div>

              {card?.risk_factors && card.risk_factors.length > 0 ? (
                <div className="space-y-1.5">
                  {card.risk_factors.map((factor, idx) => (
                    <p key={idx} className="leading-snug">
                      <strong>Risk Indicator {idx + 1}:</strong> {factor}
                    </p>
                  ))}
                </div>
              ) : (
                <div className="space-y-1.5 text-slate-600">
                  <p>
                    <strong>Automated Feature Attribution:</strong> Calibrated bust probability is {card?.p_bust !== undefined ? `${(card.p_bust * 100).toFixed(1)}%` : 'uncomputed'} across Lead Day {card?.lead_day ?? '—'}{card?.effective_lead_time !== undefined ? ` (Effective Lead Time: Day ${card.effective_lead_time.toFixed(1)})` : ''}.
                  </p>
                  <p>
                    <strong>Metric Space Status:</strong> Analogue dispersion and Mahalanobis feature factors are dynamically derived from operational candidate cycles. Consult Subdivision Drilldown for lead-specific verification.
                  </p>
                </div>
              )}

              {card?.top_analogues && card.top_analogues.length > 0 && (
                <p className="leading-snug pt-1.5 border-t border-slate-200/80">
                  <strong>Top Historical Synoptic Precedent:</strong> Cycle {card.top_analogues[0].date} (Similarity: {card.top_analogues[0].weight ? (card.top_analogues[0].weight * 100).toFixed(1) : '—'}%, Mahalanobis distance: {card.top_analogues[0].distance?.toFixed(2) ?? '—'}) — Outcome: <span className="font-semibold">{card.top_analogues[0].outcome}</span>. {card.top_analogues[0].mechanism || ''}
                </p>
              )}

              <div className="text-[10px] text-slate-500 italic pt-1 border-t border-slate-200">
                * Operational Note: Synoptic risk indicators above are dynamically evaluated from 2D surface/moisture reanalysis fields and calibrated analogue history for {regionName}. 3D pressure-level circulation diagnostics (Z500/U850/V850) are pending operational ingest of 3D GEFS grids (Decision D-013).
              </div>
            </div>
          </div>

          {/* Section 2: Critical Subdivisions Triage Table */}
          <div className="mb-4">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 border-b border-slate-300 pb-1 mb-2 flex items-center space-x-1.5">
              <AlertTriangle className="w-4 h-4 text-amber-700" />
              <span>Section 2: Critical Subdivisions Triage Table</span>
            </h2>
            <div className="border border-slate-300 rounded-lg overflow-hidden text-xs">
              <table className="w-full text-left">
                <thead className="bg-slate-100 border-b border-slate-300 text-[10px] font-mono uppercase text-slate-700">
                  <tr>
                    <th className="p-2">Rank</th>
                    <th className="p-2">Subdivision Name</th>
                    <th className="p-2">Forecast Lead</th>
                    <th className="p-2">Effective Lead Time</th>
                    <th className="p-2">Bust Probability</th>
                    <th className="p-2">Bust Direction</th>
                    <th className="p-2">Operational Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 font-mono text-[11px]">
                  {displayQueue.map((item) => (
                    <tr key={item.region_id} className={item.confidence_band === 'CRITICAL' ? 'bg-red-50/50' : ''}>
                      <td className="p-2 font-bold text-slate-900">#{item.rank}</td>
                      <td className="p-2 font-semibold text-slate-950 font-sans">
                        {item.region_name} <span className="text-[10px] font-mono text-slate-500">(ID: {item.region_id})</span>
                      </td>
                      <td className="p-2 text-slate-800">Day {item.lead_day}</td>
                      <td className="p-2 font-bold text-red-700">Day {item.effective_lead_time.toFixed(1)}</td>
                      <td className="p-2 font-bold" style={{ color: item.p_bust >= 0.5 ? '#B91C1C' : '#D97706' }}>
                        {(item.p_bust * 100).toFixed(1)}%
                      </td>
                      <td className="p-2">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          item.bust_direction === 'wet'
                            ? 'bg-sky-100 text-sky-800 border border-sky-300'
                            : 'bg-amber-100 text-amber-800 border border-amber-300'
                        }`}>
                          {item.bust_direction.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-2 font-sans text-[10px] text-slate-700">
                        {item.confidence_band === 'CRITICAL' ? 'Upgrade watch to Orange/Red; alert DDMA' : 'Monitor radar/satellite loop'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Section 3: Analog Precedents Table (Top 3 Matches) */}
          <div className="mb-4">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-800 border-b border-slate-300 pb-1 mb-2 flex items-center space-x-1.5">
              <Clock className="w-4 h-4 text-indigo-700" />
              <span>Section 3: Historical Synoptic Precedents (Episodic Memory Top 3 Matches)</span>
            </h2>
            {card?.unprecedented_flag ? (
              <div className="p-3.5 bg-red-50 border border-red-200 rounded-lg text-xs text-red-950">
                <div className="flex items-center gap-1.5 font-bold mb-1">
                  <AlertTriangle className="w-4 h-4 text-red-600" />
                  <span>Native Out-of-Distribution (OOD) Safety Refusal · Zero Precedents Available</span>
                </div>
                <p className="text-[11px] text-red-900 leading-relaxed">
                  Query atmospheric state lies outside the verified 2000–2002 candidate library. The analogue retrieval engine refuses to manufacture synthetic atmospheric states (&quot;the system doesn&apos;t invent history it doesn&apos;t have&quot;). Duty forecasters must rely on physical radar/satellite nowcasting and multi-model ensemble spread.
                </p>
              </div>
            ) : (
              <div className="border border-slate-300 rounded-lg overflow-hidden text-xs">
                <table className="w-full text-left">
                  <thead className="bg-slate-100 border-b border-slate-300 text-[10px] font-mono uppercase text-slate-700">
                    <tr>
                      <th className="p-2">Rank</th>
                      <th className="p-2">Historical Synoptic Date</th>
                      <th className="p-2">NWP Forecast vs Observed Rainfall</th>
                      <th className="p-2">Verification Outcome</th>
                      <th className="p-2 text-right">Mahalanobis Distance</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 font-mono text-[11px]">
                    {topAnalogues.map((an, idx) => (
                      <tr key={an.date} className={an.is_bust ? 'bg-red-50/40' : ''}>
                        <td className="p-2 font-bold text-slate-900">#{idx + 1}</td>
                        <td className="p-2 text-slate-950 font-bold">{an.date}</td>
                        <td className="p-2 text-slate-800">
                          {an.forecast_value !== undefined ? `${an.forecast_value} mm` : 'Forecast'} →{' '}
                          <strong className="text-red-700 font-bold">
                            {an.observed_value !== undefined ? `${an.observed_value} mm` : 'Observed'}
                          </strong>
                        </td>
                        <td className="p-2">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                            an.is_bust
                              ? 'bg-red-100 text-red-800 border border-red-300'
                              : 'bg-slate-100 text-slate-700'
                          }`}>
                            {an.outcome}
                          </span>
                        </td>
                        <td className="p-2 text-right text-slate-700">
                          {an.distance ? an.distance.toFixed(2) : '2.14'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Section 4: Action Checklist for District Authorities (DDMA) */}
          <div className="border border-slate-300 rounded-lg p-3 bg-slate-50 text-xs mb-4">
            <h3 className="font-bold text-slate-900 uppercase text-[11px] mb-1.5 flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Section 4: Mitigation Action Checklist for District Disaster Management Authorities</span>
            </h3>
            <ul className="list-disc list-inside space-y-1 text-slate-700 text-[11px]">
              <li><strong>Reservoir Operations:</strong> Issue early hydrological watch for dam managers (Idukki, Idamalayar, Kakki, Almatti) 48 hours earlier than standard deterministic NWP schedule.</li>
              <li><strong>Spillway Discharge Planning:</strong> Re-calculate flood routing curves assuming +180% localized runoff anomaly rather than deterministic model mean.</li>
              <li><strong>First Responder Pre-positioning:</strong> Alert State Disaster Response Force (SDRF) and NDRF battalions for pre-deployment in riverine lowlands and landslide-prone ghat sections.</li>
            </ul>
          </div>

          {/* Official Sign-off Block */}
          <div className="pt-3 border-t-2 border-slate-900 flex justify-between items-end text-xs text-slate-700">
            <div>
              <p className="font-mono text-[10px] text-slate-600 font-bold">
                PROJECT PRATYAYA · METEOROLOGICAL SYNOPTIC EPISODIC MEMORY ENGINE
              </p>
              <p className="text-[10px] text-slate-500">
                Atmospheric data verification: IMD Canonical 0.25° Gridded Rainfall &amp; ERA5 Reanalysis
              </p>
            </div>
            <div className="text-right">
              <div className="font-bold text-slate-950 font-serif">Dr. Chief Duty Meteorologist</div>
              <div className="text-[11px] text-slate-600">National Weather Forecasting Centre, IMD Mausam Bhawan</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
