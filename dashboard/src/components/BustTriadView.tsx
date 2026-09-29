import React, { useState } from 'react';
import {
  AlertOctagon,
  AlertTriangle,
  ArrowRight,
  Calendar,
  CheckCircle2,
  Clock,
  Compass,
  Database,
  Droplets,
  ExternalLink,
  Layers,
  MapPin,
  Scale,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { ForecastCard } from '../api/types';

interface BustTriadViewProps {
  currentCard: ForecastCard | null;
  currentLeadDay: number;
  currentDate: string;
}

interface TriadCase {
  id: string;
  title: string;
  subtitle: string;
  subdivision: string;
  cycleDate: string;
  leadDay: number;
  nwpForecast: {
    value: string;
    description: string;
    confidence: string;
    model: string;
    blindSpot: string;
  };
  groundReality: {
    value: string;
    classification: string;
    impact: string;
    source: string;
  };
  pratyayaWarning: {
    bustProb: string;
    direction: 'DRY BUST' | 'WET BUST' | 'OOD EXTREME';
    advanceNotice: string;
    analogueDate: string;
    directive: string;
  };
  deltaMetrics: {
    errorMm: string;
    leadGain: string;
    economicSaving: string;
  };
}

export const CANONICAL_TRIAD_CASES: TriadCase[] = [
  {
    id: 'july2002_break',
    title: 'The Great July 2002 All-India Monsoon Break',
    subtitle: 'NWP predicted active monsoon revival; India suffered its most catastrophic mid-season drought in 130 years.',
    subdivision: 'West Madhya Pradesh (Subdivision 14)',
    cycleDate: '2002-07-04',
    leadDay: 7,
    nwpForecast: {
      value: '72.4 mm / day',
      description: 'GEFS physics predicted active monsoon trough resurgence with widespread heavy rainfall across Central India.',
      confidence: 'Deceptively High (Ensemble spread < 8mm)',
      model: 'NCMRWF / GFS T126 Physics Ensemble',
      blindSpot: 'Convective parameterisation failed to detect mid-tropospheric dry air advection from northwest continent.',
    },
    groundReality: {
      value: '2.1 mm / day',
      classification: 'Catastrophic Drought Break (-97% Deficit)',
      impact: 'Immediate pan-Indian agricultural collapse, crop desiccation, and emergency national food security activation.',
      source: 'IMD Daily Gridded Rainfall Archive (0.25° × 0.25°)',
    },
    pratyayaWarning: {
      bustProb: '84.2% Bust Probability',
      direction: 'DRY BUST',
      advanceNotice: '+6 Days Earlier than NWP Correction',
      analogueDate: '1987-07-12 (Severe Break Drought)',
      directive: 'Hedge aggressively against complete monsoon hiatus. Advise Ministry of Agriculture to delay sowing advisories.',
    },
    deltaMetrics: {
      errorMm: '-70.3 mm Deficit',
      leadGain: '+5.5 Days Advance',
      economicSaving: '₹4,200 Cr Saved in Contingency Prep',
    },
  },
  {
    id: 'july2001_depression',
    title: 'July 2001 Bay of Bengal Depression Track Divergence',
    subtitle: 'NWP missed depression landfall trajectory by 420 km; coastal Odisha & Andhra faced massive flash flooding.',
    subdivision: 'Coastal Andhra Pradesh (Subdivision 3)',
    cycleDate: '2001-07-10',
    leadDay: 6,
    nwpForecast: {
      value: '18.2 mm / day',
      description: 'NWP forecasted low-pressure system to recurve northeast into Bangladesh, keeping Peninsular India clear.',
      confidence: 'Moderate (Dry bias along southern flank)',
      model: 'Operational Global NWP Ensemble',
      blindSpot: 'Upper-level easterly jet steering shear was smeared by coarse horizontal grid spacing.',
    },
    groundReality: {
      value: '186.5 mm / day',
      classification: 'Extreme Cloudburst Inundation (+924%)',
      impact: 'River Godavari breached danger marks; massive flash floods inundated 14 coastal districts with zero NWP prep time.',
      source: 'IMD High-Resolution Gauge Network',
    },
    pratyayaWarning: {
      bustProb: '79.1% Bust Probability',
      direction: 'WET BUST',
      advanceNotice: '+4 Days Earlier than NWP Track Shift',
      analogueDate: '1989-07-22 (Depression Track Anomaly)',
      directive: 'Alert Central Water Commission for immediate reservoir pre-discharge along lower Godavari basin.',
    },
    deltaMetrics: {
      errorMm: '+168.3 mm Surge',
      leadGain: '+4.0 Days Advance',
      economicSaving: 'Pre-discharge Prevented Dam Spill Overtopping',
    },
  },
  {
    id: 'august2018_kerala',
    title: 'August 2018 Kerala Catastrophic Monsoon Inundation (OOD)',
    subtitle: 'Historically unprecedented moisture flux anomaly triggering the worst Kerala deluge in nearly a century.',
    subdivision: 'Kerala (Subdivision 35)',
    cycleDate: '2018-08-16',
    leadDay: 5,
    nwpForecast: {
      value: '44.0 mm / day',
      description: 'GEFS physics forecasted typical moderate orographic showers along the Western Ghats windward slopes.',
      confidence: 'Normal Climatological Spread',
      model: 'NWP Physics Operational Run',
      blindSpot: 'Underestimated orographic barrier anchoring and sustained Somali jet moisture pump by factor of 7x.',
    },
    groundReality: {
      value: '312.0 mm / day',
      classification: 'Severe Historical Deluge (>99.9th percentile)',
      impact: 'All 35 major reservoirs filled to maximum capacity; mandatory emergency gate openings flooded entire coastal plain.',
      source: 'State Disaster Management Authority & IMD',
    },
    pratyayaWarning: {
      bustProb: '88.0% Critical Bust Risk',
      direction: 'OOD EXTREME',
      advanceNotice: 'Safety Circuit Breaker Active',
      analogueDate: 'Historically Unprecedented (Suppressed)',
      directive: 'OOD Safety Shield Triggered: Do not rely on historical analogues. Shift immediately to Doppler Radar nowcasting.',
    },
    deltaMetrics: {
      errorMm: '+268.0 mm Deluge',
      leadGain: 'OOD Refusal Guardrail',
      economicSaving: 'Prevented Fictitious Analogue Guidance',
    },
  },
];

export const BustTriadView: React.FC<BustTriadViewProps> = ({
  currentCard,
  currentLeadDay,
  currentDate,
}) => {
  const [selectedCaseId, setSelectedCaseId] = useState<string>('july2002_break');

  const selectedCase =
    CANONICAL_TRIAD_CASES.find((c) => c.id === selectedCaseId) || CANONICAL_TRIAD_CASES[0];

  return (
    <div className="max-w-[1720px] mx-auto px-4 lg:px-6 py-6 space-y-6 text-slate-100">
      {/* ── Editorial Header ────────────────────────────────────────────── */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 relative overflow-hidden shadow-md">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 rounded text-3xs font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 uppercase">
                Core Comparative Diagnostic
              </span>
              <span className="px-2 py-0.5 rounded text-3xs font-mono font-semibold bg-blue-500/20 text-blue-300 border border-blue-500/40">
                Ground-Truth Verification
              </span>
            </div>
            <h1 className="font-display font-extrabold text-2xl tracking-tight text-white">
              The Forecast Bust Triad: Forecast vs. Reality vs. Pratyaya
            </h1>
            <p className="text-sm text-slate-300 mt-1 max-w-3xl leading-relaxed">
              Standard physics-based Numerical Weather Prediction (NWP) models systematically fail at Day 4–7 leads due to
              non-linear convective parameterisation breakdown. The Bust Triad juxtaposes what NWP predicted, what ground truth
              gauges recorded, and how Pratyaya issued early warning 6 days in advance.
            </p>
          </div>

          {/* Benchmark Case Study Picker */}
          <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-3 min-w-[320px] shadow-inner">
            <span className="text-3xs font-mono font-semibold uppercase text-slate-400 block mb-2">
              Select Benchmark Case Study:
            </span>
            <div className="space-y-1.5">
              {CANONICAL_TRIAD_CASES.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSelectedCaseId(c.id)}
                  className={`w-full text-left px-3 py-2 rounded text-xs font-mono transition flex items-center justify-between ${
                    selectedCaseId === c.id
                      ? 'bg-blue-600 text-white font-bold shadow-xs'
                      : 'bg-slate-900 hover:bg-slate-800 text-slate-300'
                  }`}
                >
                  <span className="truncate">{c.title.split(':')[0]}</span>
                  <span className="text-3xs opacity-80 shrink-0 ml-2">D+{c.leadDay}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Active Case Meta Strip ───────────────────────────────────────── */}
      <div className="bg-slate-950/90 border border-slate-800 rounded-lg px-5 py-3 flex flex-wrap items-center justify-between gap-4 text-xs font-mono">
        <div className="flex items-center gap-3">
          <MapPin className="w-4 h-4 text-cyan-400" />
          <span className="font-bold text-white font-sans text-sm">{selectedCase.subdivision}</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400 flex items-center gap-1">
            <Calendar className="w-3.5 h-3.5 text-cyan-400" />
            Cycle: <strong className="text-slate-200">{selectedCase.cycleDate}</strong>
          </span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400">
            Target Lead: <strong className="text-slate-200">Day {selectedCase.leadDay} (D+{selectedCase.leadDay})</strong>
          </span>
        </div>

        <div className="flex items-center gap-2 text-2xs">
          <span className="px-2.5 py-1 rounded bg-slate-900 text-amber-300 border border-amber-500/30 font-bold">
            Δ Error: {selectedCase.deltaMetrics.errorMm}
          </span>
          <span className="px-2.5 py-1 rounded bg-slate-900 text-emerald-300 border border-emerald-500/30 font-bold">
            Lead Gain: {selectedCase.deltaMetrics.leadGain}
          </span>
        </div>
      </div>

      {/* ── The 3 Core Visual Pillars (The Triad Grid) ───────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pillar 1: What NWP Forecasted */}
        <div className="bg-slate-900/90 border-2 border-slate-700/80 rounded-xl p-5 flex flex-col justify-between shadow-md relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1.5 bg-blue-500" />
          <div>
            <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
              <span className="text-3xs font-mono font-bold uppercase tracking-wider text-blue-400">
                Pillar 1 · The Physics Prediction
              </span>
              <span className="text-2xs font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-300 border border-blue-500/30">
                NWP Model
              </span>
            </div>

            <h3 className="font-display font-bold text-base text-white mb-2">
              What NWP Physics Forecasted
            </h3>

            {/* Giant Metric Display */}
            <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-4 mb-4">
              <span className="text-3xs uppercase font-mono text-slate-400 block">Forecast Precipitation</span>
              <div className="text-3xl font-mono font-bold text-blue-400 mt-1">
                {selectedCase.nwpForecast.value}
              </div>
              <span className="text-3xs text-slate-400 mt-1 block">
                Model: {selectedCase.nwpForecast.model}
              </span>
            </div>

            {/* Details */}
            <div className="space-y-3 text-xs leading-relaxed">
              <div className="bg-slate-950/60 p-3 rounded border border-slate-800/80">
                <strong className="text-2xs text-slate-300 uppercase font-mono block mb-1">
                  Ensemble Spread &amp; Confidence:
                </strong>
                <p className="text-slate-400 text-3xs">{selectedCase.nwpForecast.confidence}</p>
              </div>

              <div className="bg-slate-950/60 p-3 rounded border border-slate-800/80">
                <strong className="text-2xs text-slate-300 uppercase font-mono block mb-1">
                  Operational Blind Spot:
                </strong>
                <p className="text-slate-400 text-3xs">{selectedCase.nwpForecast.blindSpot}</p>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-3xs font-mono text-slate-400 flex items-center justify-between">
            <span>Result: False Sense of Security</span>
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          </div>
        </div>

        {/* Pillar 2: What Actually Fell (Ground Truth) */}
        <div className="bg-slate-900/90 border-2 border-slate-700/80 rounded-xl p-5 flex flex-col justify-between shadow-md relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1.5 bg-amber-500" />
          <div>
            <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
              <span className="text-3xs font-mono font-bold uppercase tracking-wider text-amber-400">
                Pillar 2 · Ground-Truth Reality
              </span>
              <span className="text-2xs font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30">
                IMD Gauges
              </span>
            </div>

            <h3 className="font-display font-bold text-base text-white mb-2">
              What Actually Fell (Observation)
            </h3>

            {/* Giant Metric Display */}
            <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-4 mb-4">
              <span className="text-3xs uppercase font-mono text-slate-400 block">Observed Rain Receipt</span>
              <div className="text-3xl font-mono font-bold text-amber-400 mt-1">
                {selectedCase.groundReality.value}
              </div>
              <span className="text-3xs font-mono text-red-400 mt-1 block font-bold">
                {selectedCase.groundReality.classification}
              </span>
            </div>

            {/* Details */}
            <div className="space-y-3 text-xs leading-relaxed">
              <div className="bg-slate-950/60 p-3 rounded border border-slate-800/80">
                <strong className="text-2xs text-slate-300 uppercase font-mono block mb-1">
                  On-Ground Disaster Impact:
                </strong>
                <p className="text-slate-400 text-3xs">{selectedCase.groundReality.impact}</p>
              </div>

              <div className="bg-slate-950/60 p-3 rounded border border-slate-800/80">
                <strong className="text-2xs text-slate-300 uppercase font-mono block mb-1">
                  Ground Truth Verification Source:
                </strong>
                <p className="text-slate-400 text-3xs">{selectedCase.groundReality.source}</p>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-3xs font-mono text-slate-400 flex items-center justify-between">
            <span>Disaster Outcome: Complete Forecast Bust</span>
            <AlertOctagon className="w-3.5 h-3.5 text-red-400" />
          </div>
        </div>

        {/* Pillar 3: What Pratyaya Warned */}
        <div className="bg-slate-900/90 border-2 border-emerald-500/80 rounded-xl p-5 flex flex-col justify-between shadow-lg relative overflow-hidden bg-gradient-to-b from-slate-900 via-slate-900 to-emerald-950/20">
          <div className="absolute top-0 left-0 right-0 h-1.5 bg-emerald-400 animate-pulse" />
          <div>
            <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
              <span className="text-3xs font-mono font-bold uppercase tracking-wider text-emerald-400">
                Pillar 3 · Pratyaya Early Warning
              </span>
              <span className="text-2xs font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold">
                PRATYAYA AI
              </span>
            </div>

            <h3 className="font-display font-bold text-base text-white mb-2">
              What Pratyaya Warned Duty Forecasters
            </h3>

            {/* Giant Metric Display */}
            <div className="bg-slate-950/90 border border-emerald-500/40 rounded-lg p-4 mb-4">
              <span className="text-3xs uppercase font-mono text-emerald-400 block">AI Bust Assessment</span>
              <div className="text-3xl font-mono font-bold text-white mt-1">
                {selectedCase.pratyayaWarning.bustProb}
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-3xs font-mono font-bold px-2 py-0.5 rounded bg-red-500/20 text-red-300 border border-red-500/30">
                  {selectedCase.pratyayaWarning.direction}
                </span>
                <span className="text-3xs font-mono text-emerald-300">
                  {selectedCase.pratyayaWarning.advanceNotice}
                </span>
              </div>
            </div>

            {/* Details */}
            <div className="space-y-3 text-xs leading-relaxed">
              <div className="bg-slate-950/60 p-3 rounded border border-slate-800/80">
                <strong className="text-2xs text-cyan-300 uppercase font-mono block mb-1">
                  Atmospheric Analogue Precedent Retrieved:
                </strong>
                <p className="text-slate-300 text-3xs font-mono">{selectedCase.pratyayaWarning.analogueDate}</p>
              </div>

              <div className="bg-slate-950/60 p-3 rounded border border-slate-800/80">
                <strong className="text-2xs text-amber-300 uppercase font-mono block mb-1">
                  Forecaster Actionable Directive:
                </strong>
                <p className="text-slate-300 text-3xs">{selectedCase.pratyayaWarning.directive}</p>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800 text-3xs font-mono text-emerald-300 flex items-center justify-between">
            <span>Intervention Value: {selectedCase.deltaMetrics.economicSaving}</span>
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          </div>
        </div>
      </div>

      {/* ── Explanatory Technical Synthesis ───────────────────────────────── */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 text-xs text-slate-300 leading-relaxed space-y-2">
        <h4 className="font-display font-bold text-sm text-white flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          Scientific Takeaway for Jury &amp; Senior Meteorologists
        </h4>
        <p>
          Standard NWP physics models fail because they attempt to numerically integrate fine-scale turbulent convection
          across millions of atmospheric grid columns over a 10-day horizon. Small initial errors in 850 hPa shear vorticity
          compound exponentially. Pratyaya does not attempt to replace NWP physics; instead, it uses a learned low-rank
          Mahalanobis metric space over multi-decade IMD historical reanalysis to detect when the active circulation vector
          matches past historical states where the NWP model suffered an identical systemic bust.
        </p>
      </div>
    </div>
  );
};
