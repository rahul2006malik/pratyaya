import React, { useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Compass,
  FileSpreadsheet,
  Layers,
  Wind,
  X,
  Zap,
  AlertOctagon,
} from 'lucide-react';

interface JuryTourModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectKeralaBenchmark?: () => void;
}

export const JuryTourModal: React.FC<JuryTourModalProps> = ({
  isOpen,
  onClose,
  onSelectKeralaBenchmark,
}) => {
  const [currentStep, setCurrentStep] = useState(0);

  if (!isOpen) return null;

  const tourSteps = [
    {
      step: 1,
      badge: 'Problem Statement',
      title: 'Why Do Numerical Models Fail at Day 4–7?',
      icon: <AlertOctagon className="w-5 h-5 text-red-600" />,
      borderColor: 'border-red-200',
      bgColor: 'bg-red-50/70',
      description:
        'Standard Numerical Weather Prediction (NWP) models — ECMWF, GFS, NCMRWF — run atmospheric physics equations on supercomputers. At medium-range leads (Day 4–7), non-linear chaos and convective parameterisation breakdown cause systematic blind spots.',
      keyPoints: [
        'Models frequently forecast light rain (14 mm) while actual observed downpours reach 88+ mm catastrophic cloudburst events.',
        'Overconfidence delays floodgate operations and early warning for civil authorities.',
        'Ensemble spreads often remain deceptively narrow even when a large forecast bust is imminent.',
      ],
      actionLabel: 'How Pratyaya Solves This',
    },
    {
      step: 2,
      badge: 'Core Architecture',
      title: 'Pratyaya Fuses Multi-Year Synoptic Memory',
      icon: <Layers className="w-5 h-5 text-blue-600" />,
      borderColor: 'border-blue-200',
      bgColor: 'bg-blue-50/70',
      description:
        'Instead of relying purely on raw physics model spread, Pratyaya indexes 3 verified monsoon seasons (2000–2002) of IMD canonical reanalysis data into a high-dimensional synoptic analogue space.',
      keyPoints: [
        'Extracts 850 hPa wind vector shears, geopotential heights, and moisture flux convergence as features.',
        'Computes learned Low-Rank Mahalanobis distance against 120 initialization cycles.',
        'Identifies exact atmospheric fingerprints where numerical models systematically failed in past seasons.',
      ],
      actionLabel: 'See the Output Format',
    },
    {
      step: 3,
      badge: 'Operational Metric',
      title: 'Effective Lead Time: Translating Probabilities into Action',
      icon: <Zap className="w-5 h-5 text-amber-600" />,
      borderColor: 'border-amber-200',
      bgColor: 'bg-amber-50/70',
      description:
        'Abstract percentages confuse duty officers. Pratyaya translates calibrated bust probability into Effective Lead Time (ELT) — telling forecasters when a Day 6 forecast is behaving like Day 12.',
      keyPoints: [
        'Green Zone: High numerical confidence. Issued with routine standard operational procedure.',
        'Amber Zone: Monitor catchment spread; multi-model divergence requires caution.',
        'Red Zone: Confidence collapse. Forecaster is provided with Top 3 historical analogue cases to justify downward or upward warning revision.',
      ],
      actionLabel: 'View the Verification Proof',
    },
    {
      step: 4,
      badge: 'Scientific Verification',
      title: 'Verified on 43,200 Real NOAA & IMD Pairs (Zero Synthetic Data)',
      icon: <CheckCircle2 className="w-5 h-5 text-emerald-600" />,
      borderColor: 'border-emerald-200',
      bgColor: 'bg-emerald-50/70',
      description:
        'Under strict 3-Fold Purged LOYO-CV across all 3 Southwest Monsoon seasons with 12-day boundary purges, Model M2 achieves PR-AUC = 0.1900 [95% CI: 0.1680, 0.2098], delivering a statistically significant +2.34 percentage point gain over calibrated spread (p <= 0.001).',
      keyPoints: [
        'Beats calibrated logistic spread benchmark B3 (PR-AUC 0.1667) with 95% bootstrap CI excluding zero.',
        'BSS +12.45% positive calibration skill over operational climatology bar.',
        'Decisions D-008 & D-010 guarantee zero synthetic interpolation: 100% empirical reforecast verification.',
      ],
      actionLabel: 'Explore Interactive Workstation',
    },
  ];

  const current = tourSteps[currentStep];

  const handleNext = () => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onClose();
      if (onSelectKeralaBenchmark) onSelectKeralaBenchmark();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-fade-up">
      <div className="bg-white border border-slate-300 w-full max-w-xl rounded-xl shadow-2xl overflow-hidden flex flex-col text-slate-900">
        {/* Header */}
        <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-blue-600" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-900">
              60-Second Guided Jury Tour · Project Pratyaya
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:text-slate-800 hover:bg-slate-200 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Progress bar */}
        <div className="flex h-1 bg-slate-100">
          {tourSteps.map((_, idx) => (
            <div
              key={idx}
              className={`flex-1 transition-all duration-300 ${
                idx <= currentStep ? 'bg-blue-600' : 'bg-slate-200'
              }`}
            />
          ))}
        </div>

        {/* Content */}
        <div className="p-5 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-2xs font-semibold px-2 py-0.5 rounded-full bg-blue-50 border border-blue-200 text-blue-800">
              {current.badge}
            </span>
            <span className="text-2xs text-slate-500 font-mono font-bold">
              {currentStep + 1} / {tourSteps.length}
            </span>
          </div>

          <div className="flex items-start gap-3">
            <div className={`p-2.5 rounded-lg border shrink-0 ${current.borderColor} ${current.bgColor}`}>
              {current.icon}
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900 leading-snug">{current.title}</h2>
              <p className="text-xs text-slate-600 mt-1 leading-relaxed">{current.description}</p>
            </div>
          </div>

          <div className={`p-3.5 rounded-lg border ${current.borderColor} ${current.bgColor}`}>
            <span className="text-3xs text-slate-500 uppercase font-bold tracking-wider block mb-2">Key operational points</span>
            <ul className="space-y-2">
              {current.keyPoints.map((point, pIdx) => (
                <li key={pIdx} className="flex items-start gap-2 text-xs text-slate-800">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0 mt-0.5" />
                  <span className="leading-snug">{point}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-3.5 bg-slate-50 border-t border-slate-200 flex items-center justify-between shrink-0">
          <button onClick={() => currentStep > 0 && setCurrentStep(currentStep - 1)}
            disabled={currentStep === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold text-slate-600 hover:text-slate-900 hover:bg-slate-200 disabled:opacity-30 disabled:cursor-not-allowed transition border border-slate-300">
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Previous</span>
          </button>

          {/* Dot indicators */}
          <div className="flex gap-1.5">
            {tourSteps.map((_, idx) => (
              <button key={idx}
                onClick={() => setCurrentStep(idx)}
                className={`h-1.5 rounded-full transition-all ${
                  idx === currentStep ? 'bg-blue-600 w-5' : 'bg-slate-300 w-1.5 hover:bg-slate-400'
                }`}
              />
            ))}
          </div>

          <button onClick={handleNext}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-md text-xs font-bold bg-blue-600 hover:bg-blue-700 text-white transition shadow-xs">
            <span>{current.actionLabel}</span>
            {currentStep < tourSteps.length - 1 && <ArrowRight className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>
    </div>
  );
};
