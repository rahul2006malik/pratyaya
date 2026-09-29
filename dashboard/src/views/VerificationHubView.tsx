/**
 * VerificationHubView — Phase 7 wrapper per spec Section 7.7
 */
import React from 'react';
import { VerificationEvidenceView } from '../components/VerificationEvidenceView';
import type { ModelStatus } from '../api/types';

interface VerificationHubViewProps {
  status: ModelStatus | null;
}

export const VerificationHubView: React.FC<VerificationHubViewProps> = ({ status }) => (
  <div className="flex-1 flex flex-col" style={{ background: '#0B1220', color: '#E9EDF5', overflowY: 'auto' }}>
    <div className="px-6 pt-6 pb-2">
      <h2 className="font-sans font-bold text-xl" style={{ color: '#E9EDF5' }}>Verification Hub</h2>
      <p className="font-sans text-xs mt-1" style={{ color: '#8792A8' }}>
        Probabilistic calibration, Brier Score, BSS — complete model evidence trail
      </p>
    </div>
    <VerificationEvidenceView status={status} />
  </div>
);
