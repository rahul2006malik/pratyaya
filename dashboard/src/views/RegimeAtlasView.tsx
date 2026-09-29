/**
 * RegimeAtlasView — Phase 7 wrapper per spec Section 7.5
 */
import React from 'react';
import { RegimesView } from '../components/RegimesView';
import type { ModelStatus, RegimesResponse } from '../api/types';

interface RegimeAtlasViewProps {
  regimes: RegimesResponse | null;
  status?: ModelStatus | null;
  isLoading?: boolean;
}

export const RegimeAtlasView: React.FC<RegimeAtlasViewProps> = (props) => (
  <div className="flex-1 flex flex-col" style={{ background: '#0B1220', color: '#E9EDF5', overflowY: 'auto' }}>
    <div className="px-6 pt-6 pb-2">
      <h2 className="font-sans font-bold text-xl" style={{ color: '#E9EDF5' }}>Regime Atlas</h2>
      <p className="font-sans text-xs mt-1" style={{ color: '#8792A8' }}>
        Synoptic precursor regime clusters — Mahalanobis space discovery
      </p>
    </div>
    <RegimesView {...props} />
  </div>
);
