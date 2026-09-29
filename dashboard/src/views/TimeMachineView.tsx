/**
 * TimeMachineView — Phase 7 wrapper per spec Section 7.4
 * Re-uses the existing ReplayView with spec-compliant heading/dark wrapper.
 */
import React from 'react';
import { ReplayView } from '../components/ReplayView';
import type { ReplayScenario, ScenarioSummary } from '../api/types';

interface TimeMachineViewProps {
  scenario: ReplayScenario | null;
  scenarios: ScenarioSummary[];
  selectedScenarioId: string;
  onSelectScenario: (scenarioId: string) => void;
  currentLeadDay: number;
  onLeadDayChange: (lead: number) => void;
  isLoading: boolean;
  onSelectRegion?: (regionId: string) => void;
}

export const TimeMachineView: React.FC<TimeMachineViewProps> = (props) => (
  <div className="flex-1 flex flex-col" style={{ background: '#0B1220', color: '#E9EDF5', overflowY: 'auto' }}>
    <div className="px-6 pt-6 pb-2">
      <h2 className="font-sans font-bold text-xl" style={{ color: '#E9EDF5' }}>Time Machine</h2>
      <p className="font-sans text-xs mt-1" style={{ color: '#8792A8' }}>
        Step through canonical forecast scenarios frame-by-frame
      </p>
    </div>
    <ReplayView {...props} />
  </div>
);
