/**
 * App.tsx — Phase FINAL: Master-detail layout (64% map + 36% right-rail + bottom scrubber)
 * Per spec Sections 1, 3.1, 7.1
 *
 * Default view: [NationalConfidenceMap 64%] [TriageQueue 36%] + bottom HorizonScrubber
 * Subdivision click: RegionalDrawer slides in from right, map stays visible
 * Other tabs: full-width dedicated views
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';

// ── Phase 2: New TopNav ─────────────────────────────────────────────────────
import { TopNav, NavTabId, ActiveCycle, ReplayFrameState } from './components/TopNav';

// ── Phase 3: New NationalConfidenceMap ─────────────────────────────────────
import { NationalConfidenceMap } from './components/NationalConfidenceMap';

// ── Phase 4: New HorizonScrubber ───────────────────────────────────────────
import { HorizonScrubber } from './components/HorizonScrubber';

// ── Phase 5: New TriageQueue ───────────────────────────────────────────────
import { TriageQueue } from './components/TriageQueue';

// ── Phase 6: New RegionalDrawer ────────────────────────────────────────────
import { RegionalDrawer } from './components/RegionalDrawer';

// ── Phase 7: New dedicated views ───────────────────────────────────────────
import { BustTriadRoom } from './views/BustTriadRoom';
import { TimeMachineView } from './views/TimeMachineView';
import { RegimeAtlasView } from './views/RegimeAtlasView';
import { VerificationHubView } from './views/VerificationHubView';

// ── Phase 7.5: A4 Bulletin Modal ───────────────────────────────────────────
import { A4BulletinModal } from './components/A4BulletinModal';

// ── Existing supporting components (not replaced) ──────────────────────────
import { JuryTourModal } from './components/JuryTourModal';
import { ErrorBoundary } from './components/ErrorBoundary';

// ── API layer (preserved exactly per spec) ─────────────────────────────────
import { apiClient } from './api/client';
import {
  ConfidenceMap,
  ErrorProneAreas,
  ForecastCard as IForecastCard,
  ModelStatus,
  RegimesResponse,
  ReplayScenario,
  ScenarioSummary,
} from './api/types';

export const App: React.FC = () => {
  // ── Primary Navigation Tab ────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<NavTabId>('operations');

  // ── Global Operational State (Brief v4: ActiveCycle owned once at top) ────
  const [activeCycle, setActiveCycle] = useState<ActiveCycle>({ date: '2002-06-07', leadDay: 6 });
  const [variable]                    = useState<string>('rainfall');
  const [selectedRegionId, setSelectedRegionId] = useState<string>('35');

  const date = activeCycle.date;
  const leadDay = activeCycle.leadDay;

  const setLeadDay = (leadOrUpdater: number | ((prev: number) => number)) => {
    setActiveCycle((prev) => ({
      ...prev,
      leadDay: typeof leadOrUpdater === 'function' ? leadOrUpdater(prev.leadDay) : leadOrUpdater,
    }));
  };

  const setDate = (newDate: string) => {
    setActiveCycle((prev) => ({ ...prev, date: newDate }));
  };

  // ── Drawer & Modal States ─────────────────────────────────────────────────
  const [isDrawerOpen,   setIsDrawerOpen]   = useState<boolean>(false);
  const [isBulletinOpen, setIsBulletinOpen] = useState<boolean>(false);
  const [isJuryTourOpen, setIsJuryTourOpen] = useState<boolean>(false);

  // ── API Data States ───────────────────────────────────────────────────────
  const [confidenceMap, setConfidenceMap] = useState<ConfidenceMap | null>(null);
  const [reviewQueue,   setReviewQueue]   = useState<ErrorProneAreas | null>(null);
  const [forecastCard,  setForecastCard]  = useState<IForecastCard | null>(null);
  const [modelStatus,   setModelStatus]   = useState<ModelStatus | null>(null);
  const [regimes,       setRegimes]       = useState<RegimesResponse | null>(null);

  // ── Replay States ─────────────────────────────────────────────────────────
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('july2002_break');
  const [replayScenario,     setReplayScenario]     = useState<ReplayScenario | null>(null);
  const [replayScenarios,    setReplayScenarios]    = useState<ScenarioSummary[]>([]);
  const [isLoadingReplay,    setIsLoadingReplay]    = useState<boolean>(false);

  // ── Loading Flags ─────────────────────────────────────────────────────────
  const [isLoadingCard, setIsLoadingCard] = useState<boolean>(false);
  const [isLive,        setIsLive]        = useState<boolean>(false);
  const [forceOffline,  setForceOffline]  = useState<boolean>(apiClient.getIsForceOffline());

  // ── Derived: unified national low/critical count (C2 single source of truth) ──
  const nationalLowCriticalCount = useMemo(() => {
    if (!confidenceMap?.regions || confidenceMap.regions.length === 0) return 0;
    return confidenceMap.regions.filter(
      (r) => r.confidence_band === 'CRITICAL' || r.confidence_band === 'LOW' || r.unprecedented_flag
    ).length;
  }, [confidenceMap]);

  // ── Selected region name (for header breadcrumb & ghost-pin annotation) ──
  const selectedRegionName = useMemo(() => {
    if (!confidenceMap?.regions) return undefined;
    return confidenceMap.regions.find((r) => r.region_id === selectedRegionId)?.name;
  }, [confidenceMap, selectedRegionId]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      switch (e.key) {
        case '1': setActiveTab('operations'); break;
        case '2': setActiveTab('triad'); break;
        case '3': setActiveTab('replay'); break;
        case '4': setActiveTab('regimes'); break;
        case '5': setActiveTab('verification'); break;
        case 't': case 'T': setIsJuryTourOpen((p) => !p); break;
        case 'b': case 'B': setIsBulletinOpen((p) => !p); break;
        case 'ArrowLeft':  setLeadDay((p) => Math.max(1, p - 1)); break;
        case 'ArrowRight': setLeadDay((p) => Math.min(10, p + 1)); break;
        case 'Escape':
          setIsBulletinOpen(false);
          setIsJuryTourOpen(false);
          setIsDrawerOpen(false);
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // ── Initial health check ─────────────────────────────────────────────────
  useEffect(() => {
    apiClient.checkHealth().then((live) => setIsLive(live));
    apiClient.getReplayScenarios().then(setReplayScenarios);
  }, []);

  // ── API calls ─────────────────────────────────────────────────────────────
  const fetchMap = useCallback(async (lead: number, curVar: string, curDate: string) => {
    if (activeTab === 'replay') return;
    try {
      const data = await apiClient.getConfidenceMap(lead, curDate, curVar);
      setConfidenceMap(data);
    } catch (err) { console.error('Map fetch error:', err); }
  }, [activeTab]);

  const fetchQueue = useCallback(async (lead?: number, curDate: string = date) => {
    if (activeTab === 'replay') return;
    try {
      const data = await apiClient.getReviewQueue(lead, curDate, 15);
      setReviewQueue(data);
    } catch (err) { console.error('Queue fetch error:', err); }
  }, [activeTab, date]);

  const fetchCard = useCallback(async (regionId: string, lead: number, curDate: string) => {
    setIsLoadingCard(true);
    try {
      const data = await apiClient.getForecastRegion(regionId, lead, curDate);
      setForecastCard(data);
    } catch (err) { console.error('Card fetch error:', err); }
    finally { setIsLoadingCard(false); }
  }, []);

  const loadReplayScenario = useCallback(async (scenId: string) => {
    setIsLoadingReplay(true);
    try {
      const data = await apiClient.fetchReplay(scenId, 7);
      setReplayScenario(data);
      // NOTE: Brief v4 — Never overwrite ActiveCycle from replay frames!
    } catch (err) { console.error('Replay fetch error:', err); }
    finally { setIsLoadingReplay(false); }
  }, []);

  // ── Reactive triggers ─────────────────────────────────────────────────────
  useEffect(() => {
    if (activeTab !== 'replay') {
      fetchMap(leadDay, variable, date);
      fetchQueue(leadDay, date);
    }
  }, [activeTab, leadDay, variable, date, fetchMap, fetchQueue]);

  useEffect(() => {
    if (selectedRegionId) fetchCard(selectedRegionId, leadDay, date);
  }, [selectedRegionId, leadDay, date, fetchCard]);

  useEffect(() => {
    apiClient.getModelStatus().then(setModelStatus).catch(console.error);
    apiClient.getRegimes().then(setRegimes).catch(console.error);
  }, []);

  // Sync replay frame → map + queue
  useEffect(() => {
    if (activeTab === 'replay' && replayScenario) {
      const frame = replayScenario.frames.find((f) => f.lead_day === leadDay) || replayScenario.frames[0];
      if (frame) {
        setConfidenceMap(frame.confidence_map);
        setReviewQueue({
          date: frame.cycle_date,
          lead_day: frame.lead_day,
          total_items: frame.triage_queue.length,
          queue: frame.triage_queue,
          model_version: replayScenario.model_version,
          data_as_of: frame.confidence_map.data_as_of,
          is_live: replayScenario.is_live,
        });
        if (!selectedRegionId && frame.top_risk_region_id) setSelectedRegionId(frame.top_risk_region_id);
      }
    }
  }, [activeTab, replayScenario, leadDay, selectedRegionId]);

  useEffect(() => {
    if (activeTab === 'replay') loadReplayScenario(selectedScenarioId);
  }, [activeTab, selectedScenarioId, loadReplayScenario]);

  // ── User interaction handlers ─────────────────────────────────────────────
  const handleSelectRegion = (regionId: string) => {
    setSelectedRegionId(regionId);
    setIsDrawerOpen(true);
    fetchCard(regionId, leadDay, date);
  };

  const handleSelectLeadDay = (newLead: number) => {
    setLeadDay(newLead);
    fetchMap(newLead, variable, date);
    if (selectedRegionId) fetchCard(selectedRegionId, newLead, date);
  };

  const handleSelectRegionAndLead = (regionId: string, lead?: number) => {
    const targetLead = lead ?? leadDay;
    setSelectedRegionId(regionId);
    if (lead !== undefined) {
      setLeadDay(lead);
      fetchMap(lead, variable, date);
    }
    fetchCard(regionId, targetLead, date);
    setIsDrawerOpen(true);
  };

  const handleCycleDateSelect = (newDate: string) => {
    setDate(newDate);
    fetchMap(leadDay, variable, newDate);
    fetchQueue(undefined, newDate);
    if (selectedRegionId) fetchCard(selectedRegionId, leadDay, newDate);
  };

  const handleToggleForceOffline = () => {
    const next = !forceOffline;
    setForceOffline(next);
    apiClient.setForceOffline(next);
    if (!next) {
      apiClient.checkHealth().then(setIsLive);
    } else {
      setIsLive(false);
    }
    if (activeTab === 'replay') {
      loadReplayScenario(selectedScenarioId);
    } else {
      fetchMap(leadDay, variable, date);
      fetchQueue(undefined, date);
      if (selectedRegionId) fetchCard(selectedRegionId, leadDay, date);
    }
    apiClient.getModelStatus().then(setModelStatus).catch(console.error);
  };

  const queueItems = reviewQueue?.queue ?? [];

  // ── Derived: isolated Replay Frame for Time Machine (Section 1 Brief v4) ───
  const currentReplayFrame: ReplayFrameState | undefined = useMemo(() => {
    if (activeTab !== 'replay' || !replayScenario) return undefined;
    const f = replayScenario.frames.find((frame) => frame.lead_day === leadDay) || replayScenario.frames[0];
    return {
      cycleDate: f?.cycle_date || replayScenario.cycle_date || '2002-07-07',
      leadDay: f?.lead_day ?? leadDay,
      scenarioTitle: replayScenario.title,
    };
  }, [activeTab, replayScenario, leadDay]);

  return (
    <ErrorBoundary>
      {/* Root: full-viewport dark-canvas layout */}
      <div
        className="flex flex-col h-screen overflow-hidden"
        style={{ background: '#0B1220', color: '#E9EDF5', fontFamily: "'Mukta', sans-serif" }}
      >
        {/* ── Sticky TopNav (Row 1+2) ─────────────────────────────────────── */}
        <TopNav
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          activeCycle={activeCycle}
          onSelectCycleDate={handleCycleDateSelect}
          replayFrame={currentReplayFrame}
          criticalCount={nationalLowCriticalCount}
          isLive={isLive}
          forceOffline={forceOffline}
          onToggleForceOffline={handleToggleForceOffline}
          onOpenBulletin={() => setIsBulletinOpen(true)}
          onOpenJuryTour={() => setIsJuryTourOpen(true)}
          selectedRegionName={selectedRegionName}
        />

        {/* ── Main content area ───────────────────────────────────────────── */}
        <main className="flex-1 flex flex-col overflow-hidden" style={{ position: 'relative' }}>

          {/* ── OPERATIONS VIEW: 64% map + 36% triage + bottom scrubber ──── */}
          {activeTab === 'operations' && (
            <div className="flex-1 flex flex-col overflow-hidden">

              {/* The two-column master area */}
              <div className="flex flex-1 overflow-hidden" style={{ minHeight: 0 }}>

                {/* Left: 64% — The Map Instrument */}
                <div
                  className="flex flex-col overflow-hidden"
                  style={{
                    flex: '0 0 64%',
                    width: '64%',
                    minWidth: 0,
                    position: 'relative',
                  }}
                >
                  <NationalConfidenceMap
                    leadDay={leadDay}
                    regions={confidenceMap?.regions ?? []}
                    selectedRegionId={selectedRegionId}
                    onSelectRegion={handleSelectRegion}
                    variable={variable}
                    onSelectLeadDay={handleSelectLeadDay}
                    lowCriticalCount={nationalLowCriticalCount}
                  />
                </div>

                {/* Right: 36% — Triage Queue Rail */}
                <div
                  className="flex flex-col overflow-hidden"
                  style={{ flex: '0 0 36%', width: '36%', minWidth: 0 }}
                >
                  <TriageQueue
                    queue={queueItems}
                    totalSubdivisions={confidenceMap?.regions?.length ?? 36}
                    selectedRegionId={selectedRegionId}
                    onSelectRegion={handleSelectRegionAndLead}
                    currentLeadDay={leadDay}
                    onViewAll={() => setActiveTab('triad')}
                    nationalLowCriticalCount={nationalLowCriticalCount}
                  />
                </div>
              </div>

              {/* Bottom: Horizon Scrubber ruler — z-30 keeps ticks reachable regardless of drawer */}
              <div className="shrink-0 relative z-30" data-testid="horizon-scrubber">
                <HorizonScrubber
                  leadDay={leadDay}
                  onSelectLeadDay={handleSelectLeadDay}
                  nominalLeadDay={leadDay}
                  effectiveLeadTime={forecastCard?.effective_lead_time}
                  selectedRegionName={selectedRegionName}
                  regions={confidenceMap?.regions ?? []}
                />
              </div>
            </div>
          )}

          {/* ── BUST TRIAD ROOM ───────────────────────────────────────────── */}
          {activeTab === 'triad' && (
            <div className="flex-1 overflow-y-auto">
              <BustTriadRoom
                currentCard={forecastCard}
                currentLeadDay={leadDay}
                currentDate={date}
                regions={confidenceMap?.regions ?? []}
                onSelectRegion={(regId) => {
                  setSelectedRegionId(regId);
                  fetchCard(regId, leadDay, date);
                }}
              />
            </div>
          )}

          {/* ── TIME MACHINE (Replay) ─────────────────────────────────────── */}
          {activeTab === 'replay' && (
            <div className="flex-1 flex flex-col overflow-hidden">
              <TimeMachineView
                scenario={replayScenario}
                scenarios={replayScenarios}
                selectedScenarioId={selectedScenarioId}
                onSelectScenario={setSelectedScenarioId}
                currentLeadDay={leadDay}
                onLeadDayChange={handleSelectLeadDay}
                isLoading={isLoadingReplay}
                onSelectRegion={handleSelectRegion}
              />
            </div>
          )}

          {/* ── REGIME ATLAS ─────────────────────────────────────────────── */}
          {activeTab === 'regimes' && (
            <div className="flex-1 flex flex-col overflow-hidden">
              <RegimeAtlasView
                regimes={regimes}
                status={modelStatus}
                isLoading={!regimes}
              />
            </div>
          )}

          {/* ── VERIFICATION HUB ─────────────────────────────────────────── */}
          {activeTab === 'verification' && (
            <div className="flex-1 flex flex-col overflow-hidden">
              <VerificationHubView status={modelStatus} />
            </div>
          )}

          {/* ── Regional Drawer — no backdrop (map stays un-dimmed) ──────── */}
          <RegionalDrawer
            isOpen={isDrawerOpen}
            onClose={() => setIsDrawerOpen(false)}
            card={forecastCard}
            leadDay={leadDay}
            isLoading={isLoadingCard}
            onOpenBulletin={() => setIsBulletinOpen(true)}
          />
        </main>

        {/* ── A4 Bulletin Modal ─────────────────────────────────────────── */}
        <A4BulletinModal
          isOpen={isBulletinOpen}
          onClose={() => setIsBulletinOpen(false)}
          card={forecastCard}
          activeCycle={activeCycle}
          criticalQueue={queueItems.filter(
            (q) => q.confidence_band === 'LOW' || q.confidence_band === 'CRITICAL'
          )}
        />

        {/* ── Jury Tour Modal (keep existing) ─────────────────────────── */}
        {isJuryTourOpen && (
          <JuryTourModal
            isOpen={isJuryTourOpen}
            onClose={() => setIsJuryTourOpen(false)}
            onSelectKeralaBenchmark={() => {
              setActiveTab('operations');
              setActiveCycle({ date: '2002-06-07', leadDay: 6 });
              setSelectedRegionId('35');
              setIsJuryTourOpen(false);
            }}
          />
        )}
      </div>
    </ErrorBoundary>
  );
};
