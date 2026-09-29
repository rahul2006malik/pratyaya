/**
 * NationalConfidenceMap — Phase 3 clean-slate rebuild per spec Sections 3.1, 3.3, 8
 *
 * Two independent visual channels:
 *   - Fill hue     → confidence_band (IMD Green/Amber/Rust/Red)
 *   - Fill texture  → bust_direction (SVG hatch: / for wet, \ for dry)
 *   - unprecedented_flag → pulsing violet override
 *
 * Island inset viewports for Lakshadweep (ID 1) and Andaman & Nicobar (ID 2).
 * 100% offline TopoJSON — zero external tile servers.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { feature } from 'topojson-client';
import { ConfidenceBand, SubdivisionForecast } from '../api/types';
import topoData from '../data/india_subdivisions_topo.json';
import { RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';

// ── Section 6.2 IMD Band Colors ───────────────────────────────────────────
export const IMD_BAND_FILL: Record<ConfidenceBand, string> = {
  HIGH:     '#4C8C6D',
  MODERATE: '#C99A3A',
  LOW:      '#B8551A',
  CRITICAL: '#B8203A',
};
const IMD_UNPRECEDENTED = '#7C5CBF';
const ACCENT = '#4FB8E0';

interface HoverInfo {
  regionId: string;
  name: string;
  pBust: number;
  elt: number;
  band: ConfidenceBand;
  direction?: string;
  unprecedented?: boolean;
  x: number;
  y: number;
}

interface NationalConfidenceMapProps {
  leadDay: number;
  regions: SubdivisionForecast[];
  selectedRegionId?: string;
  onSelectRegion: (regionId: string) => void;
  variable?: string;
  onSelectLeadDay?: (lead: number) => void;
  lowCriticalCount?: number;
}

export const NationalConfidenceMap: React.FC<NationalConfidenceMapProps> = ({
  leadDay,
  regions,
  selectedRegionId,
  onSelectRegion,
  variable = 'rainfall',
  lowCriticalCount: propLowCriticalCount,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef       = useRef<SVGSVGElement>(null);
  const gRef         = useRef<SVGGElement>(null);

  const [hoverInfo,  setHoverInfo]  = useState<HoverInfo | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 620 });

  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  // Index forecasts by region_id
  const forecastMap = useMemo(() => {
    const m = new Map<string, SubdivisionForecast>();
    regions.forEach((r) => m.set(r.region_id, r));
    return m;
  }, [regions]);

  // Extract GeoJSON features from TopoJSON
  const geojsonFeatures = useMemo(() => {
    try {
      const geo: any = feature(topoData as any, (topoData as any).objects.subdivisions);
      const feats: any[] = geo.features || [];
      feats.forEach((feat: any) => {
        const polys = feat.geometry.type === 'Polygon'
          ? [feat.geometry.coordinates]
          : feat.geometry.coordinates;
        polys?.forEach((poly: any) => {
          poly.forEach((ring: any) => {
            if (d3.geoArea({ type: 'Polygon', coordinates: [ring] }) > 2 * Math.PI) ring.reverse();
          });
        });
      });
      return feats;
    } catch (e) {
      console.error('TopoJSON unpack error:', e);
      return [];
    }
  }, []);

  // Responsive resize
  useEffect(() => {
    const obs = new ResizeObserver(() => {
      if (containerRef.current) {
        const { clientWidth, clientHeight } = containerRef.current;
        if (clientWidth > 0 && clientHeight > 0) {
          setDimensions({ width: clientWidth, height: clientHeight });
        }
      }
    });
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // D3 zoom & pan
  useEffect(() => {
    if (!svgRef.current || !gRef.current) return;
    const svg = d3.select(svgRef.current);
    const g   = d3.select(gRef.current);
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.7, 8])
      .on('zoom', (ev) => g.attr('transform', ev.transform));
    zoomRef.current = zoom;
    svg.call(zoom);
    return () => { svg.on('.zoom', null); };
  }, []);

  const handleResetZoom = () => {
    if (svgRef.current && zoomRef.current)
      d3.select(svgRef.current).transition().duration(450)
        .call(zoomRef.current.transform, d3.zoomIdentity);
  };
  const handleZoomIn = () => {
    if (svgRef.current && zoomRef.current)
      d3.select(svgRef.current).transition().duration(260).call(zoomRef.current.scaleBy, 1.35);
  };
  const handleZoomOut = () => {
    if (svgRef.current && zoomRef.current)
      d3.select(svgRef.current).transition().duration(260).call(zoomRef.current.scaleBy, 1 / 1.35);
  };

  // Projection & path generator (area-weighted equal-area per spec)
  const { pathGen, proj } = useMemo(() => {
    const fc: any = { type: 'FeatureCollection', features: geojsonFeatures };
    const projection = d3.geoMercator().fitSize([dimensions.width - 20, dimensions.height - 32], fc);
    return { pathGen: d3.geoPath().projection(projection), proj: projection };
  }, [geojsonFeatures, dimensions]);

  // Island inset data (spec Section 8 — Lakshadweep ID 1, A&N ID 2)
  const islandInsets = useMemo(() => {
    if (!proj) return [];
    const w = 104;
    const h = 84;
    const bottomBase = dimensions.height - 18;
    return [
      { id: '1', label: 'Lakshadweep', center: [73.0, 10.5] as [number, number], insetX: 16, insetY: Math.max(220, bottomBase - h * 2 - 8), insetW: w, insetH: h },
      { id: '2', label: 'Andaman & Nicobar', center: [92.7, 11.5] as [number, number], insetX: 16, insetY: Math.max(312, bottomBase - h), insetW: w, insetH: h },
    ];
  }, [proj, dimensions.height]);

  // SVG pattern IDs
  const WET_HATCH_ID  = 'hatch-wet';
  const DRY_HATCH_ID  = 'hatch-dry';
  const OOD_HATCH_ID  = 'hatch-ood';
  const GLOW_FILTER_ID = 'select-glow';

  const getFill = (feat: any): { base: string; pattern?: string } => {
    const regId = String(feat.properties.region_id);
    const fc = forecastMap.get(regId);
    if (!fc) return { base: '#162238' };
    if (fc.unprecedented_flag) return { base: IMD_UNPRECEDENTED, pattern: OOD_HATCH_ID };
    const base = IMD_BAND_FILL[fc.confidence_band] ?? '#162238';
    if (fc.bust_direction === 'wet') return { base, pattern: WET_HATCH_ID };
    if (fc.bust_direction === 'dry') return { base, pattern: DRY_HATCH_ID };
    return { base };
  };

  // Count LOW/CRITICAL subdivisions (unified with TriageQueue via prop)
  const lowCriticalCount = propLowCriticalCount !== undefined
    ? propLowCriticalCount
    : regions.filter(
        (r) => r.confidence_band === 'LOW' || r.confidence_band === 'CRITICAL' || r.unprecedented_flag
      ).length;

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden" style={{ background: '#0B1220' }}>

      {/* ── Top-left status badge ─────────────────────────────────────────── */}
      <div className="absolute top-3 left-3 z-10 flex flex-col gap-1.5 pointer-events-none">
        <div
          className="px-3 py-1.5 rounded font-sans text-xs"
          style={{ background: 'rgba(18,27,46,0.92)', border: '1px solid #1E2D42', backdropFilter: 'blur(8px)' }}
        >
          <div className="font-semibold" style={{ color: '#E9EDF5' }}>
            National Forecast Confidence Map
          </div>
          <div className="font-mono text-[0.625rem] mt-0.5" style={{ color: '#8792A8' }}>
            Lead D{leadDay} · {variable === 'rainfall' ? 'Precipitation' : 'T2m'} · 36 IMD Subdivisions
          </div>
        </div>

        {lowCriticalCount > 0 && (
          <div
            className="px-3 py-1.5 rounded font-sans text-xs font-semibold"
            style={{ background: 'rgba(184,32,58,0.15)', border: '1px solid rgba(184,32,58,0.4)', color: '#B8203A' }}
          >
            {lowCriticalCount} of 36 subdivisions LOW / CRITICAL
          </div>
        )}
      </div>

      {/* ── Top-right zoom controls ───────────────────────────────────────── */}
      <div className="absolute top-3 right-3 z-10 flex flex-col gap-1">
        {[
          { fn: handleZoomIn,    icon: <ZoomIn  className="w-3.5 h-3.5" />, title: 'Zoom in' },
          { fn: handleZoomOut,   icon: <ZoomOut className="w-3.5 h-3.5" />, title: 'Zoom out' },
          { fn: handleResetZoom, icon: <RotateCcw className="w-3.5 h-3.5" />, title: 'Reset zoom' },
        ].map(({ fn, icon, title }, i) => (
          <button
            key={i}
            onClick={fn}
            title={title}
            className="p-1.5 rounded transition-colors"
            style={{ background: 'rgba(18,27,46,0.9)', border: '1px solid #1E2D42', color: '#8792A8' }}
          >
            {icon}
          </button>
        ))}
      </div>

      {/* ── Bottom-right legend ───────────────────────────────────────────── */}
      <div
        className="absolute bottom-3 right-3 z-10 p-2 rounded"
        style={{ background: 'rgba(18,27,46,0.92)', border: '1px solid #1E2D42', backdropFilter: 'blur(8px)' }}
      >
        <div className="font-sans text-[0.55rem] uppercase tracking-wider mb-1.5" style={{ color: '#8792A8' }}>
          Confidence Band
        </div>
        {([
          ['HIGH',         IMD_BAND_FILL.HIGH,     'p < 0.10'],
          ['MODERATE',     IMD_BAND_FILL.MODERATE,  '0.10–0.25'],
          ['LOW',          IMD_BAND_FILL.LOW,       '0.25–0.50'],
          ['CRITICAL',     IMD_BAND_FILL.CRITICAL,  'p ≥ 0.50'],
          ['UNPRECEDENTED', IMD_UNPRECEDENTED,      '> 99th %ile'],
        ] as const).map(([label, color, range]) => (
          <div key={label} className="flex items-center gap-2 mb-1">
            <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: color }} />
            <span className="font-mono text-[0.55rem]" style={{ color: '#8792A8' }}>
              {label} <span style={{ color: '#E9EDF5' }}>{range}</span>
            </span>
          </div>
        ))}
        <div className="mt-1.5 pt-1.5" style={{ borderTop: '1px solid #1E2D42' }}>
          <div className="font-sans text-[0.55rem] uppercase tracking-wider mb-1" style={{ color: '#8792A8' }}>
            Bust Direction (texture)
          </div>
          <div className="flex items-center gap-2">
            <svg width="14" height="14">
              <rect width="14" height="14" fill={IMD_BAND_FILL.LOW} />
              <line x1="0" y1="14" x2="14" y2="0" stroke="rgba(255,255,255,0.5)" strokeWidth="1.5" />
            </svg>
            <span className="font-mono text-[0.55rem]" style={{ color: '#E9EDF5' }}>WET /</span>
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <svg width="14" height="14">
              <rect width="14" height="14" fill={IMD_BAND_FILL.LOW} />
              <line x1="0" y1="0" x2="14" y2="14" stroke="rgba(255,255,255,0.5)" strokeWidth="1.5" />
            </svg>
            <span className="font-mono text-[0.55rem]" style={{ color: '#E9EDF5' }}>DRY \</span>
          </div>
        </div>
      </div>

      {/* ── Main SVG ─────────────────────────────────────────────────────── */}
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      >
        <defs>
          {/* WET bust hatch: forward-diagonal / */}
          <pattern id={WET_HATCH_ID} width="8" height="8" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(255,255,255,0.40)" strokeWidth="2" />
          </pattern>

          {/* DRY bust hatch: backward-diagonal \ */}
          <pattern id={DRY_HATCH_ID} width="8" height="8" patternTransform="rotate(-45 0 0)" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(255,255,255,0.40)" strokeWidth="2" />
          </pattern>

          {/* OOD unprecedented hatch: dense violet static texture */}
          <pattern id={OOD_HATCH_ID} width="6" height="6" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" />
          </pattern>

          {/* Selected subdivision glow */}
          <filter id={GLOW_FILTER_ID} x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor={ACCENT} floodOpacity="0.55" />
          </filter>
        </defs>

        {/* Ocean background */}
        <rect width="100%" height="100%" fill="#0A1628" />

        <g ref={gRef}>
          {/* ── Subdivision polygons ────────────────────────────────────── */}
          {geojsonFeatures.map((feat: any, idx: number) => {
            const regId    = String(feat.properties.region_id);
            const fc       = forecastMap.get(regId);
            const isSelected = regId === selectedRegionId;
            const pathD    = pathGen(feat) || '';
            const { base, pattern } = getFill(feat);
            const isOOD    = fc?.unprecedented_flag ?? false;

            return (
              <g key={regId}>
                {/* Base fill */}
                <path
                  d={pathD}
                  data-region-id={regId}
                  data-region-name={feat.properties.name}
                  fill={base}
                  fillOpacity={fc ? 0.88 : 0.25}
                  stroke={isSelected ? ACCENT : '#1E2D42'}
                  strokeWidth={isSelected ? 2.5 : 0.6}
                  strokeLinejoin="round"
                  filter={isSelected ? `url(#${GLOW_FILTER_ID})` : undefined}
                  className={`subdivision-path ${isSelected ? 'subdivision-selected' : ''}`}
                  style={{
                    animation: 'fadeIn 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                    animationDelay: `${(idx % 12) * 45}ms`,
                  }}
                  onClick={() => onSelectRegion(regId)}
                  onMouseEnter={(e) => {
                    const rect = containerRef.current?.getBoundingClientRect();
                    setHoverInfo({
                      regionId: regId,
                      name: feat.properties.name || feat.properties.subdivision_name || `Region ${regId}`,
                      pBust: fc?.p_bust ?? 0,
                      elt:   fc?.effective_lead_time ?? leadDay,
                      band:  fc?.confidence_band ?? 'HIGH',
                      direction: fc?.bust_direction,
                      unprecedented: fc?.unprecedented_flag,
                      x: e.clientX - (rect?.left ?? 0),
                      y: e.clientY - (rect?.top  ?? 0),
                    });
                  }}
                  onMouseMove={(e) => {
                    const rect = containerRef.current?.getBoundingClientRect();
                    setHoverInfo((prev) =>
                      prev ? { ...prev, x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) } : null
                    );
                  }}
                  onMouseLeave={() => setHoverInfo(null)}
                />
                {/* Texture overlay for bust_direction or OOD */}
                {pattern && (
                  <path
                    d={pathD}
                    fill={`url(#${pattern})`}
                    fillOpacity={1}
                    stroke="none"
                    pointerEvents="none"
                    className={isOOD ? 'animate-unprecedented' : undefined}
                  />
                )}
              </g>
            );
          })}

          {/* ── Ocean text labels ──────────────────────────────────────── */}
          {proj && [
            { coords: [67.0, 15.0] as [number,number], text: 'ARABIAN SEA' },
            { coords: [89.5, 14.0] as [number,number], text: 'BAY OF BENGAL' },
          ].map(({ coords, text }) => {
            const pt = proj(coords);
            if (!pt) return null;
            return (
              <text
                key={text}
                x={pt[0]}
                y={pt[1]}
                textAnchor="middle"
                fill="#3D536E"
                fontSize="10"
                fontFamily="'IBM Plex Mono', monospace"
                letterSpacing="0.2em"
                fontWeight="600"
                pointerEvents="none"
                className="uppercase select-none"
                opacity="0.6"
              >
                {text}
              </text>
            );
          })}
        </g>

        {/* ── Island Insets (spec Section 8, blunt pushback #2) ─────────── */}
        {islandInsets.map((inset) => {
          const fc    = forecastMap.get(inset.id);
          const feat  = geojsonFeatures.find((f: any) => String(f.properties.region_id) === inset.id);
          if (!feat || !proj) return null;

          const { base, pattern } = fc
            ? (fc.unprecedented_flag
                ? { base: IMD_UNPRECEDENTED, pattern: OOD_HATCH_ID }
                : fc.bust_direction === 'wet'
                  ? { base: IMD_BAND_FILL[fc.confidence_band], pattern: WET_HATCH_ID }
                  : fc.bust_direction === 'dry'
                    ? { base: IMD_BAND_FILL[fc.confidence_band], pattern: DRY_HATCH_ID }
                    : { base: IMD_BAND_FILL[fc.confidence_band], pattern: undefined })
            : { base: '#162238', pattern: undefined };

          // Small dedicated projection for island — fitExtent correctly computes bounds and centering (C7)
          const miniProj = d3.geoMercator().fitExtent(
            [
              [inset.insetX + 8, inset.insetY + 18],
              [inset.insetX + inset.insetW - 8, inset.insetY + inset.insetH - 8],
            ],
            feat
          );
          const miniPath = d3.geoPath().projection(miniProj);
          const pathD = miniPath(feat) || '';
          const isSelected = selectedRegionId === inset.id;
          const islandName = feat.properties.name || feat.properties.SUBDIVISION || inset.label;

          return (
            <g
              key={`inset-${inset.id}`}
              className="cursor-pointer group"
              onClick={() => onSelectRegion(inset.id)}
            >
              {/* Inset border box */}
              <rect
                x={inset.insetX}
                y={inset.insetY}
                width={inset.insetW}
                height={inset.insetH}
                fill="#0A1628"
                stroke={isSelected ? ACCENT : '#1E2D42'}
                strokeWidth={isSelected ? '1.5' : '1'}
                rx="4"
              />
              {/* Island polygon */}
              <path
                d={pathD}
                fill={base}
                fillOpacity={fc ? 0.92 : 0.4}
                stroke={isSelected ? ACCENT : '#1E2D42'}
                strokeWidth={isSelected ? '1.8' : '0.8'}
                filter={isSelected ? `url(#${GLOW_FILTER_ID})` : undefined}
                className={`subdivision-path ${isSelected ? 'subdivision-selected' : ''}`}
                onMouseEnter={(e) => {
                  const rect = containerRef.current?.getBoundingClientRect();
                  setHoverInfo({
                    regionId: inset.id,
                    name: islandName,
                    pBust: fc?.p_bust ?? 0,
                    elt: fc?.effective_lead_time ?? leadDay,
                    band: fc?.confidence_band ?? 'HIGH',
                    direction: fc?.bust_direction,
                    unprecedented: fc?.unprecedented_flag,
                    x: e.clientX - (rect?.left ?? 0),
                    y: e.clientY - (rect?.top ?? 0),
                  });
                }}
                onMouseMove={(e) => {
                  const rect = containerRef.current?.getBoundingClientRect();
                  setHoverInfo((prev) =>
                    prev ? { ...prev, x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) } : null
                  );
                }}
                onMouseLeave={() => setHoverInfo(null)}
              />
              {pattern && (
                <path d={pathD} fill={`url(#${pattern})`} fillOpacity={1} stroke="none" pointerEvents="none" />
              )}
              {/* Inset label */}
              <text
                x={inset.insetX + 6}
                y={inset.insetY + 12}
                fill={isSelected ? '#E9EDF5' : '#8792A8'}
                fontSize="7.5"
                fontFamily="'IBM Plex Mono', monospace"
                fontWeight="600"
              >
                {inset.label}
              </text>
            </g>
          );
        })}
      </svg>

      {/* ── Hover tooltip (100ms delay per spec Section 6.4) ──────────────── */}
      {hoverInfo && (
        <div
          className="absolute z-30 pointer-events-none animate-fade-in"
          style={{ left: hoverInfo.x + 14, top: hoverInfo.y - 10 }}
        >
          <div
            className="rounded p-3 font-sans text-xs min-w-[200px]"
            style={{ background: 'rgba(18,27,46,0.98)', border: '1px solid #1E2D42', backdropFilter: 'blur(12px)' }}
          >
            <div className="font-semibold mb-1.5" style={{ color: '#E9EDF5' }}>
              {hoverInfo.name}
              <span className="font-mono text-[0.55rem] ml-2" style={{ color: '#8792A8' }}>#{hoverInfo.regionId}</span>
            </div>

            {hoverInfo.unprecedented ? (
              <div className="font-sans text-[0.625rem] leading-relaxed" style={{ color: '#7C5CBF' }}>
                ⚡ UNPRECEDENTED — OOD refusal active
              </div>
            ) : (
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span style={{ color: '#8792A8' }}>P(bust)</span>
                  <span className="font-mono font-semibold" style={{ color: IMD_BAND_FILL[hoverInfo.band] }}>
                    {(hoverInfo.pBust * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: '#8792A8' }}>Effective Lead</span>
                  <span className="font-mono" style={{ color: '#E9EDF5' }}>D{hoverInfo.elt.toFixed(1)}</span>
                </div>
                {hoverInfo.direction && hoverInfo.direction !== 'none' && (
                  <div className="flex justify-between">
                    <span style={{ color: '#8792A8' }}>Direction</span>
                    <span className="font-mono uppercase font-semibold" style={{
                      color: hoverInfo.direction === 'wet' ? '#4C8C6D' : '#C99A3A',
                    }}>
                      {hoverInfo.direction === 'wet' ? '↗ WET' : '↘ DRY'}
                    </span>
                  </div>
                )}
              </div>
            )}
            <p className="mt-1.5 pt-1.5 font-sans text-[0.55rem]" style={{ color: '#3D536E', borderTop: '1px solid #1E2D42' }}>
              Click to open regional drawer
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
