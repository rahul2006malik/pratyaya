import React, { useMemo, useState } from 'react';
import { Activity, AlertTriangle, Info, TrendingDown, TrendingUp } from 'lucide-react';
import { ForecastCard } from '../api/types';

interface DivergenceTrajectoryChartProps {
  card: ForecastCard | null;
  currentLeadDay: number;
  onSelectLeadDay?: (lead: number) => void;
}

export const DivergenceTrajectoryChart: React.FC<DivergenceTrajectoryChartProps> = ({
  card,
  currentLeadDay,
  onSelectLeadDay,
}) => {
  const [hoveredLead, setHoveredLead] = useState<number | null>(null);

  // ── Build D1 to D10 Trajectory Dataset ───────────────────────────────────
  const trajectoryData = useMemo(() => {
    const isDryBust = card?.bust_direction === 'dry';
    const isWetBust = card?.bust_direction === 'wet';
    const baseRain = card?.top_analogues?.[0]?.forecast_value ?? 55;
    const trueObs = card?.top_analogues?.[0]?.observed_value ?? (isDryBust ? 14.5 : 88);

    return Array.from({ length: 10 }, (_, i) => {
      const lead = i + 1;
      // NWP Model curve: tends to maintain high rainfall if overpredicting
      let nwp = isDryBust
        ? Math.round((baseRain + Math.sin(lead * 0.4) * 8 + (lead * 1.5)) * 10) / 10
        : Math.round((35 + lead * 2.2) * 10) / 10;

      // Analogue Expectation: branches away sharply after lead 4
      let analogue = isDryBust
        ? Math.round((nwp * Math.max(0.2, 1 - (lead > 3 ? (lead - 3) * 0.18 : 0.05))) * 10) / 10
        : Math.round((nwp + (lead > 3 ? (lead - 3) * 12 : 2)) * 10) / 10;

      // Error Spread Band
      const spread = Math.round((5 + lead * 4.2) * 10) / 10;
      const upperSpread = Math.round((nwp + spread) * 10) / 10;
      const lowerSpread = Math.max(0, Math.round((nwp - spread) * 10) / 10);

      return {
        lead,
        nwp,
        analogue,
        upperSpread,
        lowerSpread,
        spread,
        divergence: Math.abs(Math.round((nwp - analogue) * 10) / 10),
      };
    });
  }, [card]);

  // Chart dimensions
  const width = 560;
  const height = 230;
  const margin = { top: 20, right: 30, bottom: 35, left: 45 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Scales
  const maxVal = Math.max(
    ...trajectoryData.map((d) => Math.max(d.upperSpread, d.analogue, d.nwp)),
    90
  );
  const minVal = 0;

  const xScale = (lead: number) => margin.left + ((lead - 1) / 9) * innerWidth;
  const yScale = (val: number) => margin.top + innerHeight - ((val - minVal) / (maxVal - minVal)) * innerHeight;

  // SVG Area path for error spread
  const areaPath = useMemo(() => {
    let top = '';
    let bottom = '';
    trajectoryData.forEach((d, idx) => {
      const x = xScale(d.lead);
      const yTop = yScale(d.upperSpread);
      const yBottom = yScale(d.lowerSpread);
      if (idx === 0) {
        top += `M ${x} ${yTop}`;
        bottom = `L ${x} ${yBottom}`;
      } else {
        top += ` L ${x} ${yTop}`;
        bottom = ` L ${x} ${yBottom}` + bottom;
      }
    });
    return top + bottom + ' Z';
  }, [trajectoryData, maxVal]);

  // SVG Line paths
  const nwpLinePath = useMemo(() => {
    return trajectoryData.reduce((acc, d, i) => {
      const cmd = i === 0 ? 'M' : 'L';
      return `${acc} ${cmd} ${xScale(d.lead)} ${yScale(d.nwp)}`;
    }, '');
  }, [trajectoryData, maxVal]);

  const analogueLinePath = useMemo(() => {
    return trajectoryData.reduce((acc, d, i) => {
      const cmd = i === 0 ? 'M' : 'L';
      return `${acc} ${cmd} ${xScale(d.lead)} ${yScale(d.analogue)}`;
    }, '');
  }, [trajectoryData, maxVal]);

  const activeData = hoveredLead
    ? trajectoryData.find((d) => d.lead === hoveredLead)
    : trajectoryData.find((d) => d.lead === currentLeadDay);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 text-slate-100 flex flex-col justify-between shadow-md">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between pb-3 mb-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h3 className="font-display font-bold text-sm tracking-tight text-white uppercase">
            The Divergence Trajectory (D+1 to D+10)
          </h3>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 text-3xs font-mono">
          <span className="flex items-center gap-1.5 text-blue-400">
            <span className="w-3 h-0.5 bg-blue-500 inline-block" />
            NWP Ensemble Mean
          </span>
          <span className="flex items-center gap-1.5 text-amber-400">
            <span className="w-3 h-0.5 bg-amber-400 inline-block" />
            Analogue Expectation
          </span>
          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-3 h-2 bg-blue-900/30 border border-blue-500/20 inline-block" />
            Spread
          </span>
        </div>
      </div>

      {/* ── Subtitle Context ────────────────────────────────────────────── */}
      <div className="flex items-center justify-between text-2xs text-slate-400 mb-2 font-mono">
        <span>Precipitation Ensemble Spread vs Synoptic Analogue Evolution</span>
        {activeData && (
          <span className="text-white font-bold bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
            Lead D+{activeData.lead}: NWP {activeData.nwp} mm vs Analogue {activeData.analogue} mm (Δ {activeData.divergence} mm)
          </span>
        )}
      </div>

      {/* ── SVG Chart Canvas ────────────────────────────────────────────── */}
      <div className="relative w-full overflow-hidden bg-slate-950/70 border border-slate-800/80 rounded-lg p-2">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto overflow-visible select-none"
        >
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map((val) => {
            if (val > maxVal) return null;
            const y = yScale(val);
            return (
              <g key={`grid-${val}`}>
                <line
                  x1={margin.left}
                  y1={y}
                  x2={width - margin.right}
                  y2={y}
                  stroke="#1E293B"
                  strokeDasharray="2,2"
                />
                <text
                  x={margin.left - 6}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-slate-500 text-[9px] font-mono"
                >
                  {val}
                </text>
              </g>
            );
          })}

          {/* Divergence Horizon Warning Zone (Days 5 to 10) */}
          <rect
            x={xScale(5)}
            y={margin.top}
            width={xScale(10) - xScale(5)}
            height={innerHeight}
            fill="#EF4444"
            fillOpacity="0.04"
          />
          <text
            x={xScale(7.5)}
            y={margin.top + 14}
            textAnchor="middle"
            className="fill-red-400/60 text-[9px] font-mono uppercase tracking-wider"
          >
            Decoupling / Bust Zone
          </text>

          {/* Uncertainty Spread Shaded Area */}
          <path d={areaPath} fill="#2563EB" fillOpacity="0.12" />

          {/* Analogue Expectation Curve */}
          <path
            d={analogueLinePath}
            fill="none"
            stroke="#F59E0B"
            strokeWidth="2.2"
            strokeLinecap="round"
          />

          {/* NWP Physics Model Curve */}
          <path
            d={nwpLinePath}
            fill="none"
            stroke="#38BDF8"
            strokeWidth="2.2"
            strokeLinecap="round"
          />

          {/* Data Points */}
          {trajectoryData.map((d) => {
            const x = xScale(d.lead);
            const yNwp = yScale(d.nwp);
            const yAna = yScale(d.analogue);
            const isCurrent = d.lead === currentLeadDay;
            const isHovered = d.lead === hoveredLead;

            return (
              <g key={`point-${d.lead}`}>
                {/* Active Lead Vertical Indicator */}
                {isCurrent && (
                  <line
                    x1={x}
                    y1={margin.top}
                    x2={x}
                    y2={margin.top + innerHeight}
                    stroke="#FFFFFF"
                    strokeWidth="1.2"
                    strokeDasharray="3,3"
                    opacity="0.8"
                  />
                )}

                {/* NWP Node */}
                <circle
                  cx={x}
                  cy={yNwp}
                  r={isCurrent ? 4.5 : 3}
                  className="fill-cyan-400 stroke-slate-900"
                  strokeWidth="1.5"
                />

                {/* Analogue Node */}
                <circle
                  cx={x}
                  cy={yAna}
                  r={isCurrent ? 4.5 : 3}
                  className="fill-amber-400 stroke-slate-900"
                  strokeWidth="1.5"
                />

                {/* Clickable Column Sensor */}
                <rect
                  x={x - 18}
                  y={margin.top}
                  width={36}
                  height={innerHeight + 25}
                  fill="transparent"
                  className="cursor-pointer"
                  onMouseEnter={() => setHoveredLead(d.lead)}
                  onMouseLeave={() => setHoveredLead(null)}
                  onClick={() => onSelectLeadDay?.(d.lead)}
                />
              </g>
            );
          })}

          {/* X Axis Labels */}
          {trajectoryData.map((d) => {
            const x = xScale(d.lead);
            const isCurrent = d.lead === currentLeadDay;
            return (
              <text
                key={`xlabel-${d.lead}`}
                x={x}
                y={margin.top + innerHeight + 16}
                textAnchor="middle"
                className={`text-[10px] font-mono cursor-pointer ${
                  isCurrent ? 'fill-cyan-400 font-bold' : 'fill-slate-400 hover:fill-white'
                }`}
                onClick={() => onSelectLeadDay?.(d.lead)}
              >
                D+{d.lead}
              </text>
            );
          })}
        </svg>
      </div>

      {/* ── Chart Key Insight Ticker ─────────────────────────────────────── */}
      <div className="mt-3 flex items-center justify-between text-2xs text-slate-400 font-mono">
        <span className="flex items-center gap-1.5 text-slate-300">
          <Info className="w-3.5 h-3.5 text-cyan-400" />
          Divergence accelerates sharply at Lead D+5 as parameterisation errors accumulate.
        </span>
        <span className="text-3xs text-slate-400">
          Click any D+N to inspect lead
        </span>
      </div>
    </div>
  );
};
