import React, { useEffect, useState } from 'react';
import { Clock, ShieldAlert, CheckCircle } from 'lucide-react';

interface ExecutiveThreatBarProps {
  currentLeadDay?: number;
  criticalCount?: number;
}

export const ExecutiveThreatBar: React.FC<ExecutiveThreatBarProps> = ({
  currentLeadDay = 4,
  criticalCount = 0,
}) => {
  const [time, setTime] = useState<Date>(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const pad = (n: number) => String(n).padStart(2, '0');
  const utc = `${pad(time.getUTCHours())}:${pad(time.getUTCMinutes())}:${pad(time.getUTCSeconds())} UTC`;

  const istMs = time.getTime() + 5.5 * 3600 * 1000;
  const istDate = new Date(istMs);
  const ist = `${pad(istDate.getUTCHours())}:${pad(istDate.getUTCMinutes())}:${pad(istDate.getUTCSeconds())} IST`;

  return (
    <div className="w-full bg-slate-950 border-b border-slate-800 px-4 h-7.5 py-1 flex items-center justify-between select-none shrink-0 font-mono text-2xs text-slate-400">
      {/* Left: Synchronized clocks */}
      <div className="flex items-center gap-2.5">
        <Clock className="w-3 h-3 text-slate-400 shrink-0" />
        <span className="text-slate-300 font-semibold">{ist}</span>
        <span className="text-slate-600">/</span>
        <span className="text-slate-400">{utc}</span>
      </div>

      {/* Center: Operational specification */}
      <div className="hidden lg:flex items-center gap-3 text-slate-400">
        <span>DOMAIN: INDIA 36 SUBDIVISIONS</span>
        <span className="text-slate-700">|</span>
        <span>INITIALIZATION: GEFS 3-DAY CYCLE</span>
        <span className="text-slate-700">|</span>
        <span>CLIMATOLOGY ARCHIVE: ERA5 1979–2017</span>
      </div>

      {/* Right: Operational triage status */}
      <div className="flex items-center gap-2">
        <span className="text-slate-400">LEAD: DAY {currentLeadDay}</span>
        <span className="text-slate-700">|</span>
        {criticalCount > 0 ? (
          <span className="flex items-center gap-1 text-red-300 font-semibold bg-red-950/60 px-1.5 py-0.5 rounded border border-red-900/60">
            <ShieldAlert className="w-3 h-3 text-red-400" />
            {criticalCount} HIGH-RISK SUBDIVISIONS FLAGGED
          </span>
        ) : (
          <span className="flex items-center gap-1 text-emerald-300 font-medium">
            <CheckCircle className="w-3 h-3 text-emerald-400" />
            ALL 36 SUBDIVISIONS WITHIN CLIMATOLOGICAL SPREAD
          </span>
        )}
      </div>
    </div>
  );
};
