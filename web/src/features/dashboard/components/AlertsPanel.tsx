import { AlertTriangle, AlertCircle, Info, ArrowRight } from 'lucide-react';
import type { Alert } from '@/types/domain';
import { useFilters } from '@/stores/filters';

export function AlertsPanel({ alerts = [], isLoading }: { alerts?: Alert[]; isLoading: boolean }) {
  const { toggleMinistry } = useFilters();

  if (isLoading) {
    return <div className="h-44 animate-pulse rounded-panel bg-rule/30" />;
  }

  if (alerts.length === 0) {
    return (
      <div className="rounded-panel border border-pos/30 bg-pos/5 p-4 flex items-center gap-3">
        <Info className="size-5 text-pos" />
        <div>
          <h4 className="text-sm font-semibold text-pos">No Active Negative Surges</h4>
          <p className="text-xs text-ink/70">All central ministries coverage is within normal statistical baselines.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-panel border border-neg/30 bg-neg/5 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neg opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-neg"></span>
          </span>
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-neg">
            Active Attention Alerts ({alerts.length})
          </h3>
        </div>
        <span className="text-[11px] font-mono text-ink/60">Statistical Outliers (z &gt; 1.5σ)</span>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {alerts.map((al) => {
          const isCritical = al.severity === 'critical';
          const Icon = isCritical ? AlertTriangle : AlertCircle;

          return (
            <div
              key={al.id}
              className={`p-3 rounded-control border bg-surface flex flex-col justify-between transition-all hover:shadow-md ${
                isCritical ? 'border-neg/40 shadow-sm' : 'border-warn/40'
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-1 mb-1">
                  <span className="text-xs font-semibold text-ink truncate font-serif">{al.ministry}</span>
                  <span
                    className={`inline-flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded font-bold uppercase ${
                      isCritical ? 'bg-neg/10 text-neg' : 'bg-warn/10 text-warn'
                    }`}
                  >
                    <Icon className="size-3" />
                    {al.zScore}σ
                  </span>
                </div>
                <p className="text-xs text-ink/70 leading-snug line-clamp-2">{al.title}</p>
              </div>

              <div className="mt-3 pt-2 border-t border-rule/40 flex items-center justify-between text-[11px]">
                <span className="font-mono text-ink/60">
                  {al.negativeCount24h} neg in 24h
                </span>
                <button
                  onClick={() => toggleMinistry(al.ministry)}
                  className="inline-flex items-center gap-1 text-chakra font-medium hover:underline text-[11px]"
                >
                  Filter Ministry <ArrowRight className="size-3" />
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
