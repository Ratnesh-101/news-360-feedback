import React from 'react';
import { Newspaper, Scale, AlertTriangle, Globe2 } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import type { OverviewKpis } from '@/types/domain';
import { cn } from '@/lib/cn';

interface KpiCardProps {
  label: string;
  value: string;
  delta?: number;
  deltaUnit?: string;
  tone?: 'default' | 'pos' | 'neg' | 'warn';
  icon: React.ReactNode;
  spark?: { v: number }[];
  hint?: string;
}

const toneClass = {
  default: 'text-ink',
  pos: 'text-pos',
  neg: 'text-neg',
  warn: 'text-warn',
};

function KpiCard({
  label,
  value,
  delta,
  deltaUnit = '%',
  tone = 'default',
  icon,
  spark,
  hint,
}: KpiCardProps) {
  return (
    <section className="rounded-panel border border-rule bg-surface p-4 shadow-sm" aria-label={label}>
      <header className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-ink/60">
        <span>{label}</span>
        <div className="text-chakra p-1.5 rounded-control bg-paper">{icon}</div>
      </header>
      <div className="mt-2 flex items-end justify-between gap-3">
        <p className={cn('text-3xl font-bold tabular-nums font-mono tracking-tight', toneClass[tone])}>
          {value}
        </p>
        {spark && spark.length > 0 && (
          <div className="h-10 w-24" aria-hidden="true">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={spark}>
                <Line
                  type="monotone"
                  dataKey="v"
                  dot={false}
                  strokeWidth={2}
                  stroke="var(--chakra)"
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
      <footer className="mt-2 text-xs text-ink/60 flex items-center justify-between">
        {delta !== undefined && (
          <span className={cn('font-semibold', delta >= 0 ? 'text-pos' : 'text-neg')}>
            {delta >= 0 ? '▲ +' : '▼ '}
            {Math.abs(delta).toFixed(1)}
            {deltaUnit}
          </span>
        )}
        <span className="truncate">{hint}</span>
      </footer>
    </section>
  );
}

export function KpiGrid({ data, isLoading }: { data?: OverviewKpis; isLoading: boolean }) {
  if (isLoading || !data) {
    return (
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="h-28 animate-pulse rounded-panel bg-rule/30" />
        ))}
      </div>
    );
  }

  const netTone = data.netSentiment > 5 ? 'pos' : data.netSentiment < -5 ? 'neg' : 'warn';
  const topLangs = data.languages
    .slice(0, 3)
    .map((l) => `${l.language.toUpperCase()} ${Math.round(l.share * 100)}%`)
    .join(' · ');

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <KpiCard
        label="Coverage Volume"
        value={data.totalArticles.toLocaleString('en-IN')}
        delta={data.totalArticlesDelta}
        icon={<Newspaper className="size-4" />}
        spark={data.sparkline?.map((p) => ({ v: p.total }))}
        hint="vs previous 14d"
      />
      <KpiCard
        label="Net Sentiment Index"
        value={`${data.netSentiment > 0 ? '+' : ''}${data.netSentiment.toFixed(1)}`}
        delta={data.netSentimentDelta}
        deltaUnit=" pts"
        tone={netTone}
        icon={<Scale className="size-4" />}
        spark={data.sparkline?.map((p) => ({ v: p.net }))}
        hint={`${data.counts.positive} pos · ${data.counts.negative} neg`}
      />
      <KpiCard
        label="Critical Alerts"
        value={String(data.criticalAlerts)}
        tone={data.criticalAlerts > 0 ? 'neg' : 'default'}
        icon={<AlertTriangle className="size-4" />}
        hint="negative volume surge"
      />
      <KpiCard
        label="Language Diversity"
        value={`${data.languages.length} Active`}
        icon={<Globe2 className="size-4" />}
        hint={topLangs}
      />
    </div>
  );
}
