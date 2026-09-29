import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { DistributionRow } from '@/types/domain';

export function DistributionChart({ data, isLoading }: { data?: DistributionRow[]; isLoading: boolean }) {
  if (isLoading || !data) {
    return <div className="h-64 animate-pulse rounded-panel bg-rule/30" />;
  }

  return (
    <section className="rounded-panel border border-rule bg-surface p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-bold uppercase tracking-wider text-ink/70">
          Sentiment by Channel & Region
        </h2>
        <span className="text-xs font-mono text-ink/50">Top Channels</span>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 10, left: 10, bottom: 0 }}>
            <CartesianGrid stroke="var(--rule)" horizontal={false} />
            <XAxis type="number" stroke="var(--neu)" fontSize={11} fontFamily="monospace" />
            <YAxis
              type="category"
              dataKey="label"
              stroke="var(--ink)"
              fontSize={11}
              width={100}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--surface)',
                borderColor: 'var(--rule)',
                borderRadius: '8px',
                fontSize: '12px',
              }}
            />
            <Bar dataKey="positive" stackId="a" fill="var(--pos)" radius={[0, 0, 0, 0]} />
            <Bar dataKey="neutral" stackId="a" fill="var(--neu)" radius={[0, 0, 0, 0]} />
            <Bar dataKey="negative" stackId="a" fill="var(--neg)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
