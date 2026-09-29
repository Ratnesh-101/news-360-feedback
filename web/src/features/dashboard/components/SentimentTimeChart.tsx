import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { TimeseriesPoint } from '@/types/domain';

export function SentimentTimeChart({ data, isLoading }: { data?: TimeseriesPoint[]; isLoading: boolean }) {
  if (isLoading || !data) {
    return <div className="h-72 animate-pulse rounded-panel bg-rule/30" />;
  }

  return (
    <section className="rounded-panel border border-rule bg-surface p-4 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-ink/70">
            Sentiment Timeline & Net Trajectory
          </h2>
          <p className="text-xs text-ink/50 mt-0.5">
            Daily distribution of positive, neutral, and negative stories with Net Index overlay
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="flex items-center gap-1.5 text-pos">
            <span className="size-2 rounded-full bg-pos" /> Positive
          </span>
          <span className="flex items-center gap-1.5 text-neu">
            <span className="size-2 rounded-full bg-neu" /> Neutral
          </span>
          <span className="flex items-center gap-1.5 text-neg">
            <span className="size-2 rounded-full bg-neg" /> Negative
          </span>
          <span className="flex items-center gap-1.5 text-chakra font-semibold">
            <span className="h-0.5 w-3 bg-chakra" /> Net Index
          </span>
        </div>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="var(--rule)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="t"
              tickFormatter={(t: string) =>
                new Date(t).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })
              }
              stroke="var(--neu)"
              fontSize={11}
              fontFamily="monospace"
            />
            <YAxis yAxisId="c" stroke="var(--neu)" fontSize={11} fontFamily="monospace" />
            <YAxis
              yAxisId="n"
              orientation="right"
              domain={[-100, 100]}
              stroke="var(--chakra)"
              fontSize={11}
              fontFamily="monospace"
              tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--surface)',
                borderColor: 'var(--rule)',
                borderRadius: '8px',
                fontSize: '12px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
              }}
              formatter={(value: any, name: string) => [
                name === 'net' ? `${Number(value).toFixed(1)} pts` : value,
                name === 'net' ? 'Net Index' : name.charAt(0).toUpperCase() + name.slice(1),
              ]}
              labelFormatter={(label) => new Date(label).toLocaleDateString('en-IN', { dateStyle: 'medium' })}
            />
            <Area
              yAxisId="c"
              type="monotone"
              dataKey="negative"
              stackId="counts"
              stroke="var(--neg)"
              fill="var(--neg)"
              fillOpacity={0.35}
            />
            <Area
              yAxisId="c"
              type="monotone"
              dataKey="neutral"
              stackId="counts"
              stroke="var(--neu)"
              fill="var(--neu)"
              fillOpacity={0.25}
            />
            <Area
              yAxisId="c"
              type="monotone"
              dataKey="positive"
              stackId="counts"
              stroke="var(--pos)"
              fill="var(--pos)"
              fillOpacity={0.35}
            />
            <Line
              yAxisId="n"
              type="monotone"
              dataKey="net"
              stroke="var(--chakra)"
              strokeWidth={2.5}
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
