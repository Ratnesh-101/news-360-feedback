import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, Building2, TrendingUp, TrendingDown, ShieldAlert, Award } from 'lucide-react';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { api } from '@/lib/api/endpoints';
import { SentimentBadge } from '@/components/SentimentBadge';
import type { MinistryStat, Sentiment } from '@/types/domain';

const clamp = (n: number, lo = 0, hi = 100) => Math.min(hi, Math.max(lo, n));

function radarData(m: MinistryStat) {
  return [
    { axis: 'Sentiment', v: clamp((m.sentimentIndex + 100) / 2) },
    { axis: 'Reach', v: clamp(Math.min(100, m.mentions * 2.5)) },
    { axis: 'Calm', v: clamp(100 - m.negativeUrgency) },
    { axis: 'Momentum', v: clamp(50 + m.momentum * 4) },
    { axis: 'Lang Reach', v: clamp(m.languageCoverage * 100) },
  ];
}

function KeywordCloud({ items = [] }: { items?: MinistryStat['topKeywords'] }) {
  if (!items.length) {
    return <p className="text-xs text-ink/50 p-4">No keywords extracted yet.</p>;
  }

  const max = Math.max(...items.map((k) => k.weight), 1);
  const colorMap: Record<Sentiment, string> = {
    positive: 'var(--pos)',
    negative: 'var(--neg)',
    neutral: 'var(--neu)',
  };

  return (
    <ul className="flex flex-wrap items-baseline gap-x-3 gap-y-2 p-4">
      {items.map((k) => (
        <li
          key={k.term}
          style={{
            fontSize: `${11 + (k.weight / max) * 16}px`,
            color: colorMap[k.sentiment] || 'var(--neu)',
          }}
          className="font-medium leading-none tracking-tight hover:underline cursor-default"
        >
          {k.term}
        </li>
      ))}
    </ul>
  );
}

export default function MinistryDetailPage() {
  const { slug = '' } = useParams();

  const { data: m, isLoading, isError } = useQuery({
    queryKey: ['ministry', slug],
    queryFn: () => api.ministry(slug),
    enabled: !!slug,
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-48 animate-pulse rounded bg-rule/30" />
        <div className="h-96 animate-pulse rounded-panel bg-rule/20" />
      </div>
    );
  }

  if (isError || !m) {
    return (
      <div className="p-12 text-center space-y-4">
        <p className="text-neg font-semibold">Ministry details could not be retrieved.</p>
        <Link to="/ministries" className="text-chakra hover:underline text-sm inline-flex items-center gap-1">
          <ArrowLeft className="size-4" /> Back to Ministry Radar
        </Link>
      </div>
    );
  }

  const net = m.sentimentIndex;
  const sentimentType = net > 5 ? 'positive' : net < -5 ? 'negative' : 'neutral';

  return (
    <div className="space-y-6">
      <nav className="flex items-center gap-2 text-xs font-mono text-ink/60">
        <Link to="/ministries" className="hover:text-chakra flex items-center gap-1">
          <ArrowLeft className="size-3.5" /> All Ministries
        </Link>
        <span>/</span>
        <span className="text-ink font-semibold">{m.name}</span>
      </nav>

      {/* Header Profile */}
      <header className="rounded-panel border border-rule bg-surface p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Building2 className="size-6 text-chakra" />
            <h1 className="text-2xl font-serif font-bold text-ink">{m.name}</h1>
          </div>
          <p className="text-xs text-ink/60">
            {m.mentions} total mentions tracked across Hindi, Marathi, and English sources · Negative Urgency:{' '}
            <strong className="text-ink font-mono">{m.negativeUrgency}/100</strong>
          </p>
        </div>

        <div className="flex items-center gap-3">
          <SentimentBadge value={sentimentType} score={Math.abs(net) / 100} />
          <div className="text-right font-mono">
            <div className={`text-lg font-bold ${net >= 0 ? 'text-pos' : 'text-neg'}`}>
              {net >= 0 ? '+' : ''}
              {net.toFixed(1)}
            </div>
            <div className="text-[10px] text-ink/50 uppercase">Net Sentiment Index</div>
          </div>
        </div>
      </header>

      {/* Grid: 5-Axis Radar + Trend */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Radar Profile */}
        <section className="rounded-panel border border-rule bg-surface p-5 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-ink/70">
              5-Axis Reputation Radar
            </h2>
            <Award className="size-4 text-chakra" />
          </div>
          <p className="text-[11px] text-ink/50 mb-4">
            Balanced composite index: Sentiment, Volume, Calm, Momentum, and Multilingual Reach
          </p>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData(m)} outerRadius="75%">
                <PolarGrid stroke="var(--rule)" />
                <PolarAngleAxis dataKey="axis" tick={{ fontSize: 10, fill: 'var(--ink)' }} />
                <Radar dataKey="v" stroke="var(--chakra)" fill="var(--chakra)" fillOpacity={0.35} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Historical Net Sentiment Trend */}
        <section className="rounded-panel border border-rule bg-surface p-5 shadow-sm lg:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-ink/70">
              Historical Net Sentiment Trajectory
            </h2>
            <span className="text-[11px] font-mono text-ink/50">Daily Net Index (-100 to +100)</span>
          </div>
          <p className="text-[11px] text-ink/50 mb-4">
            Longitudinal reception curve reflecting government policy rollouts and public reactions
          </p>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={m.trend || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                <YAxis domain={[-100, 100]} stroke="var(--neu)" fontSize={11} fontFamily="monospace" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--surface)',
                    borderColor: 'var(--rule)',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Area type="monotone" dataKey="net" stroke="var(--chakra)" fill="var(--chakra)" fillOpacity={0.25} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* Root-cause Keywords & Schemes */}
      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-panel border border-rule bg-surface shadow-sm overflow-hidden">
          <div className="border-b border-rule p-4 bg-paper/40 flex items-center justify-between">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-ink/70">
              Root-Cause Keywords
            </h2>
            <span className="text-[10px] font-mono text-ink/50">Colored by Sentiment</span>
          </div>
          <KeywordCloud items={m.topKeywords} />
        </section>

        <section className="rounded-panel border border-rule bg-surface shadow-sm overflow-hidden">
          <div className="border-b border-rule p-4 bg-paper/40 flex items-center justify-between">
            <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-ink/70">
              Welfare Schemes & Programs Monitored
            </h2>
            <ShieldAlert className="size-4 text-chakra" />
          </div>

          <div className="p-4">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-rule text-ink/50 font-mono text-[11px]">
                  <th className="py-2 text-left">Scheme / Program</th>
                  <th className="py-2 text-right">Coverage</th>
                  <th className="py-2 text-right">Sentiment Index</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-rule/50">
                {m.schemes?.map((s) => (
                  <tr key={s.name} className="hover:bg-paper/40">
                    <td className="py-2.5 font-medium text-ink">{s.name}</td>
                    <td className="py-2.5 text-right font-mono">{s.mentions} articles</td>
                    <td
                      className={`py-2.5 text-right font-mono font-bold ${
                        s.sentimentIndex >= 0 ? 'text-pos' : 'text-neg'
                      }`}
                    >
                      {s.sentimentIndex >= 0 ? '+' : ''}
                      {s.sentimentIndex.toFixed(0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
