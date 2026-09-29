import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Search, ChevronRight, TrendingUp, TrendingDown, ArrowUpDown } from 'lucide-react';
import { api } from '@/lib/api/endpoints';
import { SentimentBadge } from '@/components/SentimentBadge';

export default function MinistryLeaderboard() {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'urgency' | 'mentions' | 'sentiment'>('urgency');

  const { data: ministries = [], isLoading } = useQuery({
    queryKey: ['ministries'],
    queryFn: () => api.ministries(),
  });

  const filtered = ministries.filter((m) =>
    m.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  filtered.sort((a, b) => {
    if (sortBy === 'urgency') return b.negativeUrgency - a.negativeUrgency;
    if (sortBy === 'mentions') return b.mentions - a.mentions;
    if (sortBy === 'sentiment') return b.sentimentIndex - a.sentimentIndex;
    return 0;
  });

  return (
    <div className="space-y-6">
      <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-serif font-bold text-ink">Ministry Coverage & Reputation Radar</h1>
          <p className="text-xs text-ink/60 mt-1">
            Track governance reception, public grievances, and policy momentum across Central Ministries
          </p>
        </div>

        <div className="flex items-center gap-3 w-full sm:w-auto">
          <div className="relative flex-1 sm:w-64">
            <Search className="absolute left-3 top-2.5 size-4 text-ink/40" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search ministry or scheme..."
              className="w-full rounded-control border border-rule bg-surface pl-9 pr-3 py-1.5 text-xs text-ink outline-none focus:border-chakra focus:ring-1 focus:ring-chakra"
            />
          </div>

          <div className="flex items-center gap-1.5 text-xs font-mono text-ink/70 bg-surface border border-rule px-2 py-1.5 rounded-control">
            <ArrowUpDown className="size-3.5 text-chakra" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-transparent outline-none cursor-pointer"
            >
              <option value="urgency">Sort: Negative Urgency</option>
              <option value="mentions">Sort: Total Mentions</option>
              <option value="sentiment">Sort: Net Sentiment</option>
            </select>
          </div>
        </div>
      </header>

      {/* Leaderboard Table */}
      <div className="rounded-panel border border-rule bg-surface shadow-sm overflow-hidden">
        {isLoading ? (
          <div className="p-8 space-y-4">
            {Array.from({ length: 6 }, (_, i) => (
              <div key={i} className="h-12 animate-pulse rounded bg-rule/30" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="p-12 text-center text-ink/60">
            <p className="text-sm font-medium">No ministries found matching "{searchTerm}"</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-rule bg-paper/60 font-mono text-ink/60 uppercase tracking-wider text-[11px]">
                  <th className="py-3 px-4">Ministry / Department</th>
                  <th className="py-3 px-4 text-right">Mentions</th>
                  <th className="py-3 px-4">Net Sentiment Index</th>
                  <th className="py-3 px-4">Negative Urgency</th>
                  <th className="py-3 px-4">7d Trajectory</th>
                  <th className="py-3 px-4 text-center">Lang Reach</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-rule/60 font-sans">
                {filtered.map((m) => {
                  const net = m.sentimentIndex;
                  const sentimentType = net > 5 ? 'positive' : net < -5 ? 'negative' : 'neutral';
                  const isHighUrgency = m.negativeUrgency >= 50;

                  return (
                    <tr
                      key={m.slug}
                      className="hover:bg-paper/50 transition-colors group"
                    >
                      <td className="py-3.5 px-4">
                        <Link
                          to={`/ministries/${m.slug}`}
                          className="font-serif font-bold text-sm text-ink group-hover:text-chakra transition-colors flex items-center gap-1.5"
                        >
                          {m.name}
                        </Link>
                        <span className="text-[11px] font-mono text-ink/50">
                          {m.counts.positive} pos · {m.counts.negative} neg · {m.counts.neutral} neu
                        </span>
                      </td>

                      <td className="py-3.5 px-4 text-right font-mono font-semibold text-ink text-sm">
                        {m.mentions}
                      </td>

                      <td className="py-3.5 px-4">
                        <div className="flex items-center gap-2">
                          <SentimentBadge value={sentimentType} />
                          <span
                            className={`font-mono font-semibold text-xs ${
                              net > 0 ? 'text-pos' : net < 0 ? 'text-neg' : 'text-neu'
                            }`}
                          >
                            {net > 0 ? '+' : ''}
                            {net.toFixed(1)}
                          </span>
                        </div>
                      </td>

                      <td className="py-3.5 px-4">
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-20 bg-rule/50 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${m.negativeUrgency}%`,
                                backgroundColor: isHighUrgency ? 'var(--neg)' : 'var(--warn)',
                              }}
                            />
                          </div>
                          <span
                            className={`font-mono text-xs font-semibold ${
                              isHighUrgency ? 'text-neg' : 'text-ink/70'
                            }`}
                          >
                            {m.negativeUrgency}/100
                          </span>
                        </div>
                      </td>

                      <td className="py-3.5 px-4 font-mono text-xs">
                        <span
                          className={`inline-flex items-center gap-1 ${
                            m.momentum >= 0 ? 'text-pos' : 'text-neg'
                          }`}
                        >
                          {m.momentum >= 0 ? <TrendingUp className="size-3.5" /> : <TrendingDown className="size-3.5" />}
                          {m.momentum >= 0 ? '+' : ''}
                          {m.momentum.toFixed(1)} pts
                        </span>
                      </td>

                      <td className="py-3.5 px-4 text-center font-mono text-xs text-ink/70">
                        {Math.round(m.languageCoverage * 100)}%
                      </td>

                      <td className="py-3.5 px-4 text-right">
                        <Link
                          to={`/ministries/${m.slug}`}
                          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-control bg-chakra/10 text-chakra hover:bg-chakra hover:text-white font-medium text-xs transition-colors"
                        >
                          Radar <ChevronRight className="size-3.5" />
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
