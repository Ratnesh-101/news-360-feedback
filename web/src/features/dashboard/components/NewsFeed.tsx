import { ExternalLink, Search, Clock, Building2 } from 'lucide-react';
import type { Article } from '@/types/domain';
import { SentimentBadge } from '@/components/SentimentBadge';
import { ScoreMeter } from '@/components/ScoreMeter';
import { LangTag } from '@/components/LangTag';
import { useUiStore } from '@/stores/ui';
import { useFilters } from '@/stores/filters';

export function NewsFeed({
  articles = [],
  total = 0,
  isLoading,
}: {
  articles?: Article[];
  total?: number;
  isLoading: boolean;
}) {
  const { setSelectedArticle } = useUiStore();
  const { q, setQ } = useFilters();

  return (
    <section className="rounded-panel border border-rule bg-surface p-4 shadow-sm space-y-4">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold uppercase tracking-wider text-ink/70">
            Real-Time News Stream
          </h2>
          <p className="text-xs text-ink/50 mt-0.5">
            Showing {articles.length} of {total} indexed articles across regional and national feeds
          </p>
        </div>

        <div className="relative w-full sm:w-72">
          <Search className="absolute left-3 top-2.5 size-4 text-ink/40" />
          <input
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search headline, summary, ministry..."
            className="w-full rounded-control border border-rule bg-paper pl-9 pr-3 py-1.5 text-xs text-ink placeholder:text-ink/40 outline-none focus:border-chakra focus:ring-1 focus:ring-chakra transition-colors"
          />
        </div>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }, (_, i) => (
            <div key={i} className="h-20 animate-pulse rounded-panel bg-rule/30" />
          ))}
        </div>
      ) : articles.length === 0 ? (
        <div className="p-8 text-center text-ink/60 border border-dashed border-rule rounded-panel">
          <p className="text-sm">No articles match your active filter criteria.</p>
          <p className="text-xs mt-1 text-ink/40">Try loosening your search query or selecting other categories.</p>
        </div>
      ) : (
        <div className="divide-y divide-rule/60">
          {articles.map((art) => (
            <article
              key={art.id}
              onClick={() => setSelectedArticle(art)}
              className="py-3 px-2 -mx-2 rounded-control hover:bg-paper/70 cursor-pointer transition-colors group"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-1.5 flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <SentimentBadge value={art.sentiment} score={art.sentimentScore} />
                    <LangTag language={art.language} />
                    <span className="text-[11px] font-mono uppercase bg-rule/20 text-ink/70 px-1.5 py-0.5 rounded">
                      {art.category}
                    </span>
                    {art.ministry && (
                      <span className="inline-flex items-center gap-1 text-xs text-chakra font-medium truncate max-w-[220px]">
                        <Building2 className="size-3" />
                        {art.ministry}
                      </span>
                    )}
                  </div>

                  <h3 className="text-sm font-semibold text-ink group-hover:text-chakra transition-colors line-clamp-2 leading-snug">
                    {art.title}
                  </h3>

                  <p className="text-xs text-ink/70 line-clamp-2 leading-relaxed font-sans">
                    {art.summary}
                  </p>
                </div>

                <div className="flex flex-col items-end gap-2 shrink-0 pt-0.5">
                  <ScoreMeter score={art.sentimentScore} sentiment={art.sentiment} />
                  <span className="flex items-center gap-1 text-[11px] font-mono text-ink/50">
                    <Clock className="size-3" />
                    {art.published ? new Date(art.published).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }) : 'Recent'}
                  </span>
                  {art.link && (
                    <a
                      href={art.link}
                      target="_blank"
                      rel="noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="text-ink/40 hover:text-chakra p-1 transition-colors"
                      title="Open source URL"
                    >
                      <ExternalLink className="size-3.5" />
                    </a>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
