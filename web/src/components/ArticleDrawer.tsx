import { X, ExternalLink, Calendar, Building2, Tag, FileText, CheckCircle2 } from 'lucide-react';
import { useUiStore } from '@/stores/ui';
import { SentimentBadge } from './SentimentBadge';
import { ScoreMeter } from './ScoreMeter';
import { LangTag } from './LangTag';

export function ArticleDrawer() {
  const { selectedArticle, setSelectedArticle } = useUiStore();

  if (!selectedArticle) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-navy/40 backdrop-blur-sm transition-opacity">
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <aside className="w-screen max-w-2xl bg-surface border-l border-rule shadow-2xl flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-rule p-4 bg-paper/60">
            <div className="flex items-center gap-2">
              <SentimentBadge value={selectedArticle.sentiment} score={selectedArticle.sentimentScore} />
              <LangTag language={selectedArticle.language} />
              <span className="text-xs uppercase font-mono text-ink/60 bg-rule/30 px-2 py-0.5 rounded">
                {selectedArticle.category}
              </span>
            </div>
            <button
              onClick={() => setSelectedArticle(null)}
              className="rounded-control p-1.5 text-ink/60 hover:text-ink hover:bg-rule/40 transition-colors"
              aria-label="Close article drawer"
            >
              <X className="size-5" />
            </button>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <div>
              <h2 className="text-xl font-serif font-bold text-ink leading-snug">
                {selectedArticle.title}
              </h2>
              <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-ink/60">
                <span className="flex items-center gap-1.5">
                  <Calendar className="size-3.5 text-chakra" />
                  {selectedArticle.published ? new Date(selectedArticle.published).toLocaleString('en-IN') : 'Recent'}
                </span>
                {selectedArticle.ministry && (
                  <span className="flex items-center gap-1.5 font-medium text-ink/80">
                    <Building2 className="size-3.5 text-chakra" />
                    {selectedArticle.ministry}
                  </span>
                )}
              </div>
            </div>

            {/* Score & Intensity Card */}
            <div className="rounded-panel border border-rule bg-paper/50 p-4 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold text-ink/70">
                <span>Sentiment Confidence & Impact</span>
                <ScoreMeter score={selectedArticle.sentimentScore} sentiment={selectedArticle.sentiment} />
              </div>
              <p className="text-xs text-ink/80 leading-relaxed font-sans">
                <strong className="text-ink font-semibold">Governance Analysis:</strong> {selectedArticle.reason}
              </p>
            </div>

            {/* Summary */}
            <div className="space-y-2">
              <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-ink/60 flex items-center gap-1.5">
                <FileText className="size-3.5" /> Translated Summary
              </h3>
              <p className="text-sm text-ink/90 leading-relaxed font-sans bg-surface border border-rule/50 p-3.5 rounded-panel">
                {selectedArticle.summary}
              </p>
            </div>

            {/* Original Text (for regional or audio news) */}
            {selectedArticle.originalText && (
              <div className="space-y-2">
                <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-ink/60 flex items-center gap-1.5">
                  <FileText className="size-3.5" /> Original Transcript / Regional Text
                </h3>
                <div
                  lang={selectedArticle.language}
                  className="text-sm text-ink/85 leading-relaxed bg-paper p-3.5 rounded-panel border border-rule/50 font-serif"
                >
                  {selectedArticle.originalText}
                </div>
              </div>
            )}

            {/* Keywords */}
            {selectedArticle.keywords && selectedArticle.keywords.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-ink/60 flex items-center gap-1.5">
                  <Tag className="size-3.5" /> Associated Keywords & Entities
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {selectedArticle.keywords.map((kw, idx) => (
                    <span
                      key={idx}
                      className="px-2.5 py-1 rounded-full text-xs bg-chakra/10 text-chakra border border-chakra/20 font-medium"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="border-t border-rule p-4 bg-paper/60 flex items-center justify-between">
            <span className="text-xs text-ink/50 font-mono">
              ID: {selectedArticle.id}
            </span>
            {selectedArticle.link && (
              <a
                href={selectedArticle.link}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-control bg-chakra text-white text-xs font-semibold hover:bg-chakra/90 transition-colors shadow-sm"
              >
                <span>View Source Article</span>
                <ExternalLink className="size-3.5" />
              </a>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
