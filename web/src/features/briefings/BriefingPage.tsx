import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Printer, RefreshCw, AlertTriangle, CheckCircle, FileText, Calendar } from 'lucide-react';
import { api } from '@/lib/api/endpoints';

export default function BriefingPage() {
  const queryClient = useQueryClient();

  const { data: brief, isLoading } = useQuery({
    queryKey: ['briefing', 'latest'],
    queryFn: () => api.latestBriefing(),
  });

  const generateMutation = useMutation({
    mutationFn: () => api.generateBriefing(),
    onSuccess: (data) => {
      queryClient.setQueryData(['briefing', 'latest'], data);
    },
  });

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Action Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 print:hidden">
        <div>
          <h1 className="text-2xl font-serif font-bold text-ink">Daily Executive Briefing</h1>
          <p className="text-xs text-ink/60 mt-1">
            Automated intelligence report synthesizing national and regional sentiment for senior officials
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-control border border-rule hover:bg-surface text-xs font-semibold text-ink transition-colors"
          >
            <RefreshCw className={`size-3.5 ${generateMutation.isPending ? 'animate-spin' : ''}`} />
            <span>Regenerate Brief</span>
          </button>

          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-control bg-navy text-white text-xs font-semibold hover:bg-navy/90 transition-colors shadow-sm"
          >
            <Printer className="size-3.5" />
            <span>Print / Save PDF</span>
          </button>
        </div>
      </div>

      {/* Gazette Document Container */}
      {isLoading || !brief ? (
        <div className="rounded-panel border border-rule bg-surface p-12 space-y-4">
          <div className="h-8 w-64 animate-pulse rounded bg-rule/30" />
          <div className="h-4 w-full animate-pulse rounded bg-rule/20" />
          <div className="h-32 w-full animate-pulse rounded bg-rule/20" />
        </div>
      ) : (
        <article className="rounded-panel border border-rule bg-surface p-8 sm:p-12 shadow-sm space-y-8 font-serif leading-relaxed text-ink">
          {/* Masthead */}
          <div className="border-b-2 border-ink/80 pb-6 text-center space-y-2">
            <span className="text-[11px] font-mono tracking-widest uppercase text-ink/60 font-sans block">
              Government of India · Media Intelligence & Feedback Cell
            </span>
            <h2 className="text-3xl font-bold tracking-tight text-ink font-serif">
              {brief.headline}
            </h2>
            <div className="flex items-center justify-center gap-4 text-xs font-mono text-ink/60 font-sans pt-1">
              <span className="flex items-center gap-1">
                <Calendar className="size-3.5 text-chakra" />
                Date: {brief.date}
              </span>
              <span>·</span>
              <span>Net Sentiment Index: <strong className="text-pos font-bold">+{brief.netSentiment.toFixed(1)} pts</strong></span>
            </div>
          </div>

          {/* Section 1: Grievances & Crisis Warnings */}
          <section className="space-y-4">
            <div className="flex items-center gap-2 border-b border-rule pb-2">
              <AlertTriangle className="size-5 text-neg shrink-0" />
              <h3 className="text-lg font-bold text-ink font-serif">
                Critical Public Grievances & Vulnerabilities
              </h3>
            </div>

            <div className="space-y-3 font-sans text-xs">
              {brief.grievances.map((g, i) => (
                <div key={i} className="p-3.5 rounded-panel border border-neg/30 bg-neg/5 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-neg text-xs font-serif">{g.ministry}</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-neg/10 text-neg font-semibold">
                      {g.severity}
                    </span>
                  </div>
                  <p className="text-ink/80 leading-relaxed font-sans">{g.summary}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Section 2: Milestones & Positive Coverage */}
          <section className="space-y-4">
            <div className="flex items-center gap-2 border-b border-rule pb-2">
              <CheckCircle className="size-5 text-pos shrink-0" />
              <h3 className="text-lg font-bold text-ink font-serif">
                Governance Milestones & Positive Media Resonance
              </h3>
            </div>

            <div className="space-y-3 font-sans text-xs">
              {brief.milestones.map((m, i) => (
                <div key={i} className="p-3.5 rounded-panel border border-pos/30 bg-pos/5 space-y-1">
                  <span className="font-bold text-pos text-xs font-serif">{m.ministry}</span>
                  <p className="text-ink/80 leading-relaxed font-sans">{m.summary}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Signoff */}
          <div className="border-t border-rule pt-6 flex items-center justify-between text-xs font-mono text-ink/50 font-sans">
            <span>Generated: {new Date(brief.generatedAt).toLocaleTimeString('en-IN')}</span>
            <span>SIH1329 Automated Intelligence Engine</span>
          </div>
        </article>
      )}
    </div>
  );
}
