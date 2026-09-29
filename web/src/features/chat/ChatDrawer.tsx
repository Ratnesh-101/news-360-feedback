import React, { useState, useRef, useEffect } from 'react';
import { X, Send, Square, Bot, User, ExternalLink, ChevronDown, Sparkles } from 'lucide-react';
import { useUiStore } from '@/stores/ui';
import { useFilters } from '@/stores/filters';
import { streamPost } from '@/lib/api/sse';
import { SentimentBadge } from '@/components/SentimentBadge';
import type { ChatMessage, ChatSource } from '@/types/domain';

const PROMPTS = [
  'What are the biggest negative stories about Railways this week?',
  'Summarise regional reaction to PM-KISAN in Hindi and Marathi',
  'Which ministries have seen positive coverage on infrastructure?',
  'What are the main concerns reported regarding crude oil prices?',
];

function CitationsList({ sources }: { sources: ChatSource[] }) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <div className="mt-3 space-y-1.5 border-t border-rule/50 pt-2">
      <div className="text-[11px] font-mono uppercase tracking-wider text-ink/50 flex items-center gap-1">
        <span>Verified Sources ({sources.length})</span>
      </div>
      <div className="space-y-1">
        {sources.map((s, i) => {
          const isOpen = openIndex === i;
          return (
            <div key={s.articleId} className="rounded border border-rule bg-paper/60 text-xs">
              <button
                type="button"
                onClick={() => setOpenIndex(isOpen ? null : i)}
                className="w-full flex items-center justify-between p-2 text-left hover:bg-rule/30 transition-colors"
              >
                <div className="flex items-center gap-2 truncate">
                  <span className="font-mono text-[10px] text-chakra font-bold">[{i + 1}]</span>
                  <span className="truncate font-medium text-ink/90">{s.title}</span>
                </div>
                <ChevronDown
                  className={`size-3 text-ink/50 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                />
              </button>

              {isOpen && (
                <div className="p-2.5 pt-0 space-y-2 border-t border-rule/30 text-ink/80 text-[11px] bg-surface">
                  <p className="line-clamp-3 font-sans leading-relaxed">{s.snippet}</p>
                  <div className="flex items-center justify-between pt-1">
                    <div className="flex items-center gap-2">
                      <SentimentBadge value={s.sentiment} />
                      <span className="text-ink/50 font-mono text-[10px]">
                        Relevance: {Math.round(s.score * 100)}%
                      </span>
                    </div>
                    {s.link && (
                      <a
                        href={s.link}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 text-chakra hover:underline text-[11px]"
                      >
                        Read Article <ExternalLink className="size-3" />
                      </a>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ChatDrawer() {
  const { chatOpen, setChatOpen } = useUiStore();
  const getChatFilters = useFilters((s) => s.getChatFilters);
  const chatFilters = getChatFilters();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const patchLastAssistant = (fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => {
      let idx = -1;
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === 'assistant') {
          idx = i;
          break;
        }
      }
      if (idx === -1) return prev;
      const copy = [...prev];
      copy[idx] = fn(copy[idx]);
      return copy;
    });
  };

  const handleSend = async (questionText: string) => {
    const q = questionText.trim();
    if (!q || isStreaming) return;

    setDraft('');
    const now = new Date().toISOString();
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: q,
      status: 'done',
      createdAt: now,
    };
    const assistantMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      status: 'streaming',
      createdAt: now,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const stream = streamPost(
        '/api/v1/chat/stream',
        {
          question: q,
          filters: chatFilters,
          history: messages.slice(-4).map((m) => ({ role: m.role, content: m.content })),
        },
        ctrl.signal
      );

      for await (const event of stream) {
        if (event.type === 'sources') {
          patchLastAssistant((m) => ({ ...m, sources: event.sources }));
        } else if (event.type === 'token') {
          patchLastAssistant((m) => ({ ...m, content: m.content + event.text }));
        } else if (event.type === 'done') {
          patchLastAssistant((m) => ({ ...m, status: 'done' }));
        } else if (event.type === 'error') {
          patchLastAssistant((m) => ({ ...m, status: 'error', content: m.content || event.message }));
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        patchLastAssistant((m) => ({
          ...m,
          status: 'error',
          content: m.content || 'Error connecting to news intelligence pipeline. Please try again.',
        }));
      }
    } finally {
      setIsStreaming(false);
      patchLastAssistant((m) => ({ ...m, status: 'done' }));
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setIsStreaming(false);
    patchLastAssistant((m) => ({ ...m, status: 'done' }));
  };

  if (!chatOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-navy/40 backdrop-blur-sm transition-opacity">
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <aside className="w-screen max-w-xl bg-surface border-l border-rule shadow-2xl flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-rule p-4 bg-paper/60">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-control bg-chakra text-white">
                <Bot className="size-4" />
              </div>
              <div>
                <h2 className="text-sm font-serif font-bold text-ink flex items-center gap-1.5">
                  RAG News Intelligence Analyst
                </h2>
                <p className="text-[11px] font-mono text-ink/50">
                  Direct vector retrieval over all 391 verified news articles
                </p>
              </div>
            </div>
            <button
              onClick={() => setChatOpen(false)}
              className="p-1.5 rounded-control text-ink/60 hover:text-ink hover:bg-rule/40"
              aria-label="Close chat"
            >
              <X className="size-5" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="space-y-4 py-8">
                <div className="text-center space-y-1">
                  <Sparkles className="size-6 text-chakra mx-auto" />
                  <h3 className="text-sm font-semibold text-ink">Ask Anything About News & Policies</h3>
                  <p className="text-xs text-ink/60 max-w-xs mx-auto">
                    Semantic RAG querying synthesizes central policy impacts and regional grievances.
                  </p>
                </div>

                <div className="space-y-2 pt-2">
                  <span className="text-[11px] font-mono uppercase text-ink/50 block">Suggested Inquiries:</span>
                  <div className="space-y-1.5">
                    {PROMPTS.map((p, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSend(p)}
                        className="w-full text-left p-2.5 rounded-control border border-rule/70 hover:border-chakra hover:bg-paper text-xs text-ink transition-colors flex items-center justify-between group"
                      >
                        <span className="font-sans leading-snug">{p}</span>
                        <Send className="size-3 text-ink/30 group-hover:text-chakra transition-colors shrink-0 ml-2" />
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              messages.map((m) => (
                <div
                  key={m.id}
                  className={`flex gap-3 text-xs leading-relaxed ${
                    m.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  {m.role === 'assistant' && (
                    <div className="size-6 rounded-full bg-chakra/10 text-chakra flex items-center justify-center shrink-0 mt-0.5">
                      <Bot className="size-3.5" />
                    </div>
                  )}

                  <div
                    className={`max-w-[85%] rounded-panel p-3.5 ${
                      m.role === 'user'
                        ? 'bg-chakra text-white rounded-br-none shadow-sm'
                        : 'bg-paper/70 border border-rule/70 rounded-bl-none'
                    }`}
                  >
                    <div className="whitespace-pre-wrap font-sans text-xs leading-relaxed">
                      {m.content}
                      {m.status === 'streaming' && (
                        <span className="inline-block size-2 rounded-full bg-chakra animate-ping ml-1.5" />
                      )}
                    </div>

                    {m.sources && m.sources.length > 0 && <CitationsList sources={m.sources} />}
                  </div>

                  {m.role === 'user' && (
                    <div className="size-6 rounded-full bg-ink/10 text-ink flex items-center justify-center shrink-0 mt-0.5">
                      <User className="size-3.5" />
                    </div>
                  )}
                </div>
              ))
            )}
            <div ref={endRef} />
          </div>

          {/* Footer Input */}
          <div className="border-t border-rule p-3 bg-paper/60">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend(draft);
              }}
              className="flex items-center gap-2"
            >
              <input
                type="text"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="Ask about a policy, ministry, or regional response..."
                className="flex-1 rounded-control border border-rule bg-surface px-3 py-2 text-xs text-ink placeholder:text-ink/40 outline-none focus:border-chakra focus:ring-1 focus:ring-chakra transition-colors"
              />

              {isStreaming ? (
                <button
                  type="button"
                  onClick={handleStop}
                  className="p-2 rounded-control bg-neg text-white hover:bg-neg/90 transition-colors"
                  title="Stop generation"
                >
                  <Square className="size-4" />
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!draft.trim()}
                  className="p-2 rounded-control bg-chakra text-white hover:bg-chakra/90 disabled:opacity-40 transition-colors shadow-sm"
                  title="Send query"
                >
                  <Send className="size-4" />
                </button>
              )}
            </form>
          </div>
        </aside>
      </div>
    </div>
  );
}
