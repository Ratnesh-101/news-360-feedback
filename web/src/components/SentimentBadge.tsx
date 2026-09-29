import { SENTIMENT_META } from '@/lib/sentiment';
import type { Sentiment } from '@/types/domain';

export function SentimentBadge({
  value,
  score,
  className = '',
}: {
  value: Sentiment;
  score?: number;
  className?: string;
}) {
  const meta = SENTIMENT_META[value] || SENTIMENT_META.neutral;
  const { label, color, bgClass, Icon } = meta;

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold tracking-wide ${bgClass} ${className}`}
      style={{ borderColor: color }}
    >
      <Icon className="size-3 stroke-[2.5]" aria-hidden />
      <span>{label}</span>
      {score !== undefined && (
        <span className="tabular-nums opacity-90 font-mono text-[11px] ml-0.5">
          {Math.round(score * 100)}%
        </span>
      )}
    </span>
  );
}
