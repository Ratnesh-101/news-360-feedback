export function ScoreMeter({ score, sentiment }: { score: number; sentiment: string }) {
  const pct = Math.round(Math.min(1.0, Math.max(0.0, score)) * 100);
  const color =
    sentiment === 'positive'
      ? 'var(--pos)'
      : sentiment === 'negative'
      ? 'var(--neg)'
      : 'var(--neu)';

  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 bg-rule/50 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono tabular-nums text-ink/70">
        {score.toFixed(2)}
      </span>
    </div>
  );
}
