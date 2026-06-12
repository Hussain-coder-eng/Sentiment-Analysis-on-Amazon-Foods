import type { AspectScore } from '@/lib/types';

interface Props {
  aspects: AspectScore[];
}

export default function AspectBars({ aspects }: Props) {
  return (
    <section aria-label="What reviewers say" className="mt-8">
      <h2 className="text-slate-400 text-xs font-mono tracking-widest uppercase mb-4">
        What reviewers say
      </h2>
      <div className="rounded-2xl border border-slate-700/60 bg-slate-800/60 p-5 backdrop-blur-sm">
        <ul className="space-y-3">
          {aspects.map(aspect => {
            const positive = aspect.polarity >= 0;
            const width = Math.min(100, Math.round(Math.abs(aspect.polarity) * 100));
            return (
              <li key={aspect.label} className="flex items-center gap-3">
                <span className="w-36 shrink-0 truncate text-xs font-mono text-slate-300">
                  {aspect.label}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-700">
                  <div
                    className={positive ? 'h-full bg-green-500' : 'h-full bg-red-400'}
                    style={{ width: `${width}%` }}
                  />
                </div>
                <span
                  className={`w-12 shrink-0 text-right text-xs font-mono ${
                    positive ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {aspect.polarity >= 0 ? '+' : '−'}
                  {Math.abs(aspect.polarity).toFixed(2)}
                </span>
                <span className="w-8 shrink-0 text-right text-[10px] font-mono text-slate-500">
                  {aspect.mentions}×
                </span>
              </li>
            );
          })}
        </ul>
        <p className="mt-4 text-[10px] font-mono text-slate-600">
          zero-shot NLI · facebook/bart-large-mnli · review-level attribution
        </p>
      </div>
    </section>
  );
}
