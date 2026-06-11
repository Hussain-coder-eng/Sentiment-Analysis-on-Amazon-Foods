import type { Verdict } from '@/lib/verdict';

interface Props {
  verdict: Verdict;
  asin: string;
  productTitle?: string;
}

const RING_RADIUS = 34;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

export default function VerdictCard({ verdict, asin, productTitle }: Props) {
  const isNegative = verdict.label === 'Mostly negative';
  const accent = isNegative ? '#F87171' : '#22C55E';
  const dash = (verdict.score / 100) * RING_CIRCUMFERENCE;

  return (
    <section
      aria-label="Verdict"
      className="rounded-2xl border border-slate-700/60 bg-gradient-to-br from-slate-800 to-slate-800/40 px-6 py-8 text-center backdrop-blur-sm"
    >
      <p className="text-slate-500 text-xs font-mono tracking-wide uppercase truncate">
        {productTitle ?? 'Product'} · {asin}
      </p>

      <div className="relative mx-auto mt-5 h-24 w-24">
        <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90" aria-hidden="true">
          <circle cx="40" cy="40" r={RING_RADIUS} fill="none" stroke="#334155" strokeWidth="6" />
          <circle
            cx="40"
            cy="40"
            r={RING_RADIUS}
            fill="none"
            stroke={accent}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${RING_CIRCUMFERENCE - dash}`}
          />
        </svg>
        <span
          className="absolute inset-0 flex items-center justify-center font-heading text-3xl font-bold"
          style={{ color: accent }}
        >
          {verdict.score}
        </span>
      </div>

      <h2 className="mt-4 font-heading text-2xl font-bold text-white">{verdict.label}</h2>
      <p className="mt-2 text-sm font-mono text-slate-400">{verdict.oneLiner}</p>
    </section>
  );
}
