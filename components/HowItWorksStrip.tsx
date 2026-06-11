const STEPS: { title: string; body: string }[] = [
  { title: 'Fetch reviews', body: 'Top customer reviews pulled live from Amazon via the Canopy GraphQL API.' },
  { title: 'Score twice', body: 'Each review scored by VADER (lexicon) and RoBERTa (transformer) independently.' },
  { title: 'Find aspects', body: 'Zero-shot NLI tags what reviewers discuss — taste, value, packaging, more.' },
  { title: 'Synthesize', body: 'Scores fuse into a 0–100 verdict with model-agreement transparency.' },
];

export default function HowItWorksStrip() {
  return (
    <section aria-label="How it works" className="mt-10">
      <h2 className="text-slate-400 text-xs font-mono tracking-widest uppercase mb-4">
        How it works
      </h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {STEPS.map((step, i) => (
          <div
            key={step.title}
            className="rounded-xl border border-slate-700/60 bg-slate-800/60 p-4"
          >
            <span className="font-mono text-xs text-green-400">0{i + 1}</span>
            <h3 className="mt-1 text-sm font-semibold text-slate-100">{step.title}</h3>
            <p className="mt-1 text-xs leading-relaxed text-slate-400">{step.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
