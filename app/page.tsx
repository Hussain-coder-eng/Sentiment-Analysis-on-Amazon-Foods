'use client';

import { useEffect, useRef, useState } from 'react';
import DisagreementPanel from '@/components/DisagreementPanel';
import SentimentPlot from '@/components/SentimentPlot';
import type { ReviewScore, AnalyzeApiResponse } from '@/lib/types';

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 30_000;

export default function Home() {
  const [asinInput, setAsinInput] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [retryCountdown, setRetryCountdown] = useState<number | null>(null);
  const [reviews, setReviews] = useState<ReviewScore[] | null>(null);
  const [resultAsin, setResultAsin] = useState<string | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const heroRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const blob1Ref = useRef<HTMLDivElement>(null);
  const blob2Ref = useRef<HTMLDivElement>(null);

  // Warmup on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && !sessionStorage.getItem('warmup-done')) {
      sessionStorage.setItem('warmup-done', '1');
      fetch('/api/warmup').catch(() => {});
    }
  }, []);

  // Hero entrance + ambient blobs
  useEffect(() => {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReduced) return;

    import('animejs').then(({ default: anime }) => {
      // Hero text entrance
      if (heroRef.current) {
        anime({
          targets: heroRef.current.querySelectorAll('.animate-in'),
          translateY: [30, 0],
          opacity: [0, 1],
          duration: 600,
          delay: anime.stagger(120, { start: 100 }),
          easing: 'easeOutExpo',
        });
      }

      // Form entrance
      if (formRef.current) {
        anime({
          targets: formRef.current,
          translateY: [20, 0],
          opacity: [0, 1],
          duration: 500,
          delay: 500,
          easing: 'easeOutExpo',
        });
      }

      // Ambient blob oscillation
      if (blob1Ref.current) {
        anime({
          targets: blob1Ref.current,
          translateX: ['-10px', '10px'],
          translateY: ['-15px', '15px'],
          duration: 8000,
          direction: 'alternate',
          loop: true,
          easing: 'easeInOutSine',
        });
      }
      if (blob2Ref.current) {
        anime({
          targets: blob2Ref.current,
          translateX: ['10px', '-10px'],
          translateY: ['10px', '-20px'],
          duration: 10000,
          direction: 'alternate',
          loop: true,
          easing: 'easeInOutSine',
        });
      }
    });
  }, []);

  // Results entrance animation
  useEffect(() => {
    if (!reviews) return;
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReduced || !resultsRef.current) return;

    import('animejs').then(({ default: anime }) => {
      anime({
        targets: resultsRef.current,
        translateY: [24, 0],
        opacity: [0, 1],
        duration: 500,
        easing: 'easeOutExpo',
      });
    });
  }, [reviews, analyzing]);

  function startCountdown(seconds: number, onDone: () => void) {
    setRetryCountdown(seconds);
    let remaining = seconds;
    countdownRef.current = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(countdownRef.current!);
        setRetryCountdown(null);
        onDone();
      } else {
        setRetryCountdown(remaining);
      }
    }, 1000);
  }

  async function doAnalyze(asin: string, attempt: number): Promise<void> {
    const res = await fetch(`/api/analyze?asin=${asin}`);
    const data = await res.json() as { error?: string } | AnalyzeApiResponse;

    if (res.status === 202 && attempt < MAX_RETRIES) {
      await new Promise<void>(resolve =>
        startCountdown(Math.round(RETRY_DELAY_MS / 1000), resolve)
      );
      return doAnalyze(asin, attempt + 1);
    }

    if (res.status === 202) {
      setAnalyzeError('Analysis still in progress — please try again in a moment.');
      return;
    }
    if (!res.ok) {
      setAnalyzeError((data as { error?: string }).error ?? `Request failed (${res.status})`);
    } else {
      const response = data as AnalyzeApiResponse;
      setReviews(response.reviews);
      setResultAsin(response.asin);
    }
  }

  async function handleAnalyze(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = asinInput.trim().toUpperCase();
    if (!/^[A-Z0-9]{10}$/.test(trimmed)) {
      setAnalyzeError('Invalid ASIN — must be 10 uppercase letters/digits (e.g. B000E7L2R4).');
      return;
    }
    if (countdownRef.current) clearInterval(countdownRef.current);
    setRetryCountdown(null);
    setAnalyzeError(null);
    setReviews(null);
    setResultAsin(null);
    setAnalyzing(true);
    try {
      await doAnalyze(trimmed, 1);
    } catch {
      setAnalyzeError('Network error — please try again.');
    } finally {
      setAnalyzing(false);
      setRetryCountdown(null);
      if (countdownRef.current) clearInterval(countdownRef.current);
    }
  }

  function handleClear() {
    if (countdownRef.current) clearInterval(countdownRef.current);
    setRetryCountdown(null);
    setReviews(null);
    setResultAsin(null);
    setAnalyzeError(null);
  }

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#0F172A]">
      {/* Ambient light blobs */}
      <div
        ref={blob1Ref}
        className="ambient-blob w-96 h-96 bg-green-500 top-[-100px] left-[-80px]"
        aria-hidden="true"
      />
      <div
        ref={blob2Ref}
        className="ambient-blob w-80 h-80 bg-blue-600 bottom-[-60px] right-[-60px]"
        aria-hidden="true"
      />

      <div className="relative z-10 max-w-2xl mx-auto px-6 pt-24 pb-16">
        {/* Hero */}
        <div ref={heroRef} className="mb-12">
          <div className="animate-in opacity-0">
            <span className="inline-block text-green-400 font-mono text-xs tracking-widest uppercase mb-4 border border-green-500/30 rounded-full px-3 py-1 bg-green-500/10">
              ML Sentiment Analysis
            </span>
          </div>
          <h1 className="animate-in opacity-0 font-heading text-4xl sm:text-5xl font-bold text-white leading-tight mb-4 tracking-tight">
            Amazon Review
            <span className="block text-green-400 glow-green-text">
              Sentiment Analyzer
            </span>
          </h1>
          <p className="animate-in opacity-0 text-slate-400 text-base leading-relaxed max-w-lg">
            Enter any Amazon ASIN to score real customer reviews with{' '}
            <span className="text-slate-300 font-medium">VADER</span> and{' '}
            <span className="text-slate-300 font-medium">RoBERTa</span> sentiment models.
          </p>
        </div>

        {/* ASIN form */}
        <form
          ref={formRef}
          onSubmit={handleAnalyze}
          className="opacity-0 mb-6"
          aria-label="Analyze Amazon product"
        >
          <label
            htmlFor="asin-input"
            className="block text-slate-400 text-xs font-mono tracking-widest uppercase mb-2"
          >
            Amazon ASIN
          </label>
          <div className="flex gap-3">
            <input
              id="asin-input"
              type="text"
              value={asinInput}
              onChange={e => {
                  const val = e.target.value;
                  const match = val.match(/(?:\/dp\/|\/product\/)([A-Z0-9]{10})/i);
                  setAsinInput(match ? match[1].toUpperCase() : val);
                }}
              placeholder="e.g. B000E7L2R4"
              disabled={analyzing}
              autoComplete="off"
              spellCheck={false}
              className={[
                'flex-1 bg-slate-800 border border-slate-700 rounded-lg',
                'px-4 py-3 text-sm font-mono text-slate-100 placeholder-slate-500',
                'focus:outline-none focus:ring-2 focus:ring-green-500/60 focus:border-green-500/60',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'transition-all duration-200',
              ].join(' ')}
            />
            <button
              type="submit"
              disabled={analyzing || asinInput.trim().length === 0}
              className={[
                'px-6 py-3 rounded-lg text-sm font-semibold font-heading',
                'bg-green-500 text-slate-950 hover:bg-green-400',
                'disabled:opacity-40 disabled:cursor-not-allowed',
                'transition-all duration-150 cursor-pointer',
                'focus:outline-none focus:ring-2 focus:ring-green-500/60 focus:ring-offset-2 focus:ring-offset-slate-900',
                !analyzing && asinInput.trim().length > 0 ? 'glow-green' : '',
              ].join(' ')}
            >
              {analyzing ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Analyzing
                </span>
              ) : 'Analyze'}
            </button>
            {reviews !== null && !analyzing && (
              <button
                type="button"
                onClick={handleClear}
                className="px-4 py-3 rounded-lg text-sm font-mono text-slate-400 border border-slate-700 hover:border-slate-500 hover:text-slate-200 transition-all duration-150 cursor-pointer focus:outline-none focus:ring-2 focus:ring-slate-500/60"
              >
                Clear
              </button>
            )}
          </div>

          {/* Status messages */}
          {analyzing && (
            <p className="mt-3 text-slate-400 text-sm font-mono">
              {retryCountdown !== null
                ? `⟳ Another analysis in progress — retrying in ${retryCountdown}s…`
                : '⟳ Waking up the ML model — this takes ~10–30s on first use.'}
            </p>
          )}
          {analyzeError && (
            <p role="alert" className="mt-3 text-red-400 text-sm font-mono">
              {analyzeError}
            </p>
          )}
        </form>

        {/* Results */}
        {reviews !== null && !analyzing && (
          <div ref={resultsRef} className="results-panel opacity-0">
            <div className="mb-6 px-4 py-3 rounded-lg bg-slate-800/60 border border-slate-700/60 backdrop-blur-sm">
              <p className="text-slate-400 text-xs font-mono tracking-wide">
                <span className="text-green-400">✓</span>{' '}
                {reviews.length} reviews scored for ASIN{' '}
                <span className="text-slate-200 font-medium">{resultAsin}</span>
              </p>
            </div>
            <SentimentPlot reviews={reviews} />
            <DisagreementPanel reviews={reviews} />
          </div>
        )}
      </div>
    </main>
  );
}
