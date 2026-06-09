'use client';

import { useEffect, useState } from 'react';
import DisagreementPanel from '@/components/DisagreementPanel';
import SentimentPlot from '@/components/SentimentPlot';
import type { ReviewScore, DemoApiResponse, AnalyzeApiResponse } from '@/lib/types';

export default function Home() {
  const [reviews, setReviews] = useState<ReviewScore[]>([]);
  const [demoLoading, setDemoLoading] = useState(true);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [asinInput, setAsinInput] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    // Warmup: fire once per browser session, fire-and-forget
    if (typeof window !== 'undefined' && !sessionStorage.getItem('warmup-done')) {
      sessionStorage.setItem('warmup-done', '1');
      fetch('/api/warmup').catch(() => {});
    }

    fetch('/api/demo')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<DemoApiResponse>;
      })
      .then(data => {
        setReviews(data.reviews);
        setDemoLoading(false);
      })
      .catch(err => {
        setDemoError(err.message ?? 'Failed to load demo data');
        setDemoLoading(false);
      });
  }, []);

  async function handleAnalyze(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = asinInput.trim().toUpperCase();
    if (!/^[A-Z0-9]{10}$/.test(trimmed)) {
      setAnalyzeError('Invalid ASIN — must be 10 uppercase letters/digits (e.g. B001E4KFG0).');
      return;
    }
    setAnalyzing(true);
    setAnalyzeError(null);
    try {
      const res = await fetch(`/api/analyze?asin=${trimmed}`);
      const data = await res.json() as { error?: string } | AnalyzeApiResponse;
      if (!res.ok) {
        setAnalyzeError((data as { error?: string }).error ?? `Request failed (${res.status})`);
      } else {
        setReviews((data as AnalyzeApiResponse).reviews);
        setIsLive(true);
      }
    } catch {
      setAnalyzeError('Network error — please try again.');
    } finally {
      setAnalyzing(false);
    }
  }

  function handleResetDemo() {
    setAnalyzeError(null);
    setIsLive(false);
    fetch('/api/demo')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<DemoApiResponse>;
      })
      .then(data => {
        setReviews(data.reviews);
        setDemoLoading(false);
      })
      .catch(err => {
        setDemoError(err.message ?? 'Failed to load demo data');
        setDemoLoading(false);
      });
  }

  if (demoLoading) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <p className="text-gray-500 text-lg">Loading demo data...</p>
      </main>
    );
  }

  if (demoError) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <p className="text-red-500 text-lg">Error: {demoError}</p>
      </main>
    );
  }

  return (
    <main className="min-h-screen p-8">
      <h1 className="text-2xl font-bold mb-2">Amazon Review Sentiment Analyzer</h1>
      <p className="text-gray-600 mb-6">
        Dual-model sentiment analysis: VADER vs RoBERTa on {reviews.length} Amazon food reviews.
        Color indicates model disagreement (blue = agreement, red = high disagreement).
      </p>

      {/* ASIN input form */}
      <form onSubmit={handleAnalyze} className="flex gap-2 mb-4">
        <label htmlFor="asin-input" className="sr-only">Amazon ASIN</label>
        <input
          id="asin-input"
          type="text"
          value={asinInput}
          onChange={e => setAsinInput(e.target.value)}
          placeholder="Enter Amazon ASIN (e.g. B001E4KFG0)"
          disabled={analyzing}
          className="border rounded px-3 py-2 w-72 text-sm disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={analyzing || asinInput.trim().length === 0}
          className="bg-blue-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
        >
          {analyzing ? 'Analyzing…' : 'Analyze'}
        </button>
        {(isLive || analyzeError) && !analyzing && (
          <button
            type="button"
            onClick={handleResetDemo}
            className="border border-gray-300 px-4 py-2 rounded text-sm hover:bg-gray-50"
          >
            Reset to demo
          </button>
        )}
      </form>

      {/* Status messages */}
      {analyzing && (
        <p className="text-gray-500 text-sm mb-4">
          Waking up the ML model — this takes ~10-30s on first use.
        </p>
      )}
      {analyzeError && (
        <p className="text-red-500 text-sm mb-4">{analyzeError}</p>
      )}

      {/* Charts — hidden while analysis is in flight, shown for both demo and live results */}
      {!analyzing && (
        <>
          <SentimentPlot reviews={reviews} />
          <DisagreementPanel reviews={reviews} />
        </>
      )}
    </main>
  );
}
