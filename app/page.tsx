'use client';

import { useEffect, useState, useRef } from 'react';
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

  useEffect(() => {
    // Warmup: fire once per browser session, fire-and-forget
    if (typeof window !== 'undefined' && !sessionStorage.getItem('warmup-done')) {
      sessionStorage.setItem('warmup-done', '1');
      fetch('/api/warmup').catch(() => {});
    }
  }, []);

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
    setAnalyzing(true);
    setAnalyzeError(null);
    setReviews(null);
    setResultAsin(null);
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
    <main className="min-h-screen p-8">
      <h1 className="text-2xl font-bold mb-2">Amazon Review Sentiment Analyzer</h1>
      <p className="text-gray-600 mb-6">
        Enter any Amazon ASIN to analyze real customer reviews with VADER and RoBERTa sentiment models.
      </p>

      {/* ASIN input form */}
      <form onSubmit={handleAnalyze} className="flex gap-2 mb-4">
        <label htmlFor="asin-input" className="sr-only">Amazon ASIN</label>
        <input
          id="asin-input"
          type="text"
          value={asinInput}
          onChange={e => setAsinInput(e.target.value)}
          placeholder="Enter ASIN (e.g. B000E7L2R4)"
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
        {reviews !== null && !analyzing && (
          <button
            type="button"
            onClick={handleClear}
            className="border border-gray-300 px-4 py-2 rounded text-sm hover:bg-gray-50"
          >
            Clear
          </button>
        )}
      </form>

      {/* Status messages */}
      {analyzing && (
        <p className="text-gray-500 text-sm mb-4">
          {retryCountdown !== null
            ? `Another analysis in progress — retrying in ${retryCountdown}s…`
            : 'Waking up the ML model — this takes ~10-30s on first use.'}
        </p>
      )}
      {analyzeError && (
        <p className="text-red-500 text-sm mb-4">{analyzeError}</p>
      )}

      {/* Results — only shown after a successful analyze */}
      {reviews !== null && !analyzing && (
        <>
          <p className="text-gray-500 text-sm mb-4">
            Showing {reviews.length} reviews for ASIN {resultAsin}
          </p>
          <SentimentPlot reviews={reviews} />
          <DisagreementPanel reviews={reviews} />
        </>
      )}
    </main>
  );
}
