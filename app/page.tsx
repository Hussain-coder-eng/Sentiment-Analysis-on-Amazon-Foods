'use client';

import { useEffect, useState } from 'react';
import SentimentPlot from '@/components/SentimentPlot';
import type { ReviewScore, DemoApiResponse } from '@/lib/types';

export default function Home() {
  const [reviews, setReviews] = useState<ReviewScore[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Warmup: fire once per browser session, no await (fire-and-forget)
    if (typeof window !== 'undefined' && !sessionStorage.getItem('warmup-done')) {
      fetch('/api/warmup').catch(() => {}); // silence errors — warmup is best-effort
      sessionStorage.setItem('warmup-done', '1');
    }

    // Load demo data
    fetch('/api/demo')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<DemoApiResponse>;
      })
      .then(data => {
        setReviews(data.reviews);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message ?? 'Failed to load demo data');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <p className="text-gray-500 text-lg">Loading demo data...</p>
      </main>
    );
  }

  if (error) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <p className="text-red-500 text-lg">Error: {error}</p>
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
      <SentimentPlot reviews={reviews} />
    </main>
  );
}
