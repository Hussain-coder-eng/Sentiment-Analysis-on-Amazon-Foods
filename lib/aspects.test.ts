import { describe, it, expect } from 'vitest';
import { aggregateAspects, ASPECT_LABELS, type ZeroShotResult } from './aspects';
import type { ReviewScore } from './types';

function review(robertaPos: number, robertaNeg: number): ReviewScore {
  return {
    text: 'x',
    rating: 3,
    vader: { compound: 0, pos: 0, neg: 0, neu: 1 },
    roberta: { positive: robertaPos, negative: robertaNeg, neutral: 1 - robertaPos - robertaNeg },
    disagreement: 0,
  };
}

/** Zero-shot result scoring `hits` labels at 0.9 and everything else at 0.1. */
function zs(...hits: string[]): ZeroShotResult {
  return {
    labels: [...ASPECT_LABELS],
    scores: ASPECT_LABELS.map(l => (hits.includes(l) ? 0.9 : 0.1)),
  };
}

describe('aggregateAspects', () => {
  it('includes aspect mentioned by >= 2 reviews with mean polarity of mentioners', () => {
    const reviews = [review(0.9, 0.1), review(0.5, 0.3), review(0.1, 0.8)];
    const zsResults = [zs('taste & flavor'), zs('taste & flavor'), zs('quality')];
    const out = aggregateAspects(zsResults, reviews);
    expect(out).toHaveLength(1);
    expect(out[0].label).toBe('taste & flavor');
    expect(out[0].mentions).toBe(2);
    expect(out[0].polarity).toBeCloseTo(((0.9 - 0.1) + (0.5 - 0.3)) / 2, 10);
  });

  it('excludes aspects with fewer than 2 mentions', () => {
    const reviews = [review(0.9, 0.1), review(0.8, 0.1)];
    const zsResults = [zs('quality'), zs('ease of use')];
    expect(aggregateAspects(zsResults, reviews)).toHaveLength(0);
  });

  it('scores below the 0.7 threshold do not count as mentions', () => {
    const reviews = [review(0.9, 0.1), review(0.8, 0.1)];
    // zs() scores non-hit labels at 0.1 — all below threshold
    const zsResults = [zs(), zs()];
    expect(aggregateAspects(zsResults, reviews)).toHaveLength(0);
  });

  it('sorts by mentions desc, ties by label asc', () => {
    const reviews = [review(0.9, 0.1), review(0.8, 0.1), review(0.7, 0.1)];
    const zsResults = [
      zs('quality', 'value for money'),
      zs('quality', 'value for money'),
      zs('quality'),
    ];
    const out = aggregateAspects(zsResults, reviews);
    expect(out.map(a => a.label)).toEqual(['quality', 'value for money']);
    expect(out[0].mentions).toBe(3);
    expect(out[1].mentions).toBe(2);
  });

  it('handles unknown labels in zero-shot output by ignoring them', () => {
    const reviews = [review(0.9, 0.1), review(0.8, 0.1)];
    const weird: ZeroShotResult = { labels: ['something else'], scores: [0.99] };
    expect(aggregateAspects([weird, weird], reviews)).toHaveLength(0);
  });

  it('throws on length mismatch between results and reviews', () => {
    const reviews = [review(0.9, 0.1)];
    expect(() => aggregateAspects([], reviews)).toThrow();
  });

  it('returns empty array for empty inputs', () => {
    expect(aggregateAspects([], [])).toEqual([]);
  });
});
