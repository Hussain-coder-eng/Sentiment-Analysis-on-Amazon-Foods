import { describe, it, expect } from 'vitest';
import { computeVerdict } from './verdict';
import type { ReviewScore } from './types';

/** Build a minimal ReviewScore; only fields verdict math reads are meaningful. */
function review(robertaPos: number, robertaNeg: number, vaderCompound: number): ReviewScore {
  return {
    text: 'x',
    rating: 3,
    vader: { compound: vaderCompound, pos: 0, neg: 0, neu: 1 },
    roberta: { positive: robertaPos, negative: robertaNeg, neutral: 1 - robertaPos - robertaNeg },
    disagreement: 0,
  };
}

describe('computeVerdict', () => {
  it('throws on empty reviews', () => {
    expect(() => computeVerdict([])).toThrow();
  });

  it('all strongly positive → high score, "Loved it"', () => {
    const v = computeVerdict([review(0.9, 0.05, 0.8), review(0.95, 0.02, 0.9)]);
    expect(v.score).toBeGreaterThanOrEqual(75);
    expect(v.label).toBe('Loved it');
    expect(v.positiveCount).toBe(2);
    expect(v.total).toBe(2);
  });

  it('all strongly negative → low score, "Mostly negative"', () => {
    const v = computeVerdict([review(0.05, 0.9, -0.7), review(0.02, 0.95, -0.8)]);
    expect(v.score).toBeLessThan(45);
    expect(v.label).toBe('Mostly negative');
    expect(v.positiveCount).toBe(0);
  });

  it('neutral polarity → score 50, "Mixed"', () => {
    const v = computeVerdict([review(0.3, 0.3, 0.0)]);
    expect(v.score).toBe(50);
    expect(v.label).toBe('Mixed');
  });

  it('band boundaries: 75 → Loved it, 60 → Mostly positive, 45 → Mixed', () => {
    expect(computeVerdict([review(0.6, 0.1, 0.5)]).score).toBe(75);
    expect(computeVerdict([review(0.6, 0.1, 0.5)]).label).toBe('Loved it');
    expect(computeVerdict([review(0.4, 0.2, 0.3)]).label).toBe('Mostly positive');
    expect(computeVerdict([review(0.2, 0.3, -0.1)]).label).toBe('Mixed');
  });

  it('agreement: same-sign reviews agree, opposite-sign disagree', () => {
    const v = computeVerdict([
      review(0.8, 0.1, 0.7),
      review(0.1, 0.8, 0.5),
    ]);
    expect(v.agreementPct).toBe(50);
  });

  it('one-liner format', () => {
    const v = computeVerdict([review(0.8, 0.1, 0.7), review(0.1, 0.8, -0.5)]);
    expect(v.oneLiner).toBe('1 of 2 reviews positive · models agree 100%');
  });
});
