import type { ReviewScore } from './types';

const LOVED_THRESHOLD = 75;
const POSITIVE_THRESHOLD = 60;
const MIXED_THRESHOLD = 45;

export type VerdictLabel = 'Loved it' | 'Mostly positive' | 'Mixed' | 'Mostly negative';

export interface Verdict {
  score: number;        // 0-100
  label: VerdictLabel;
  positiveCount: number;
  total: number;
  agreementPct: number; // 0-100, % of reviews where VADER and RoBERTa share sign
  oneLiner: string;
}

/** RoBERTa polarity for one review: positive minus negative probability, in [-1, 1]. */
function polarity(r: ReviewScore): number {
  return r.roberta.positive - r.roberta.negative;
}

function labelFor(score: number): VerdictLabel {
  if (score >= LOVED_THRESHOLD) return 'Loved it';
  if (score >= POSITIVE_THRESHOLD) return 'Mostly positive';
  if (score >= MIXED_THRESHOLD) return 'Mixed';
  return 'Mostly negative';
}

/** Synthesize a product-level verdict from per-review model scores. Throws on empty input. */
export function computeVerdict(reviews: ReviewScore[]): Verdict {
  if (reviews.length === 0) {
    throw new Error('computeVerdict requires at least one review');
  }

  const polarities = reviews.map(polarity);
  const mean = polarities.reduce((a, b) => a + b, 0) / polarities.length;
  const score = Math.min(100, Math.max(0, Math.round((mean + 1) * 50)));

  const positiveCount = polarities.filter(p => p > 0).length;
  // Models "agree" when both place the review on the same side of neutral.
  const agreeCount = reviews.filter(
    (r, i) => (r.vader.compound >= 0) === (polarities[i] >= 0)
  ).length;
  const agreementPct = Math.round((agreeCount / reviews.length) * 100);

  return {
    score,
    label: labelFor(score),
    positiveCount,
    total: reviews.length,
    agreementPct,
    oneLiner: `${positiveCount} of ${reviews.length} reviews positive · models agree ${agreementPct}%`,
  };
}
