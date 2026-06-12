import type { ReviewScore, AspectScore } from './types';

export const ASPECT_LABELS = [
  'taste & flavor',
  'quality',
  'value for money',
  'packaging & shipping',
  'ease of use',
] as const;

/** Zero-shot score >= this counts the review as mentioning the aspect. */
export const MENTION_THRESHOLD = 0.7;
/** Aspects mentioned by fewer reviews than this are dropped (too thin to chart). */
export const MIN_MENTIONS = 2;

/** Raw multi-label zero-shot output for one review (labels/scores are parallel arrays). */
export interface ZeroShotResult {
  labels: string[];
  scores: number[];
}

/**
 * Aggregate per-review zero-shot aspect detections into product-level aspect scores.
 * Aspect polarity reuses each mentioning review's RoBERTa polarity (review-level
 * attribution — documented approximation, honest at 8-review scale).
 * Throws on length mismatch (pipeline bug); zeroShot[i] must describe reviews[i].
 */
export function aggregateAspects(
  zeroShot: ZeroShotResult[],
  reviews: ReviewScore[],
): AspectScore[] {
  if (zeroShot.length !== reviews.length) {
    throw new Error('aggregateAspects: zeroShot/reviews length mismatch');
  }

  const out: AspectScore[] = [];

  for (const label of ASPECT_LABELS) {
    const polarities: number[] = [];
    for (let i = 0; i < zeroShot.length; i++) {
      const idx = zeroShot[i].labels.indexOf(label);
      if (idx === -1) continue;
      const score = zeroShot[i].scores[idx];
      if (typeof score === 'number' && score >= MENTION_THRESHOLD) {
        polarities.push(reviews[i].roberta.positive - reviews[i].roberta.negative);
      }
    }
    if (polarities.length >= MIN_MENTIONS) {
      out.push({
        label,
        polarity: polarities.reduce((a, b) => a + b, 0) / polarities.length,
        mentions: polarities.length,
      });
    }
  }

  return out.sort((a, b) => b.mentions - a.mentions || a.label.localeCompare(b.label));
}
