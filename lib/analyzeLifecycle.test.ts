import { describe, expect, it } from 'vitest';
import type { AnalyzeApiResponse } from './types';
import {
  INVALID_ASIN_MESSAGE,
  getInitialAsinDecision,
  getInvalidAnalysisState,
  shouldApplyRequestUpdate,
} from './analyzeLifecycle';

describe('analyze lifecycle helpers', () => {
  it('rejects invalid initial route ASINs before analysis can start', () => {
    expect(getInitialAsinDecision('notvalid')).toEqual({
      normalizedAsin: 'NOTVALID',
      shouldAnalyze: false,
      error: INVALID_ASIN_MESSAGE,
    });
  });

  it('accepts valid initial route ASINs after normalizing input', () => {
    expect(getInitialAsinDecision('b000e7l2r4')).toEqual({
      normalizedAsin: 'B000E7L2R4',
      shouldAnalyze: true,
      error: null,
    });
  });

  it('clears analysis and sharing state when an invalid ASIN resets the view', () => {
    const staleResponse: AnalyzeApiResponse = {
      asin: 'B000E7L2R4',
      count: 1,
      reviews: [
        {
          text: 'stale review',
          rating: 5,
          vader: { compound: 0.8, pos: 0.8, neg: 0, neu: 0.2 },
          roberta: { positive: 0.9, neutral: 0.08, negative: 0.02 },
          disagreement: 0.1,
        },
      ],
      productTitle: 'Old Product',
      aspects: [{ label: 'taste & flavor', polarity: 0.7, mentions: 2 }],
    };

    expect(staleResponse.reviews).toHaveLength(1);
    expect(getInvalidAnalysisState()).toEqual({
      analyzing: false,
      retryCountdown: null,
      reviews: null,
      resultAsin: null,
      productTitle: undefined,
      aspects: undefined,
      shareStatus: null,
      shareError: null,
      analyzeError: INVALID_ASIN_MESSAGE,
    });
  });

  it('allows only the active request to update analysis state', () => {
    expect(shouldApplyRequestUpdate({ activeRequestId: 2, requestId: 1 })).toBe(false);
    expect(shouldApplyRequestUpdate({ activeRequestId: 2, requestId: 2 })).toBe(true);
  });
});
