import { isValidAsin, normalizeAsinInput } from './shareRoutes';
import type { AspectScore, ReviewScore } from './types';

export const INVALID_ASIN_MESSAGE =
  'Invalid ASIN - must be 10 uppercase letters/digits (e.g. B000E7L2R4).';

export type InitialAsinDecision = {
  normalizedAsin: string;
  shouldAnalyze: boolean;
  error: string | null;
};

export type AnalysisStateReset = {
  analyzing: boolean;
  retryCountdown: number | null;
  reviews: ReviewScore[] | null;
  resultAsin: string | null;
  productTitle: string | undefined;
  aspects: AspectScore[] | undefined;
  shareStatus: string | null;
  shareError: string | null;
  analyzeError: string | null;
};

export function getInitialAsinDecision(initialAsin?: string): InitialAsinDecision {
  const normalizedAsin = initialAsin ? normalizeAsinInput(initialAsin) : '';

  if (!normalizedAsin) {
    return { normalizedAsin, shouldAnalyze: false, error: null };
  }

  if (!isValidAsin(normalizedAsin)) {
    return { normalizedAsin, shouldAnalyze: false, error: INVALID_ASIN_MESSAGE };
  }

  return { normalizedAsin, shouldAnalyze: true, error: null };
}

export function getInvalidAnalysisState(): AnalysisStateReset {
  return {
    analyzing: false,
    retryCountdown: null,
    reviews: null,
    resultAsin: null,
    productTitle: undefined,
    aspects: undefined,
    shareStatus: null,
    shareError: null,
    analyzeError: INVALID_ASIN_MESSAGE,
  };
}

export function shouldApplyRequestUpdate({
  activeRequestId,
  requestId,
}: {
  activeRequestId: number;
  requestId: number;
}): boolean {
  return activeRequestId === requestId;
}
