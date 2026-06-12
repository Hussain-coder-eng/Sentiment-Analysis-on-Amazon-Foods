export interface VaderScore {
  compound: number; // -1 to +1
  pos: number;      // 0 to 1
  neg: number;      // 0 to 1
  neu: number;      // 0 to 1
}

export interface RobertaScore {
  positive: number; // 0 to 1
  neutral: number;  // 0 to 1
  negative: number; // 0 to 1
}

export interface ReviewScore {
  text: string;           // plain text, HTML-stripped, max 1000 chars
  rating: number;         // 1-5 stars
  vader: VaderScore;
  roberta: RobertaScore;
  disagreement: number;   // |vader.compound - (roberta.positive - roberta.negative)|, range [0,2]
}

export interface AspectScore {
  label: string;     // one of the candidate aspect labels
  polarity: number;  // mean RoBERTa polarity of mentioning reviews, [-1, 1]
  mentions: number;  // count of reviews mentioning this aspect
}

/** Value stored at asin:v2:<ASIN>:scored (24h TTL). v1 keys held bare ReviewScore[]. */
export interface ScoredCacheV2 {
  reviews: ReviewScore[];
  productTitle?: string;
  aspects?: AspectScore[];
  analyzedAt: string; // ISO timestamp
}

export interface AnalyzeApiResponse {
  reviews: ReviewScore[];
  count: number;
  asin: string;
  productTitle?: string;
  aspects?: AspectScore[];
}
