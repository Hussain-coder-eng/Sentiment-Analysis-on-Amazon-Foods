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

export interface DemoApiResponse {
  reviews: ReviewScore[];
  count: number;
  asin: null;
}

export interface AnalyzeApiResponse {
  reviews: ReviewScore[];
  count: number;
  asin: string;
}
