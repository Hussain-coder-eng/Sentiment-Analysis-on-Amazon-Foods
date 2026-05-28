import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { ReviewScore } from '@/lib/types';

interface Props {
  reviews: ReviewScore[];
}

export default function DisagreementPanel({ reviews }: Props) {
  const top10 = [...reviews]
    .sort((a, b) => b.disagreement - a.disagreement)
    .slice(0, 10);

  return (
    <div className="mt-8">
      <h2 className="text-xl font-semibold mb-4">Top 10 Model Disagreements</h2>
      <p className="text-sm text-gray-500 mb-4">
        Reviews where VADER and RoBERTa disagree most — the most interesting cases.
      </p>

      <div className="grid gap-3">
        {top10.map((review, i) => (
          <Card key={i}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex justify-between">
                <span>
                  #{i + 1} — Disagreement: {review.disagreement.toFixed(3)}
                </span>
                <span className="text-gray-500">{review.rating}★</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-700 mb-2 line-clamp-3">
                {review.text.replace(/<[^>]*>/g, '')}
              </p>
              <div className="text-xs text-gray-500 flex gap-4">
                <span>VADER compound: {review.vader.compound.toFixed(3)}</span>
                <span>RoBERTa positive: {review.roberta.positive.toFixed(3)}</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
