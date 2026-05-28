'use client';

import dynamic from 'next/dynamic';
import type { ComponentType, CSSProperties } from 'react';
import type { Layout, PlotData } from 'plotly.js';

import type { ReviewScore } from '@/lib/types';

interface ReactPlotlyProps {
  data: Array<Partial<PlotData>>;
  layout: Partial<Layout>;
  useResizeHandler?: boolean;
  style?: CSSProperties;
  config?: {
    displayModeBar?: boolean;
  };
}

// @ts-expect-error react-plotly.js does not ship TypeScript declarations in this project.
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-96 text-gray-500">
      Loading chart...
    </div>
  ),
}) as ComponentType<ReactPlotlyProps>;

interface SentimentPlotProps {
  reviews: ReviewScore[];
}

const HTML_TAG_PATTERN = /<[^>]*>/g;
const MAX_HOVER_TEXT_LENGTH = 200;
const MIN_VADER_COMPOUND = -1;
const MAX_VADER_COMPOUND = 1;
const MIN_ROBERTA_POSITIVE = 0;
const MAX_ROBERTA_POSITIVE = 1;
const CHART_HEIGHT_PX = 500;

function stripHtmlTags(text: string): string {
  return text.replace(HTML_TAG_PATTERN, '').slice(0, MAX_HOVER_TEXT_LENGTH);
}

function formatScore(score: number): string {
  return score.toFixed(3);
}

export default function SentimentPlot({ reviews }: SentimentPlotProps) {
  const customdata = reviews.map((review) => [
    stripHtmlTags(review.text),
    review.rating.toString(),
    formatScore(review.vader.compound),
    formatScore(review.vader.pos),
    formatScore(review.vader.neg),
    formatScore(review.vader.neu),
    formatScore(review.roberta.positive),
    formatScore(review.roberta.neutral),
    formatScore(review.roberta.negative),
    formatScore(review.disagreement),
  ]);

  const data: Array<Partial<PlotData>> = [
    {
      type: 'scatter',
      mode: 'markers',
      x: reviews.map((review) => review.vader.compound),
      y: reviews.map((review) => review.roberta.positive),
      customdata,
      marker: {
        color: reviews.map((review) => review.disagreement),
        colorscale: [
          [0, 'blue'],
          [0.5, 'yellow'],
          [1, 'red'],
        ],
        showscale: true,
        colorbar: { title: { text: 'Disagreement' } },
        cmin: 0,
        cmax: 2,
        size: 7,
        opacity: 0.7,
      },
      hovertemplate:
        '<b>Review</b>: %{customdata[0]}<br>' +
        '<b>Rating</b>: %{customdata[1]}<br>' +
        '<b>VADER compound</b>: %{customdata[2]}<br>' +
        '<b>VADER pos</b>: %{customdata[3]}<br>' +
        '<b>VADER neg</b>: %{customdata[4]}<br>' +
        '<b>VADER neu</b>: %{customdata[5]}<br>' +
        '<b>RoBERTa positive</b>: %{customdata[6]}<br>' +
        '<b>RoBERTa neutral</b>: %{customdata[7]}<br>' +
        '<b>RoBERTa negative</b>: %{customdata[8]}<br>' +
        '<b>Disagreement</b>: %{customdata[9]}<extra></extra>',
    },
  ];

  const layout: Partial<Layout> = {
    title: {
      text: 'VADER vs RoBERTa Sentiment',
    },
    xaxis: {
      title: {
        text: 'VADER Compound Score',
      },
      range: [MIN_VADER_COMPOUND, MAX_VADER_COMPOUND],
    },
    yaxis: {
      title: {
        text: 'RoBERTa Positive Score',
      },
      range: [MIN_ROBERTA_POSITIVE, MAX_ROBERTA_POSITIVE],
    },
    hovermode: 'closest',
    autosize: true,
  };

  return (
    <Plot
      data={data}
      layout={layout}
      useResizeHandler
      style={{ width: '100%', height: `${CHART_HEIGHT_PX}px` }}
      config={{ displayModeBar: false }}
    />
  );
}
