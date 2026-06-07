import { NextResponse } from "next/server";

// Temporary spike validation route — delete after spike gate passes.

const HF_BASE_URL =
  "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment";
const HF_LATEST_URL =
  "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest";
const CANOPY_URL =
  "https://rest.canopyapi.co/v1/amazon/product/reviews?asin=B07FZ8S74R&domain=US&page=1";
const HF_INPUT = { inputs: "I love this product" };

function extractHfLabels(raw: unknown): string[] | null {
  try {
    // HF returns [[{label, score}, ...]] or [{label, score}, ...]
    const arr = Array.isArray(raw) ? raw : null;
    if (!arr) return null;
    const inner = Array.isArray(arr[0]) ? arr[0] : arr;
    return (inner as Array<{ label: string }>).map((x) => x.label);
  } catch {
    return null;
  }
}

async function testHf(
  url: string,
  token: string
): Promise<{ status: number; labels: string[] | null; raw: unknown; error?: string; error_cause_code?: string | null; error_cause_message?: string | null }> {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(HF_INPUT),
      signal: AbortSignal.timeout(60_000),
    });
    const raw: unknown = await res.json().catch(() => null);
    return { status: res.status, labels: extractHfLabels(raw), raw };
  } catch (err) {
    return {
      status: 0,
      labels: null,
      raw: null,
      error: String(err),
      error_cause_code: (err as NodeJS.ErrnoException & { cause?: NodeJS.ErrnoException })?.cause?.code ?? null,
      error_cause_message: (err as NodeJS.ErrnoException & { cause?: NodeJS.ErrnoException })?.cause?.message ?? null,
    };
  }
}

async function testCanopy(
  apiKey: string
): Promise<{
  status: number;
  reviews_count: number | null;
  first_review_id: string | null;
  first_body_preview: string | null;
  raw_keys: string[];
  error?: string;
  error_cause_code?: string | null;
  error_cause_message?: string | null;
}> {
  try {
    const res = await fetch(CANOPY_URL, {
      headers: { "API-KEY": apiKey },
      signal: AbortSignal.timeout(20_000),
    });
    const raw: unknown = await res.json().catch(() => null);
    const data = (raw as Record<string, unknown> | null)?.data as
      | Record<string, unknown>
      | undefined;
    const reviews: Array<{ id?: string; body?: string }> =
      (data?.reviewsPaginated as Record<string, unknown> | undefined)
        ?.reviews as Array<{ id?: string; body?: string }> ?? [];
    return {
      status: res.status,
      reviews_count: reviews.length,
      first_review_id: reviews[0]?.id ?? null,
      first_body_preview: reviews[0]?.body?.slice(0, 100) ?? null,
      raw_keys: raw && typeof raw === "object" ? Object.keys(raw as object) : [],
    };
  } catch (err) {
    return {
      status: 0,
      reviews_count: null,
      first_review_id: null,
      first_body_preview: null,
      raw_keys: [],
      error: String(err),
      error_cause_code: (err as NodeJS.ErrnoException & { cause?: NodeJS.ErrnoException })?.cause?.code ?? null,
      error_cause_message: (err as NodeJS.ErrnoException & { cause?: NodeJS.ErrnoException })?.cause?.message ?? null,
    };
  }
}

async function testControl(): Promise<{ status: number; ok: boolean; error?: string; error_cause_code?: string | null; error_cause_message?: string | null }> {
  try {
    const res = await fetch('https://httpbin.org/get', { signal: AbortSignal.timeout(10_000) });
    return { status: res.status, ok: res.ok };
  } catch (err) {
    return {
      status: 0,
      ok: false,
      error: String(err),
      error_cause_code: (err as NodeJS.ErrnoException & { cause?: NodeJS.ErrnoException })?.cause?.code ?? null,
      error_cause_message: (err as NodeJS.ErrnoException & { cause?: NodeJS.ErrnoException })?.cause?.message ?? null,
    };
  }
}

export async function GET() {
  const hfToken = process.env.HF_API_KEY ?? "";
  const canopyKey = process.env.CANOPY_API_KEY ?? "";

  const [hfBaseResult, hfLatestResult, canopyResult, controlResult] = await Promise.allSettled([
    testHf(HF_BASE_URL, hfToken),
    testHf(HF_LATEST_URL, hfToken),
    testCanopy(canopyKey),
    testControl(),
  ]);

  return NextResponse.json({
    hf_base:
      hfBaseResult.status === "fulfilled"
        ? hfBaseResult.value
        : { status: 0, labels: null, raw: null, error: String(hfBaseResult.reason) },
    hf_latest:
      hfLatestResult.status === "fulfilled"
        ? hfLatestResult.value
        : { status: 0, labels: null, raw: null, error: String(hfLatestResult.reason) },
    canopy:
      canopyResult.status === "fulfilled"
        ? canopyResult.value
        : {
            status: 0,
            reviews_count: null,
            first_review_id: null,
            first_body_preview: null,
            raw_keys: [],
            error: String(canopyResult.reason),
          },
    control:
      controlResult.status === "fulfilled"
        ? controlResult.value
        : { status: 0, ok: false, error: String(controlResult.reason) },
  });
}
