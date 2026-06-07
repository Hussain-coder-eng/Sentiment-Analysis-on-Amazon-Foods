// Spike: validate Canopy API for Amazon product reviews
// Run with: CANOPY_API_KEY=xxx node spikes/pipe2-canopy.js
// New endpoint: rest.canopyapi.co (old api.canopyapi.co deprecated/NXDOMAIN)
// New structure: data.amazonProduct.topReviews (old reviewsPaginated.reviews gone)

const ASIN = 'B07FZ8S74R';
const API_KEY = process.env.CANOPY_API_KEY;

if (!API_KEY) {
  console.error('Error: CANOPY_API_KEY env var not set');
  process.exit(1);
}

(async () => {
  const res = await fetch(
    `https://rest.canopyapi.co/api/amazon/product/reviews?asin=${ASIN}&domain=US&page=1`,
    {
      headers: { 'API-KEY': API_KEY },
      signal: AbortSignal.timeout(10000),
    }
  );

  console.log(`Status: ${res.status}`);
  const data = await res.json();

  const reviews = data?.data?.amazonProduct?.topReviews ?? [];
  console.log(`Reviews count: ${reviews.length}`);

  if (reviews.length > 0) {
    console.log('\nFirst review fields:');
    console.log(`  id: ${reviews[0]?.id}`);
    console.log(`  body (first 200 chars): ${String(reviews[0]?.body ?? '').slice(0, 200)}`);
    console.log(`  rating: ${reviews[0]?.rating}`);
    console.log(`  verifiedPurchase: ${reviews[0]?.verifiedPurchase}`);
  }
})();
