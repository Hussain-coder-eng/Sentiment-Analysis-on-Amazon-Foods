// Spike: validate HF Inference API for cardiffnlp/twitter-roberta-base-sentiment-latest
// Run with: HF_API_KEY=hf_xxx node spikes/pipe1-hf.js
// New endpoint: router.huggingface.co (old api-inference.huggingface.co deprecated/NXDOMAIN)
// Must use -latest model: base model returns LABEL_0/1/2, -latest returns named labels

const HF_URL = 'https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest';
const API_KEY = process.env.HF_API_KEY;

if (!API_KEY) {
  console.error('Error: HF_API_KEY env var not set');
  process.exit(1);
}

async function callHF(inputs) {
  const res = await fetch(HF_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ inputs }),
    signal: AbortSignal.timeout(30000),
  });
  console.log(`Status: ${res.status}`);
  const data = await res.json();
  console.log('Response:', JSON.stringify(data, null, 2));
  return { status: res.status, data };
}

(async () => {
  console.log('\n=== Test 1: Single string input ===');
  await callHF('This product is absolutely amazing and I love it!');

  console.log('\n=== Test 2: Array input (batch) ===');
  await callHF([
    'This product is absolutely amazing and I love it!',
    'Terrible quality, completely disappointed.',
  ]);
})();
