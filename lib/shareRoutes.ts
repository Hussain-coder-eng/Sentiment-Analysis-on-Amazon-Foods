export const ASIN_PATTERN = /^[A-Z0-9]{10}$/;

export function normalizeAsinInput(value: string): string {
  const trimmed = value.trim();
  const match = trimmed.match(/(?:\/dp\/|\/product\/)([A-Z0-9]{10})/i);
  return (match ? match[1] : trimmed).toUpperCase();
}

export function isValidAsin(value: string): boolean {
  return ASIN_PATTERN.test(value);
}

export function sharePathForAsin(asin: string): string | null {
  const normalized = normalizeAsinInput(asin);
  return isValidAsin(normalized) ? `/p/${normalized}` : null;
}
