import { describe, expect, it } from 'vitest';
import { isValidAsin, normalizeAsinInput, sharePathForAsin } from './shareRoutes';

describe('share route helpers', () => {
  it('normalizes raw ASIN input', () => {
    expect(normalizeAsinInput(' b000e7l2r4 ')).toBe('B000E7L2R4');
  });

  it('extracts ASINs from Amazon product URLs', () => {
    expect(normalizeAsinInput('https://www.amazon.com/example/dp/b000e7l2r4?th=1')).toBe(
      'B000E7L2R4'
    );
    expect(normalizeAsinInput('https://www.amazon.com/product/B000E7L2R4')).toBe('B000E7L2R4');
  });

  it('validates exactly 10 alphanumeric uppercase characters', () => {
    expect(isValidAsin('B000E7L2R4')).toBe(true);
    expect(isValidAsin('NOTVALID')).toBe(false);
    expect(isValidAsin('B000E7L2R!')).toBe(false);
  });

  it('builds share paths only for valid ASINs', () => {
    expect(sharePathForAsin('b000e7l2r4')).toBe('/p/B000E7L2R4');
    expect(sharePathForAsin('notvalid')).toBeNull();
  });
});
