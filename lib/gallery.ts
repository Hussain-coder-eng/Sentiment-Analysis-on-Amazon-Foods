export interface GalleryItem {
  asin: string;
  shortName: string;
  emoji: string;
}

/**
 * Known-cached ASINs (instant results, no Canopy call, no rate-limit cost).
 * Keep in sync with HANDOFF.md "Confirmed Working ASINs" table.
 * First item doubles as the product that emerges from the 3D box hero (Phase 3).
 */
export const GALLERY_ITEMS: GalleryItem[] = [
  { asin: 'B01B57DVNE', shortName: "Jack Link's Snack Sticks", emoji: '🥩' },
  { asin: 'B017835JPC', shortName: 'Teriyaki Snack Packs', emoji: '🍖' },
  { asin: 'B0C2FV4W2S', shortName: 'Snack Mix Variety', emoji: '🥨' },
  { asin: 'B000E7L2R4', shortName: 'Fine Foods Classic', emoji: '🛒' },
  { asin: 'B00032G1S0', shortName: 'Pantry Pick', emoji: '🍪' },
];
