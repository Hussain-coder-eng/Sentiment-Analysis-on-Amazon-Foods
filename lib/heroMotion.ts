// ─── Stage boundaries ─────────────────────────────────────────────────────────
export const STAGE_IDLE_END = 0.05;       // below: box idles
export const STAGE_TUMBLE_END = 0.55;     // idle end → tumble end
export const STAGE_OPEN_END = 0.7;        // flaps fully open
export const STAGE_FADE_START = 0.85;     // box begins shrink/fade

// ─── Headline ─────────────────────────────────────────────────────────────────
export const HEADLINE_FADE_END = 0.35;    // headline fully gone

// ─── Card ─────────────────────────────────────────────────────────────────────
export const CARD_EMERGE_START = 0.6;
export const CARD_EMERGE_END = 0.75;

// ─── Overlay (card + chips + steps) fade-out ─────────────────────────────────
export const OVERLAY_FADE_START = 0.88;   // card+chips+steps fade out
export const OVERLAY_FADE_END = 0.97;

// ─── Story steps ─────────────────────────────────────────────────────────────
export const STORY_STEP_STARTS = [0.55, 0.62, 0.69] as const;
export const STORY_STEP_FADE_SPAN = 0.08;

// ─── Chips ────────────────────────────────────────────────────────────────────
export const CHIP_OFFSETS = [
  { dx: -90, dy: -50 },
  { dx: 80, dy: -30 },
  { dx: 10, dy: -80 },
] as const;
export const CHIP_START_STAGGER = 0.04;   // chip i starts at 0.62 + i * stagger
export const CHIP_SPAN = 0.1;

// Private base for chip start calculation — keeps function body magic-number free.
const CHIP_START_BASE = 0.62;

// ─── Types ────────────────────────────────────────────────────────────────────

export interface BoxPose {
  rotateX: number;    // deg
  rotateY: number;    // deg
  scale: number;
  translateX: number; // unitless; component decides units (vw-ish)
  translateY: number;
  opacity: number;
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/** Linearly maps value from [inMin, inMax] to [outMin, outMax]. No clamping. */
function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number {
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}

/** Overlay fade-out envelope: 1 before OVERLAY_FADE_START, 0 after OVERLAY_FADE_END. */
function overlayFadeOut(p: number): number {
  if (p <= OVERLAY_FADE_START) return 1;
  if (p >= OVERLAY_FADE_END) return 0;
  return mapRange(p, OVERLAY_FADE_START, OVERLAY_FADE_END, 1, 0);
}

// ─── Exported functions ───────────────────────────────────────────────────────

/** Clamps to [0, 1]. NaN → 0. */
export function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

/**
 * Piecewise linear box pose at scroll progress p.
 * Stages: idle (≤0.05) → tumble (0.05→0.55) → open/settle (0.55→0.85) → fade (0.85→1).
 */
export function boxPoseAt(progress: number): BoxPose {
  const p = clamp01(progress);

  // Idle
  if (p <= STAGE_IDLE_END) {
    return { rotateX: -12, rotateY: 20, scale: 1.25, translateX: 0, translateY: 0, opacity: 1 };
  }

  // Tumble: 0.05 → 0.55
  if (p <= STAGE_TUMBLE_END) {
    return {
      rotateY: mapRange(p, STAGE_IDLE_END, STAGE_TUMBLE_END, 20, 380),
      rotateX: mapRange(p, STAGE_IDLE_END, STAGE_TUMBLE_END, -12, 18),
      scale: mapRange(p, STAGE_IDLE_END, STAGE_TUMBLE_END, 1.25, 0.9),
      translateX: mapRange(p, STAGE_IDLE_END, STAGE_TUMBLE_END, 0, 18),
      translateY: mapRange(p, STAGE_IDLE_END, STAGE_TUMBLE_END, 0, 22),
      opacity: 1,
    };
  }

  // Open/settle: 0.55 → 0.85
  if (p <= STAGE_FADE_START) {
    return {
      rotateY: mapRange(p, STAGE_TUMBLE_END, STAGE_FADE_START, 380, 360),
      rotateX: mapRange(p, STAGE_TUMBLE_END, STAGE_FADE_START, 18, 0),
      scale: mapRange(p, STAGE_TUMBLE_END, STAGE_FADE_START, 0.9, 0.85),
      translateX: mapRange(p, STAGE_TUMBLE_END, STAGE_FADE_START, 18, 20),
      translateY: mapRange(p, STAGE_TUMBLE_END, STAGE_FADE_START, 22, 26),
      opacity: 1,
    };
  }

  // Fade: 0.85 → 1 — rotate/translate hold at 0.85 values
  return {
    rotateY: 360,
    rotateX: 0,
    translateX: 20,
    translateY: 26,
    scale: mapRange(p, STAGE_FADE_START, 1, 0.85, 0.4),
    opacity: mapRange(p, STAGE_FADE_START, 1, 1, 0),
  };
}

/** 0 before opening; linear 0→1 over [0.55, 0.7]; 1 after. Monotonic non-decreasing. */
export function flapProgressAt(progress: number): number {
  const p = clamp01(progress);
  if (p <= STAGE_TUMBLE_END) return 0;
  if (p >= STAGE_OPEN_END) return 1;
  return mapRange(p, STAGE_TUMBLE_END, STAGE_OPEN_END, 0, 1);
}

/** Headline fades out and rises over [0, 0.35]; clamped after. */
export function headlineStyleAt(progress: number): { opacity: number; translateY: number } {
  const p = clamp01(progress);
  if (p >= HEADLINE_FADE_END) return { opacity: 0, translateY: -60 };
  return {
    opacity: mapRange(p, 0, HEADLINE_FADE_END, 1, 0),
    translateY: mapRange(p, 0, HEADLINE_FADE_END, 0, -60),
  };
}

/**
 * Step i fades in 0→1 over [STORY_STEP_STARTS[i], STORY_STEP_STARTS[i] + STORY_STEP_FADE_SPAN],
 * then ALL steps fade 1→0 via the overlay envelope. Returns min of both envelopes.
 * stepIndex outside 0–2 → 0.
 */
export function storyStepOpacityAt(progress: number, stepIndex: number): number {
  if (stepIndex < 0 || stepIndex > 2) return 0;
  const p = clamp01(progress);
  const start = STORY_STEP_STARTS[stepIndex];
  const end = start + STORY_STEP_FADE_SPAN;

  let fadeIn: number;
  if (p <= start) {
    fadeIn = 0;
  } else if (p >= end) {
    fadeIn = 1;
  } else {
    fadeIn = mapRange(p, start, end, 0, 1);
  }

  return Math.min(fadeIn, overlayFadeOut(p));
}

/**
 * Card emerges linearly 0→1 over [0.6, 0.75]; opacity additionally multiplied by overlay
 * fade-out envelope. translateY/scale hold emerged values during fade-out.
 */
export function cardEmergeAt(
  progress: number,
): { opacity: number; translateY: number; scale: number } {
  const p = clamp01(progress);

  if (p <= CARD_EMERGE_START) {
    return { opacity: 0, translateY: 40, scale: 0.8 };
  }

  const localProgress =
    p >= CARD_EMERGE_END
      ? 1
      : mapRange(p, CARD_EMERGE_START, CARD_EMERGE_END, 0, 1);

  const opacity = localProgress * overlayFadeOut(p);
  const translateY = mapRange(localProgress, 0, 1, 40, 0);
  const scale = mapRange(localProgress, 0, 1, 0.8, 1);

  return { opacity, translateY, scale };
}

/**
 * Chip i emerges over [start, start + CHIP_SPAN] with offset scaled by local progress.
 * Opacity multiplied by overlay fade-out. chipIndex outside 0–2 → zeroed.
 */
export function chipStyleAt(
  progress: number,
  chipIndex: number,
): { opacity: number; translateX: number; translateY: number } {
  if (chipIndex < 0 || chipIndex > 2) {
    return { opacity: 0, translateX: 0, translateY: 0 };
  }

  const p = clamp01(progress);
  const start = CHIP_START_BASE + chipIndex * CHIP_START_STAGGER;
  const end = start + CHIP_SPAN;
  const offset = CHIP_OFFSETS[chipIndex];

  let localProgress: number;
  if (p <= start) {
    localProgress = 0;
  } else if (p >= end) {
    localProgress = 1;
  } else {
    localProgress = mapRange(p, start, end, 0, 1);
  }

  return {
    opacity: localProgress * overlayFadeOut(p),
    translateX: offset.dx * localProgress,
    translateY: offset.dy * localProgress,
  };
}
