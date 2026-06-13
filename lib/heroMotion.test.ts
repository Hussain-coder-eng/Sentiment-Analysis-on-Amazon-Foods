import { describe, it, expect } from 'vitest';
import {
  clamp01,
  boxPoseAt,
  flapProgressAt,
  headlineStyleAt,
  storyStepOpacityAt,
  cardEmergeAt,
  chipStyleAt,
  STAGE_IDLE_END,
  STAGE_TUMBLE_END,
  STAGE_OPEN_END,
  STAGE_FADE_START,
  HEADLINE_FADE_END,
  CARD_EMERGE_END,
  OVERLAY_FADE_START,
  OVERLAY_FADE_END,
  STORY_STEP_STARTS,
  STORY_STEP_FADE_SPAN,
  CHIP_OFFSETS,
  CHIP_START_STAGGER,
  CHIP_SPAN,
} from './heroMotion';

// ─── clamp01 ─────────────────────────────────────────────────────────────────

describe('clamp01', () => {
  it('clamps negative to 0', () => expect(clamp01(-1)).toBe(0));
  it('clamps > 1 to 1', () => expect(clamp01(2)).toBe(1));
  it('NaN → 0', () => expect(clamp01(NaN)).toBe(0));
  it('0 → 0', () => expect(clamp01(0)).toBe(0));
  it('1 → 1', () => expect(clamp01(1)).toBe(1));
  it('0.5 → 0.5', () => expect(clamp01(0.5)).toBe(0.5));
});

// ─── boxPoseAt ───────────────────────────────────────────────────────────────

describe('boxPoseAt', () => {
  it('p = 0 → idle pose', () => {
    const pose = boxPoseAt(0);
    expect(pose.rotateX).toBe(-12);
    expect(pose.rotateY).toBe(20);
    expect(pose.scale).toBe(1.25);
    expect(pose.translateX).toBe(0);
    expect(pose.translateY).toBe(0);
    expect(pose.opacity).toBe(1);
  });

  it('p = 0.05 → still idle pose (boundary)', () => {
    const pose = boxPoseAt(STAGE_IDLE_END);
    expect(pose.rotateX).toBe(-12);
    expect(pose.rotateY).toBe(20);
    expect(pose.scale).toBe(1.25);
    expect(pose.opacity).toBe(1);
  });

  it('p = 0.3 → midpoint of tumble: rotateY ≈ 200', () => {
    // 0.3 is midpoint of [0.05, 0.55] range (0.05 + (0.55-0.05)/2 = 0.3)
    const pose = boxPoseAt(0.3);
    expect(pose.rotateY).toBeCloseTo(200, 5);
  });

  it('p = 0.55 → end of tumble', () => {
    const pose = boxPoseAt(STAGE_TUMBLE_END);
    expect(pose.rotateY).toBeCloseTo(380, 5);
    expect(pose.rotateX).toBeCloseTo(18, 5);
    expect(pose.scale).toBeCloseTo(0.9, 5);
    expect(pose.translateX).toBeCloseTo(18, 5);
    expect(pose.translateY).toBeCloseTo(22, 5);
    expect(pose.opacity).toBe(1);
  });

  it('p = 0.7 → in open/settle phase', () => {
    // 0.7 is midpoint of [0.55, 0.85]
    const pose = boxPoseAt(0.7);
    expect(pose.rotateY).toBeCloseTo(370, 5);
    expect(pose.rotateX).toBeCloseTo(9, 5);
    expect(pose.scale).toBeCloseTo(0.875, 5);
    expect(pose.opacity).toBe(1);
  });

  it('p = 0.85 → end of open/settle, opacity still 1', () => {
    const pose = boxPoseAt(STAGE_FADE_START);
    expect(pose.rotateY).toBeCloseTo(360, 5);
    expect(pose.rotateX).toBeCloseTo(0, 5);
    expect(pose.scale).toBeCloseTo(0.85, 5);
    expect(pose.translateX).toBeCloseTo(20, 5);
    expect(pose.translateY).toBeCloseTo(26, 5);
    expect(pose.opacity).toBe(1);
  });

  it('p = 1 → docked beside the form as a visible accent', () => {
    const pose = boxPoseAt(1);
    expect(pose.scale).toBeCloseTo(0.48, 5);
    expect(pose.opacity).toBeCloseTo(0.92, 5);
    expect(pose.rotateY).toBeCloseTo(366, 5);
    expect(pose.rotateX).toBeCloseTo(0, 5);
    expect(pose.translateX).toBeCloseTo(24, 5);
    expect(pose.translateY).toBeCloseTo(32, 5);
  });

  it('p < 0 → clamps to idle pose', () => {
    const pose = boxPoseAt(-0.5);
    expect(pose.rotateY).toBe(20);
    expect(pose.scale).toBe(1.25);
  });

  it('p > 1 → clamps to fade-end pose', () => {
    const pose = boxPoseAt(2);
    expect(pose.scale).toBeCloseTo(0.48, 5);
    expect(pose.opacity).toBeCloseTo(0.92, 5);
  });
});

// ─── flapProgressAt ──────────────────────────────────────────────────────────

describe('flapProgressAt', () => {
  it('0 at p = 0.55 (start of open)', () => {
    expect(flapProgressAt(STAGE_TUMBLE_END)).toBe(0);
  });

  it('0.5 at p = 0.625 (midpoint)', () => {
    expect(flapProgressAt(0.625)).toBeCloseTo(0.5, 5);
  });

  it('1 at p = 0.7 (fully open)', () => {
    expect(flapProgressAt(STAGE_OPEN_END)).toBeCloseTo(1, 5);
  });

  it('1 at p = 1 (holds after open)', () => {
    expect(flapProgressAt(1)).toBe(1);
  });

  it('0 before open starts', () => {
    expect(flapProgressAt(0)).toBe(0);
    expect(flapProgressAt(0.3)).toBe(0);
  });

  it('monotonic non-decreasing across sweep', () => {
    let prev = flapProgressAt(0);
    for (let i = 1; i <= 100; i++) {
      const cur = flapProgressAt(i / 100);
      expect(cur).toBeGreaterThanOrEqual(prev - 1e-10);
      prev = cur;
    }
  });
});

// ─── headlineStyleAt ──────────────────────────────────────────────────────────

describe('headlineStyleAt', () => {
  it('at p = 0 → opacity 1, translateY 0', () => {
    const s = headlineStyleAt(0);
    expect(s.opacity).toBe(1);
    expect(s.translateY).toBe(0);
  });

  it('at p = 0.175 → opacity 0.5, translateY −30', () => {
    const s = headlineStyleAt(0.175);
    expect(s.opacity).toBeCloseTo(0.5, 5);
    expect(s.translateY).toBeCloseTo(-30, 5);
  });

  it('at p = 0.35 → opacity 0, translateY −60', () => {
    const s = headlineStyleAt(HEADLINE_FADE_END);
    expect(s.opacity).toBeCloseTo(0, 5);
    expect(s.translateY).toBeCloseTo(-60, 5);
  });

  it('at p = 1 → clamped: opacity 0, translateY −60', () => {
    const s = headlineStyleAt(1);
    expect(s.opacity).toBeCloseTo(0, 5);
    expect(s.translateY).toBeCloseTo(-60, 5);
  });
});

// ─── storyStepOpacityAt ──────────────────────────────────────────────────────

describe('storyStepOpacityAt', () => {
  it('invalid index → 0', () => {
    expect(storyStepOpacityAt(0.6, -1)).toBe(0);
    expect(storyStepOpacityAt(0.6, 3)).toBe(0);
  });

  it('each step is 0 before its start', () => {
    for (let i = 0; i < 3; i++) {
      expect(storyStepOpacityAt(STORY_STEP_STARTS[i] - 0.001, i)).toBeCloseTo(0, 5);
    }
  });

  it('each step is 1 after start + span', () => {
    for (let i = 0; i < 3; i++) {
      expect(storyStepOpacityAt(STORY_STEP_STARTS[i] + STORY_STEP_FADE_SPAN + 0.001, i)).toBeCloseTo(1, 5);
    }
  });

  it('fade-out envelope at 0.925 → approx 0.5 for all steps', () => {
    // 0.925 is midpoint of [OVERLAY_FADE_START=0.88, OVERLAY_FADE_END=0.97]
    for (let i = 0; i < 3; i++) {
      const val = storyStepOpacityAt(0.925, i);
      expect(val).toBeCloseTo(0.5, 1);
    }
  });

  it('0 at or after OVERLAY_FADE_END for all steps', () => {
    for (let i = 0; i < 3; i++) {
      expect(storyStepOpacityAt(OVERLAY_FADE_END, i)).toBeCloseTo(0, 5);
      expect(storyStepOpacityAt(1, i)).toBeCloseTo(0, 5);
    }
  });
});

// ─── cardEmergeAt ─────────────────────────────────────────────────────────────

describe('cardEmergeAt', () => {
  it('hidden before 0.6', () => {
    const c = cardEmergeAt(0);
    expect(c.opacity).toBe(0);
    expect(c.translateY).toBe(40);
    expect(c.scale).toBeCloseTo(0.8, 5);
  });

  it('midpoint 0.675 → opacity 0.5, translateY 20, scale 0.9', () => {
    const c = cardEmergeAt(0.675);
    expect(c.opacity).toBeCloseTo(0.5, 5);
    expect(c.translateY).toBeCloseTo(20, 5);
    expect(c.scale).toBeCloseTo(0.9, 5);
  });

  it('fully emerged at 0.75 → opacity 1, translateY 0, scale 1', () => {
    const c = cardEmergeAt(CARD_EMERGE_END);
    expect(c.opacity).toBeCloseTo(1, 5);
    expect(c.translateY).toBeCloseTo(0, 5);
    expect(c.scale).toBeCloseTo(1, 5);
  });

  it('holds emerged at 0.88 (just before overlay fade starts)', () => {
    const c = cardEmergeAt(OVERLAY_FADE_START);
    expect(c.opacity).toBeCloseTo(1, 5);
    expect(c.translateY).toBeCloseTo(0, 5);
    expect(c.scale).toBeCloseTo(1, 5);
  });

  it('faded out by 0.97 → opacity 0', () => {
    const c = cardEmergeAt(OVERLAY_FADE_END);
    expect(c.opacity).toBeCloseTo(0, 5);
    // translateY and scale hold emerged values
    expect(c.translateY).toBeCloseTo(0, 5);
    expect(c.scale).toBeCloseTo(1, 5);
  });
});

// ─── chipStyleAt ──────────────────────────────────────────────────────────────

describe('chipStyleAt', () => {
  it('invalid index → zeroed', () => {
    const c = chipStyleAt(0.7, -1);
    expect(c).toEqual({ opacity: 0, translateX: 0, translateY: 0 });
    const c2 = chipStyleAt(0.7, 3);
    expect(c2).toEqual({ opacity: 0, translateX: 0, translateY: 0 });
  });

  it('chip 0: hidden before its start (0.62)', () => {
    const c = chipStyleAt(0.619, 0);
    expect(c.opacity).toBeCloseTo(0, 5);
    expect(c.translateX).toBeCloseTo(0, 5);
    expect(c.translateY).toBeCloseTo(0, 5);
  });

  it('chip 0: stagger start respected at localProgress 0.5 (midpoint)', () => {
    // chip 0: start = 0.62, span = 0.1 → midpoint at 0.67
    const c = chipStyleAt(0.67, 0);
    const offset = CHIP_OFFSETS[0];
    expect(c.opacity).toBeCloseTo(0.5, 5);
    expect(c.translateX).toBeCloseTo(offset.dx * 0.5, 5);
    expect(c.translateY).toBeCloseTo(offset.dy * 0.5, 5);
  });

  it('chip 1: stagger respected (start = 0.62 + 1 * 0.04 = 0.66)', () => {
    const start1 = 0.62 + 1 * CHIP_START_STAGGER;
    const c = chipStyleAt(start1 + CHIP_SPAN * 0.5, 1);
    const offset = CHIP_OFFSETS[1];
    expect(c.opacity).toBeCloseTo(0.5, 5);
    expect(c.translateX).toBeCloseTo(offset.dx * 0.5, 5);
    expect(c.translateY).toBeCloseTo(offset.dy * 0.5, 5);
  });

  it('chip 2: stagger respected (start = 0.62 + 2 * 0.04 = 0.70)', () => {
    const start2 = 0.62 + 2 * CHIP_START_STAGGER;
    const c = chipStyleAt(start2 + CHIP_SPAN * 0.5, 2);
    const offset = CHIP_OFFSETS[2];
    expect(c.opacity).toBeCloseTo(0.5, 5);
    expect(c.translateX).toBeCloseTo(offset.dx * 0.5, 5);
    expect(c.translateY).toBeCloseTo(offset.dy * 0.5, 5);
  });

  it('chip fully emerged then faded by OVERLAY_FADE_END', () => {
    const c = chipStyleAt(OVERLAY_FADE_END, 0);
    expect(c.opacity).toBeCloseTo(0, 5);
  });

  it('offsets scale with local progress', () => {
    const c0 = chipStyleAt(0.62, 0);
    const c1 = chipStyleAt(0.72, 0);
    // At full local progress, translateX = CHIP_OFFSETS[0].dx
    expect(c1.translateX).toBeCloseTo(CHIP_OFFSETS[0].dx, 5);
    // At start, translateX = 0
    expect(c0.translateX).toBeCloseTo(0, 5);
  });
});
