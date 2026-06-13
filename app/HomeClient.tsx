'use client';

import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
} from 'react';
import AspectBars from '@/components/AspectBars';
import DisagreementPanel from '@/components/DisagreementPanel';
import HowItWorksStrip from '@/components/HowItWorksStrip';
import SentimentPlot from '@/components/SentimentPlot';
import VerdictCard from '@/components/VerdictCard';
import CardboardBox from '@/components/box/CardboardBox';
import {
  STAGE_OPEN_END,
  boxPoseAt,
  cardEmergeAt,
  chipStyleAt,
  clamp01,
  flapProgressAt,
  headlineStyleAt,
  storyStepOpacityAt,
} from '@/lib/heroMotion';
import { GALLERY_ITEMS } from '@/lib/gallery';
import { isValidAsin, normalizeAsinInput, sharePathForAsin } from '@/lib/shareRoutes';
import type { AnalyzeApiResponse, AspectScore, ReviewScore } from '@/lib/types';
import { computeVerdict } from '@/lib/verdict';

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 30_000;
const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';
const MOBILE_QUERY = '(max-width: 767px)';
const STORY_SCROLL_HEIGHT_CLASS = 'min-h-[220svh] md:min-h-[235svh]';
const SCROLL_PROGRESS_EPSILON = 0.001;

const HERO_BOX_SIZE_DESKTOP = 252;
const HERO_BOX_SIZE_MOBILE = 200;

// Tuned against the sticky viewport to keep the box in-frame on desktop and mobile.
const BOX_TRANSLATE_X_FACTOR_DESKTOP = 14;
const BOX_TRANSLATE_Y_FACTOR_DESKTOP = 10;
const BOX_TRANSLATE_X_FACTOR_MOBILE = 7;
const BOX_TRANSLATE_Y_FACTOR_MOBILE = 6;
const DOCK_BOX_X_OFFSET_DESKTOP = 380;
const DOCK_BOX_Y_OFFSET_DESKTOP = 320;
const DOCK_BOX_X_OFFSET_MOBILE = 18;
const DOCK_BOX_Y_OFFSET_MOBILE = 12;

const IDLE_BOX_ANIMATION_END = 0.08;
const DOCK_BOX_SHIFT_START = 0.88;
const FORM_REVEAL_START = 0.84;
const FORM_REVEAL_END = 1;
const GALLERY_REVEAL_START = 0.9;
const GALLERY_REVEAL_END = 1;
const HERO_CARD_INTERACTIVE_OPACITY_MIN = 0.12;
const GALLERY_INTERACTIVE_REVEAL_MIN = 0.12;

const HERO_STEPS = [
  {
    eyebrow: 'Step 1',
    title: 'Collect review language',
    body: 'Pull real Amazon review text for the ASIN you paste or tap from the gallery.',
  },
  {
    eyebrow: 'Step 2',
    title: 'Score with two sentiment models',
    body: 'Run VADER and RoBERTa together so the fast lexical read has a transformer check.',
  },
  {
    eyebrow: 'Step 3',
    title: 'Surface the friction points',
    body: 'Summarize verdict, aspect sentiment, and reviewer disagreement in one pass.',
  },
] as const;

const HERO_SENTIMENT_CHIPS = [
  { label: 'Mostly positive', toneClass: 'border-green-500/35 bg-green-500/12 text-green-200' },
  { label: 'Taste is the standout', toneClass: 'border-emerald-400/30 bg-emerald-400/12 text-emerald-100' },
  { label: 'Some value friction', toneClass: 'border-amber-400/30 bg-amber-400/14 text-amber-100' },
] as const;

function ramp(progress: number, start: number, end: number): number {
  if (progress <= start) return 0;
  if (progress >= end) return 1;
  return (progress - start) / (end - start);
}

const INVALID_ASIN_MESSAGE = 'Invalid ASIN - must be 10 uppercase letters/digits (e.g. B000E7L2R4).';

type HomeClientProps = {
  initialAsin?: string;
};

export default function HomeClient({ initialAsin }: HomeClientProps) {
  const normalizedInitialAsin = initialAsin ? normalizeAsinInput(initialAsin) : '';
  const initialAsinError =
    normalizedInitialAsin && !isValidAsin(normalizedInitialAsin) ? INVALID_ASIN_MESSAGE : null;
  const [asinInput, setAsinInput] = useState(() =>
    normalizedInitialAsin
  );
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [retryCountdown, setRetryCountdown] = useState<number | null>(null);
  const [reviews, setReviews] = useState<ReviewScore[] | null>(null);
  const [resultAsin, setResultAsin] = useState<string | null>(null);
  const [productTitle, setProductTitle] = useState<string | undefined>(undefined);
  const [aspects, setAspects] = useState<AspectScore[] | undefined>(undefined);
  const [shareStatus, setShareStatus] = useState<string | null>(null);
  const [shareError, setShareError] = useState<string | null>(null);
  const [heroProgress, setHeroProgress] = useState(0);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const storySectionRef = useRef<HTMLElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const blob1Ref = useRef<HTMLDivElement>(null);
  const blob2Ref = useRef<HTMLDivElement>(null);
  const idleFloatRef = useRef<HTMLDivElement>(null);
  const idleRotateRef = useRef<HTMLDivElement>(null);
  const productCardInnerRef = useRef<HTMLDivElement>(null);
  const formPanelRef = useRef<HTMLDivElement>(null);
  const chipRefs = useRef<Array<HTMLSpanElement | null>>([]);
  const openFlourishPlayedRef = useRef(false);
  const lastInitialAsinRef = useRef<string | null>(null);
  const beginAnalysisRef = useRef<(asin: string) => void>(() => {});

  const heroItem = GALLERY_ITEMS[0];
  const isIdleMotionActive = !prefersReducedMotion && heroProgress <= IDLE_BOX_ANIMATION_END;

  beginAnalysisRef.current = beginAnalysis;

  // Warmup on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && !sessionStorage.getItem('warmup-done')) {
      sessionStorage.setItem('warmup-done', '1');
      fetch('/api/warmup').catch(() => {});
    }
  }, []);

  useEffect(() => {
    if (!normalizedInitialAsin || lastInitialAsinRef.current === normalizedInitialAsin) return;

    lastInitialAsinRef.current = normalizedInitialAsin;
    setAsinInput(normalizedInitialAsin);

    if (!isValidAsin(normalizedInitialAsin)) {
      setAnalyzeError(INVALID_ASIN_MESSAGE);
      return;
    }

    beginAnalysisRef.current(normalizedInitialAsin);
  }, [normalizedInitialAsin]);

  useEffect(() => {
    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const reducedQuery = window.matchMedia(REDUCED_MOTION_QUERY);
    const mobileQuery = window.matchMedia(MOBILE_QUERY);

    const syncMedia = () => {
      setPrefersReducedMotion(reducedQuery.matches);
      setIsMobile(mobileQuery.matches);
    };

    syncMedia();

    const bind = (query: MediaQueryList, listener: () => void) => {
      if (typeof query.addEventListener === 'function') {
        query.addEventListener('change', listener);
        return () => query.removeEventListener('change', listener);
      }

      query.addListener(listener);
      return () => query.removeListener(listener);
    };

    const cleanupReduced = bind(reducedQuery, syncMedia);
    const cleanupMobile = bind(mobileQuery, syncMedia);

    return () => {
      cleanupReduced();
      cleanupMobile();
    };
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined' || prefersReducedMotion) {
      setHeroProgress(0);
      return;
    }

    let frameId = 0;

    const updateProgress = () => {
      frameId = 0;

      if (!storySectionRef.current) return;

      const rect = storySectionRef.current.getBoundingClientRect();
      const scrollableDistance = Math.max(rect.height - window.innerHeight, 1);
      const nextProgress = clamp01(-rect.top / scrollableDistance);

      setHeroProgress(previous =>
        Math.abs(previous - nextProgress) < SCROLL_PROGRESS_EPSILON ? previous : nextProgress
      );
    };

    const requestProgressUpdate = () => {
      if (frameId !== 0) return;
      frameId = window.requestAnimationFrame(updateProgress);
    };

    requestProgressUpdate();
    window.addEventListener('scroll', requestProgressUpdate, { passive: true });
    window.addEventListener('resize', requestProgressUpdate);

    return () => {
      if (frameId !== 0) {
        window.cancelAnimationFrame(frameId);
      }
      window.removeEventListener('scroll', requestProgressUpdate);
      window.removeEventListener('resize', requestProgressUpdate);
    };
  }, [prefersReducedMotion]);

  useEffect(() => {
    if (prefersReducedMotion) return;

    let cleanedUp = false;
    let cleanup = () => {};

    import('animejs').then(({ default: anime }) => {
      if (cleanedUp) return;

      const targets = [blob1Ref.current, blob2Ref.current].filter(Boolean);

      const animations: Array<{ pause: () => void }> = [];

      if (blob1Ref.current) {
        animations.push(
          anime({
            targets: blob1Ref.current,
            translateX: ['-12px', '16px'],
            translateY: ['-16px', '14px'],
            duration: 8000,
            direction: 'alternate',
            loop: true,
            easing: 'easeInOutSine',
          })
        );
      }

      if (blob2Ref.current) {
        animations.push(
          anime({
            targets: blob2Ref.current,
            translateX: ['10px', '-14px'],
            translateY: ['14px', '-22px'],
            duration: 10000,
            direction: 'alternate',
            loop: true,
            easing: 'easeInOutSine',
          })
        );
      }

      cleanup = () => {
        animations.forEach(animation => animation.pause());
        anime.remove(targets);
      };
    });

    return () => {
      cleanedUp = true;
      cleanup();
    };
  }, [prefersReducedMotion]);

  useEffect(() => {
    if (!isIdleMotionActive) {
      if (typeof window !== 'undefined') {
        import('animejs').then(({ default: anime }) => {
          anime.remove([idleFloatRef.current, idleRotateRef.current].filter(Boolean));
          if (idleFloatRef.current) {
            idleFloatRef.current.style.transform = '';
          }
          if (idleRotateRef.current) {
            idleRotateRef.current.style.transform = '';
          }
        });
      }
      return;
    }

    let cleanedUp = false;
    let cleanup = () => {};

    import('animejs').then(({ default: anime }) => {
      if (cleanedUp) return;

      const targets = [idleFloatRef.current, idleRotateRef.current].filter(Boolean);
      const animations: Array<{ pause: () => void }> = [];

      if (idleFloatRef.current) {
        animations.push(
          anime({
            targets: idleFloatRef.current,
            translateY: ['-10px', '10px'],
            duration: 2800,
            direction: 'alternate',
            loop: true,
            easing: 'easeInOutSine',
          })
        );
      }

      if (idleRotateRef.current) {
        animations.push(
          anime({
            targets: idleRotateRef.current,
            rotateY: ['-8deg', '8deg'],
            rotateZ: ['-1.2deg', '1.2deg'],
            duration: 5400,
            direction: 'alternate',
            loop: true,
            easing: 'easeInOutSine',
          })
        );
      }

      cleanup = () => {
        animations.forEach(animation => animation.pause());
        anime.remove(targets);
        if (idleFloatRef.current) {
          idleFloatRef.current.style.transform = '';
        }
        if (idleRotateRef.current) {
          idleRotateRef.current.style.transform = '';
        }
      };
    });

    return () => {
      cleanedUp = true;
      cleanup();
    };
  }, [isIdleMotionActive]);

  useEffect(() => {
    if (prefersReducedMotion || openFlourishPlayedRef.current || heroProgress < STAGE_OPEN_END) {
      return;
    }

    let cancelled = false;

    import('animejs').then(({ default: anime }) => {
      if (cancelled) return;

      openFlourishPlayedRef.current = true;

      const chipTargets = chipRefs.current.filter(Boolean);
      const timeline = anime.timeline({ easing: 'easeOutExpo' });

      if (productCardInnerRef.current) {
        timeline.add({
          targets: productCardInnerRef.current,
          scale: [0.94, 1.04, 1],
          rotate: ['-2deg', '0deg'],
          duration: 480,
        });
      }

      if (chipTargets.length > 0) {
        timeline.add(
          {
            targets: chipTargets,
            translateY: ['8px', '0px'],
            rotate: ['-6deg', '0deg'],
            opacity: [0.7, 1],
            delay: anime.stagger(60),
            duration: 360,
          },
          '-=220'
        );
      }
    });

    return () => {
      cancelled = true;
    };
  }, [heroProgress, prefersReducedMotion]);

  useEffect(() => {
    if (!reviews || !resultsRef.current) return;

    if (prefersReducedMotion) {
      resultsRef.current.scrollIntoView({ behavior: 'auto', block: 'start' });
      return;
    }

    import('animejs').then(({ default: anime }) => {
      if (!resultsRef.current) return;

      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      anime({
        targets: resultsRef.current,
        translateY: [24, 0],
        opacity: [0, 1],
        duration: 500,
        easing: 'easeOutExpo',
      });
    });
  }, [prefersReducedMotion, reviews]);

  function startCountdown(seconds: number, onDone: () => void) {
    setRetryCountdown(seconds);
    let remaining = seconds;
    countdownRef.current = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        if (countdownRef.current) {
          clearInterval(countdownRef.current);
        }
        setRetryCountdown(null);
        onDone();
      } else {
        setRetryCountdown(remaining);
      }
    }, 1000);
  }

  async function doAnalyze(asin: string, attempt: number): Promise<void> {
    const res = await fetch(`/api/analyze?asin=${asin}`);
    const data = (await res.json()) as { error?: string } | AnalyzeApiResponse;

    if (res.status === 202 && attempt < MAX_RETRIES) {
      await new Promise<void>(resolve =>
        startCountdown(Math.round(RETRY_DELAY_MS / 1000), resolve)
      );
      return doAnalyze(asin, attempt + 1);
    }

    if (res.status === 202) {
      setAnalyzeError('Analysis still in progress - please try again in a moment.');
      return;
    }

    if (!res.ok) {
      setAnalyzeError((data as { error?: string }).error ?? `Request failed (${res.status})`);
      return;
    }

    const response = data as AnalyzeApiResponse;
    setReviews(response.reviews);
    setResultAsin(response.asin);
    setProductTitle(response.productTitle);
    setAspects(response.aspects);
  }

  function beginAnalysis(asin: string) {
    if (countdownRef.current) clearInterval(countdownRef.current);
    setRetryCountdown(null);
    setAnalyzeError(null);
    setShareStatus(null);
    setShareError(null);
    setReviews(null);
    setResultAsin(null);
    setProductTitle(undefined);
    setAspects(undefined);
    setAnalyzing(true);

    doAnalyze(asin, 1)
      .catch(() => setAnalyzeError('Network error - please try again.'))
      .finally(() => {
        setAnalyzing(false);
        setRetryCountdown(null);
        if (countdownRef.current) clearInterval(countdownRef.current);
      });
  }

  function handleAnalyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = normalizeAsinInput(asinInput);
    setAsinInput(trimmed);
    if (!isValidAsin(trimmed)) {
      setAnalyzeError(INVALID_ASIN_MESSAGE);
      return;
    }
    beginAnalysis(trimmed);
  }

  function handleClear() {
    if (countdownRef.current) clearInterval(countdownRef.current);
    setRetryCountdown(null);
    setReviews(null);
    setResultAsin(null);
    setAnalyzeError(null);
    setShareStatus(null);
    setShareError(null);
    setProductTitle(undefined);
    setAspects(undefined);
  }

  function handleGalleryClick(asin: string) {
    if (analyzing) return;
    setAsinInput(asin);
    beginAnalysis(asin);
  }

  function copyWithTextareaFallback(value: string): boolean {
    if (typeof document === 'undefined' || !document.body) return false;

    const textarea = document.createElement('textarea');
    textarea.value = value;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    textarea.style.top = '0';
    document.body.appendChild(textarea);
    textarea.select();
    textarea.setSelectionRange(0, value.length);

    try {
      return document.execCommand('copy');
    } catch {
      return false;
    } finally {
      document.body.removeChild(textarea);
    }
  }

  async function handleShare() {
    if (!resultAsin) return;

    const sharePath = sharePathForAsin(resultAsin);
    if (!sharePath) {
      setShareStatus(null);
      setShareError('Could not create share link for this ASIN.');
      return;
    }

    try {
      if (!navigator.clipboard?.writeText) {
        if (!copyWithTextareaFallback(sharePath)) {
          throw new Error('Clipboard API unavailable');
        }
      } else {
        try {
          await navigator.clipboard.writeText(sharePath);
        } catch {
          if (!copyWithTextareaFallback(sharePath)) {
            throw new Error('Clipboard fallback failed');
          }
        }
      }

      setShareError(null);
      setShareStatus(`Copied ${sharePath}`);
    } catch {
      setShareStatus(null);
      setShareError(`Copy failed. Share route: ${sharePath}`);
    }
  }

  const motionProgress = prefersReducedMotion ? 0.68 : heroProgress;
  const boxPose = prefersReducedMotion
    ? { rotateX: -10, rotateY: 22, scale: isMobile ? 1 : 1.08, translateX: 0, translateY: 0, opacity: 1 }
    : boxPoseAt(motionProgress);
  const boxSize = isMobile ? HERO_BOX_SIZE_MOBILE : HERO_BOX_SIZE_DESKTOP;
  const headlinePose = prefersReducedMotion
    ? { opacity: 1, translateY: 0 }
    : headlineStyleAt(motionProgress);
  const cardPose = prefersReducedMotion
    ? { opacity: 1, translateY: 0, scale: 1 }
    : cardEmergeAt(motionProgress);
  const flapProgress = prefersReducedMotion ? 1 : flapProgressAt(motionProgress);
  const boxTranslateX =
    boxPose.translateX *
    (isMobile ? BOX_TRANSLATE_X_FACTOR_MOBILE : BOX_TRANSLATE_X_FACTOR_DESKTOP);
  const boxTranslateY =
    boxPose.translateY *
    (isMobile ? BOX_TRANSLATE_Y_FACTOR_MOBILE : BOX_TRANSLATE_Y_FACTOR_DESKTOP);
  const dockBoxShiftProgress = prefersReducedMotion
    ? 0
    : ramp(motionProgress, DOCK_BOX_SHIFT_START, FORM_REVEAL_END);
  const dockBoxTranslateX =
    dockBoxShiftProgress *
    (isMobile ? DOCK_BOX_X_OFFSET_MOBILE : DOCK_BOX_X_OFFSET_DESKTOP);
  const dockBoxTranslateY =
    dockBoxShiftProgress *
    (isMobile ? DOCK_BOX_Y_OFFSET_MOBILE : DOCK_BOX_Y_OFFSET_DESKTOP);
  const formReveal = prefersReducedMotion ? 1 : ramp(motionProgress, FORM_REVEAL_START, FORM_REVEAL_END);
  const galleryReveal = prefersReducedMotion
    ? 1
    : ramp(motionProgress, GALLERY_REVEAL_START, GALLERY_REVEAL_END);
  const isHeroCardInteractive = prefersReducedMotion || cardPose.opacity >= HERO_CARD_INTERACTIVE_OPACITY_MIN;
  const isGalleryInteractive = prefersReducedMotion || galleryReveal >= GALLERY_INTERACTIVE_REVEAL_MIN;

  const boxShellStyle: CSSProperties = {
    opacity: boxPose.opacity,
    transform: [
      isMobile ? 'translateX(-50%)' : '',
      `translate3d(${boxTranslateX + dockBoxTranslateX}px, ${boxTranslateY + dockBoxTranslateY}px, 0)`,
      `scale(${boxPose.scale})`,
    ]
      .filter(Boolean)
      .join(' '),
    transformOrigin: isMobile ? 'top center' : 'top left',
  };

  const boxRotationStyle: CSSProperties = {
    transform: `rotateX(${boxPose.rotateX}deg) rotateY(${boxPose.rotateY}deg)`,
    transformStyle: 'preserve-3d',
  };

  const headlineStyle: CSSProperties = {
    opacity: headlinePose.opacity,
    transform: `translateY(${headlinePose.translateY}px)`,
  };

  const formPanelStyle: CSSProperties | undefined = prefersReducedMotion
    ? undefined
    : {
        opacity: formReveal,
        transform: `translateY(${(1 - formReveal) * 48}px) scale(${0.94 + formReveal * 0.06})`,
        pointerEvents: formReveal === 0 ? 'none' : 'auto',
        visibility: formReveal === 0 ? 'hidden' : 'visible',
      };

  return (
    <main className="relative min-h-screen overflow-x-clip bg-[#0F172A]">
      <div
        ref={blob1Ref}
        className="ambient-blob left-[-80px] top-[-100px] h-96 w-96 bg-green-500"
        aria-hidden="true"
      />
      <div
        ref={blob2Ref}
        className="ambient-blob bottom-[-60px] right-[-60px] h-80 w-80 bg-blue-600"
        aria-hidden="true"
      />

      <section
        ref={storySectionRef}
        className={prefersReducedMotion ? 'relative z-10' : `relative z-10 ${STORY_SCROLL_HEIGHT_CLASS}`}
      >
        <div
          className={
            prefersReducedMotion
              ? 'relative'
              : 'sticky top-0 overflow-hidden'
          }
        >
          <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-6 pb-12 pt-14 md:px-8 lg:px-10">
            <div className="grid flex-1 gap-12 lg:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.72fr)] lg:items-center">
              <div className="relative min-h-[560px] sm:min-h-[640px] lg:min-h-[720px]">
                <div className="absolute inset-0 overflow-visible">
                  <div
                    className="absolute top-3 sm:top-6 lg:top-28 xl:top-24"
                    style={{
                      left: isMobile ? '50%' : '1.25rem',
                    }}
                  >
                    <div className="pointer-events-none absolute inset-0 -z-10 rounded-full bg-green-500/10 blur-3xl" />
                    <div style={boxShellStyle}>
                      <div ref={idleFloatRef}>
                        <div ref={idleRotateRef}>
                          <div className="relative" style={{ width: boxSize, height: boxSize * 1.2, perspective: '1200px' }}>
                            <div style={boxRotationStyle}>
                              <CardboardBox flapProgress={flapProgress} size={boxSize} />
                            </div>

                            {heroItem.imageUrl ? (
                              <button
                                type="button"
                                onClick={() => handleGalleryClick(heroItem.asin)}
                                disabled={analyzing || !isHeroCardInteractive}
                                aria-label={`Analyze ${heroItem.shortName}`}
                                aria-hidden={!isHeroCardInteractive}
                                tabIndex={isHeroCardInteractive ? 0 : -1}
                                className="absolute left-1/2 top-[-5%] w-[68%] -translate-x-1/2 text-left transition-transform duration-150 focus:outline-none focus:ring-2 focus:ring-green-500/60 disabled:cursor-not-allowed disabled:opacity-60"
                                style={{
                                  opacity: cardPose.opacity,
                                  transform: `translate(-50%, -${cardPose.translateY}px) scale(${cardPose.scale})`,
                                  transformOrigin: 'bottom center',
                                  pointerEvents: isHeroCardInteractive ? 'auto' : 'none',
                                }}
                              >
                                <div
                                  ref={productCardInnerRef}
                                  className="overflow-hidden rounded-lg border border-slate-700 bg-slate-950/90 shadow-[0_18px_60px_rgba(15,23,42,0.45)]"
                                >
                                  <img
                                    src={heroItem.imageUrl}
                                    alt={heroItem.shortName}
                                    className="h-36 w-full object-contain bg-slate-900/80 p-3"
                                  />
                                  <div className="border-t border-slate-800 px-4 py-3">
                                    <p className="text-[10px] font-mono uppercase tracking-[0.24em] text-green-400">
                                      Cached hero sample
                                    </p>
                                    <p className="mt-2 text-sm font-semibold text-slate-100">
                                      {heroItem.shortName}
                                    </p>
                                    <p className="mt-1 text-xs font-mono text-slate-400">
                                      {heroItem.asin}
                                    </p>
                                  </div>
                                </div>
                              </button>
                            ) : null}

                            {HERO_SENTIMENT_CHIPS.map((chip, index) => {
                              const chipPose = prefersReducedMotion
                                ? { opacity: 1, translateX: 0, translateY: 0 }
                                : chipStyleAt(motionProgress, index);

                              return (
                                <span
                                  key={chip.label}
                                  ref={element => {
                                    chipRefs.current[index] = element;
                                  }}
                                  className={[
                                    'pointer-events-none absolute left-1/2 top-[22%] -translate-x-1/2 rounded-full border px-3 py-1.5 text-[11px] font-mono uppercase tracking-[0.18em] shadow-[0_10px_24px_rgba(15,23,42,0.26)]',
                                    chip.toneClass,
                                  ].join(' ')}
                                  style={{
                                    opacity: chipPose.opacity,
                                    transform: `translate(calc(-50% + ${chipPose.translateX}px), ${chipPose.translateY}px) rotate(${index === 1 ? '6deg' : index === 2 ? '-8deg' : '-3deg'})`,
                                  }}
                                >
                                  {chip.label}
                                </span>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="relative z-10 flex h-full flex-col justify-end pb-6 pt-[19rem] sm:pt-[24rem] lg:justify-center lg:pb-0 lg:pt-0">
                  <div className="max-w-lg lg:ml-[25rem] lg:mr-4 xl:ml-[27rem]" style={headlineStyle}>
                    <span className="inline-flex rounded-full border border-green-500/30 bg-green-500/10 px-3 py-1 text-xs font-mono uppercase tracking-[0.24em] text-green-300">
                      Scroll-driven review intelligence
                    </span>
                    <h1 className="mt-5 font-heading text-4xl font-bold leading-tight tracking-tight text-white sm:text-5xl lg:text-6xl">
                      Turn raw Amazon review noise into one readable verdict.
                    </h1>
                    <p className="mt-5 max-w-lg text-base leading-7 text-slate-300 sm:text-lg">
                      Follow the box as it tumbles open, pulls out a real product sample, and hands
                      you the same analysis flow that powers the results below.
                    </p>
                    {initialAsinError ? (
                      <p
                        role="alert"
                        className="mt-5 rounded-lg border border-red-500/35 bg-red-500/10 px-4 py-3 text-sm font-mono leading-6 text-red-300"
                      >
                        {initialAsinError}
                      </p>
                    ) : null}
                  </div>

                  <div className="mt-10 grid max-w-lg gap-3 lg:ml-[25rem] lg:mr-4 xl:ml-[27rem]">
                    {HERO_STEPS.map((step, index) => {
                      const stepOpacity = prefersReducedMotion ? 1 : storyStepOpacityAt(motionProgress, index);

                      return (
                        <div
                          key={step.title}
                          className="rounded-lg border border-slate-800/90 bg-slate-950/55 px-4 py-4 backdrop-blur-sm"
                          style={{
                            opacity: stepOpacity,
                            transform: prefersReducedMotion
                              ? 'none'
                              : `translateY(${(1 - stepOpacity) * 18}px)`,
                          }}
                        >
                          <p className="text-[10px] font-mono uppercase tracking-[0.24em] text-green-400">
                            {step.eyebrow}
                          </p>
                          <h2 className="mt-2 text-lg font-semibold text-slate-100">{step.title}</h2>
                          <p className="mt-2 text-sm leading-6 text-slate-400">{step.body}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div className="relative z-20 flex flex-col justify-end lg:min-h-[720px] lg:justify-center">
                <div
                  ref={formPanelRef}
                  style={formPanelStyle}
                  className="rounded-lg border border-slate-800 bg-slate-950/82 p-6 shadow-[0_20px_80px_rgba(2,6,23,0.32)] backdrop-blur"
                >
                  <div className="mb-6 flex items-start justify-between gap-4">
                    <div>
                      <p className="text-xs font-mono uppercase tracking-[0.24em] text-slate-400">
                        Analyze a live ASIN
                      </p>
                      <h2 className="mt-2 text-2xl font-semibold text-white">
                        Paste a product URL or drop straight to the ASIN.
                      </h2>
                      <p className="mt-3 max-w-md text-sm leading-6 text-slate-400">
                        The box docks here at the end of the scroll story, then the same workflow stays
                        usable for the gallery chips and any ASIN you want to test.
                      </p>
                    </div>
                    <div className="hidden min-w-[88px] rounded-lg border border-green-500/20 bg-green-500/10 px-3 py-2 text-right text-[11px] font-mono uppercase tracking-[0.18em] text-green-300 sm:block">
                      2-model
                      <br />
                      sentiment
                    </div>
                  </div>

                  <form onSubmit={handleAnalyze} aria-label="Analyze Amazon product">
                    <label
                      htmlFor="asin-input"
                      className="block text-xs font-mono uppercase tracking-[0.24em] text-slate-400"
                    >
                      Amazon ASIN
                    </label>
                    <div className="mt-3 flex flex-col gap-3 sm:flex-row">
                      <input
                        id="asin-input"
                        type="text"
                        value={asinInput}
                        onChange={event => {
                          setAsinInput(normalizeAsinInput(event.target.value));
                        }}
                        placeholder="e.g. B000E7L2R4 or an Amazon product URL"
                        disabled={analyzing}
                        autoComplete="off"
                        spellCheck={false}
                        className={[
                          'min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-900 px-4 py-3 text-sm font-mono text-slate-100 placeholder:text-slate-500',
                          'focus:outline-none focus:ring-2 focus:ring-green-500/60 focus:border-green-500/60',
                          'disabled:cursor-not-allowed disabled:opacity-50',
                        ].join(' ')}
                      />
                      <div className="flex gap-3">
                        <button
                          type="submit"
                          disabled={analyzing || asinInput.trim().length === 0}
                          className={[
                            'rounded-lg bg-green-500 px-6 py-3 text-sm font-semibold text-slate-950 transition-all duration-150',
                            'hover:bg-green-400 focus:outline-none focus:ring-2 focus:ring-green-500/60 focus:ring-offset-2 focus:ring-offset-slate-950',
                            'disabled:cursor-not-allowed disabled:opacity-40',
                            !analyzing && asinInput.trim().length > 0 ? 'glow-green' : '',
                          ].join(' ')}
                        >
                          {analyzing ? (
                            <span className="flex items-center gap-2">
                              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                              </svg>
                              Analyzing
                            </span>
                          ) : (
                            'Analyze'
                          )}
                        </button>
                        {reviews !== null && !analyzing ? (
                          <button
                            type="button"
                            onClick={handleClear}
                            className="rounded-lg border border-slate-700 px-4 py-3 text-sm font-mono text-slate-300 transition-colors hover:border-slate-500 hover:text-white focus:outline-none focus:ring-2 focus:ring-slate-500/60"
                          >
                            Clear
                          </button>
                        ) : null}
                      </div>
                    </div>

                    {analyzing ? (
                      <p className="mt-3 text-sm font-mono text-slate-400">
                        {retryCountdown !== null
                          ? `Retrying in ${retryCountdown}s while another analysis finishes.`
                          : 'Waking up the ML model - the first request can take 10 to 30 seconds.'}
                      </p>
                    ) : null}
                    {analyzeError ? (
                      <p role="alert" className="mt-3 text-sm font-mono text-red-400">
                        {analyzeError}
                      </p>
                    ) : null}
                  </form>

                  {reviews === null && !analyzing ? (
                    <div
                      className="mt-8"
                      aria-hidden={!isGalleryInteractive}
                      style={{
                        opacity: galleryReveal,
                        transform: prefersReducedMotion
                          ? 'none'
                          : `translateY(${(1 - galleryReveal) * 16}px)`,
                        pointerEvents: isGalleryInteractive ? 'auto' : 'none',
                      }}
                    >
                      <p className="text-xs font-mono uppercase tracking-[0.24em] text-slate-500">
                        Cached gallery
                      </p>
                      <div className="mt-4 flex flex-wrap gap-2">
                        {GALLERY_ITEMS.map(item => (
                          <button
                            key={item.asin}
                            type="button"
                            onClick={() => handleGalleryClick(item.asin)}
                            disabled={analyzing || !isGalleryInteractive}
                            tabIndex={isGalleryInteractive ? 0 : -1}
                            className="rounded-full border border-slate-700 bg-slate-900/70 px-4 py-2 text-xs font-mono text-slate-300 transition-all duration-150 hover:border-green-500/50 hover:text-green-300 focus:outline-none focus:ring-2 focus:ring-green-500/60"
                          >
                            <span aria-hidden="true">{item.emoji}</span> {item.shortName}
                          </button>
                        ))}
                      </div>
                      <p className="mt-3 text-[11px] font-mono text-slate-600">
                        Instant samples - no Canopy wait, no extra rate limit pressure.
                      </p>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {reviews !== null && reviews.length > 0 && !analyzing ? (
        <section className="relative z-10 mx-auto max-w-6xl px-6 pb-20 md:px-8 lg:px-10">
          <div
            ref={resultsRef}
            className={`results-panel ${prefersReducedMotion ? '' : 'opacity-0'}`}
          >
            <VerdictCard
              verdict={computeVerdict(reviews)}
              asin={resultAsin ?? ''}
              productTitle={productTitle}
            />
            {resultAsin ? (
              <div className="mt-6 rounded-lg border border-slate-800 bg-slate-950/70 px-4 py-4">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-xs font-mono uppercase tracking-[0.24em] text-slate-500">
                      Share this analysis
                    </p>
                    <p className="mt-1 text-sm font-mono text-slate-300">
                      {sharePathForAsin(resultAsin)}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleShare}
                    className="rounded-lg border border-green-500/40 bg-green-500/10 px-4 py-3 text-sm font-semibold text-green-200 transition-colors hover:border-green-400 hover:bg-green-500/15 focus:outline-none focus:ring-2 focus:ring-green-500/60"
                  >
                    Share
                  </button>
                </div>
                {shareStatus ? (
                  <p className="mt-3 text-sm font-mono text-green-300" role="status">
                    {shareStatus}
                  </p>
                ) : null}
                {shareError ? (
                  <p className="mt-3 text-sm font-mono text-red-400" role="alert">
                    {shareError}
                  </p>
                ) : null}
              </div>
            ) : null}
            {aspects && aspects.length > 0 ? <AspectBars aspects={aspects} /> : null}
            <div className="mt-8">
              <h2 className="mb-4 text-xs font-mono uppercase tracking-[0.24em] text-slate-400">
                Model deep-dive
              </h2>
              <SentimentPlot reviews={reviews} />
              <DisagreementPanel reviews={reviews} />
            </div>
            <HowItWorksStrip />
          </div>
        </section>
      ) : null}
    </main>
  );
}
