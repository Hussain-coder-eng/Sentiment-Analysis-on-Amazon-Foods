import React from 'react';
import styles from './CardboardBox.module.css';

interface CardboardBoxProps {
  /** 0 = closed, 1 = top flaps fully open. Default 0. */
  flapProgress?: number;
  /** Box width in px. Height/depth scale proportionally. Default 220. */
  size?: number;
  className?: string;
}

const SMILE_SVG = (
  <svg viewBox="0 0 60 16" xmlns="http://www.w3.org/2000/svg">
    <path
      d="M4 4 Q30 18 52 6"
      stroke="#22C55E"
      strokeWidth="2.5"
      fill="none"
      strokeLinecap="round"
    />
    <path d="M48 3 L54 6 L47 9" fill="#22C55E" />
  </svg>
);

export default function CardboardBox({
  flapProgress = 0,
  size = 220,
  className = '',
}: CardboardBoxProps): React.ReactElement {
  const rootStyle: React.CSSProperties = {
    '--box-w': `${size}px`,
    '--flap': flapProgress,
  } as React.CSSProperties;

  return (
    <div
      className={`${styles.box} ${className}`}
      style={rootStyle}
      aria-hidden="true"
      role="presentation"
    >
      {/* Front face */}
      <div className={`${styles.face} ${styles.front}`}>
        <div className={styles.tape} />
        <div className={styles.smile}>{SMILE_SVG}</div>
      </div>

      {/* Back face */}
      <div className={`${styles.face} ${styles.back}`} />

      {/* Left face */}
      <div className={`${styles.face} ${styles.left}`} />

      {/* Right face */}
      <div className={`${styles.face} ${styles.right}`} />

      {/* Bottom face */}
      <div className={`${styles.face} ${styles.bottom}`} />

      {/* Front flap (top-front) */}
      <div className={`${styles.flap} ${styles['flap-front']}`}>
        <div className={`${styles['tape-flap']} ${styles['tape-flap-front']}`} />
      </div>

      {/* Back flap (top-back) */}
      <div className={`${styles.flap} ${styles['flap-back']}`}>
        <div className={`${styles['tape-flap']} ${styles['tape-flap-back']}`} />
      </div>
    </div>
  );
}
