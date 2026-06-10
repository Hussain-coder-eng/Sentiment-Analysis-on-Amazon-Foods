import type { Metadata } from 'next';
import { Exo, Roboto_Mono } from 'next/font/google';
import './globals.css';

const exo = Exo({
  subsets: ['latin'],
  weight: ['300', '400', '600', '700'],
  variable: '--font-heading',
  display: 'swap',
});

const robotoMono = Roboto_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Amazon Review Sentiment Analyzer',
  description: 'Analyze Amazon product reviews with VADER and RoBERTa ML sentiment models.',
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${exo.variable} ${robotoMono.variable} dark`}>
      <body className="antialiased font-mono bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
