import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'
import CookieConsentBanner from '@/components/CookieConsentBanner'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'Attrahere - ML Code Quality Platform',
  description:
    'Attrahere: AI-powered ML code analysis and quality assurance platform. The GitHub of ML Code Quality.',
  keywords:
    'Attrahere, ML, machine learning, code quality, AI, analysis, python, patterns, GitHub for ML',
  authors: [{ name: 'Attrahere Team' }],
  robots: 'index, follow',
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:bg-primary focus:text-background focus:px-4 focus:py-2 focus:rounded focus:shadow-lg"
        >
          Skip to main content
        </a>
        <main id="main-content">{children}</main>
        <CookieConsentBanner />
      </body>
    </html>
  )
}
