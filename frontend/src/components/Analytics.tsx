'use client'

import { useEffect } from 'react'

// Types for analytics events
interface AnalyticsEvent {
  action: string
  category: string
  label?: string
  value?: number
  custom_parameters?: Record<string, any>
}

// Core analytics tracking function
export function trackEvent(event: AnalyticsEvent) {
  // Log for development
  if (process.env.NODE_ENV === 'development') {
    console.log('🔍 Analytics Event:', event)
  }

  // Google Analytics 4 (if available)
  if (typeof window !== 'undefined' && (window as any).gtag) {
    ;(window as any).gtag('event', event.action, {
      event_category: event.category,
      event_label: event.label,
      value: event.value,
      ...event.custom_parameters,
    })
  }

  // Custom analytics endpoint (for internal tracking)
  if (typeof window !== 'undefined') {
    fetch('/api/analytics', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...event,
        timestamp: new Date().toISOString(),
        url: window.location.href,
        user_agent: navigator.userAgent,
        referrer: document.referrer,
      }),
    }).catch(err => console.warn('Analytics tracking failed:', err))
  }
}

// Predefined conversion events
export const ConversionEvents = {
  DEMO_REQUEST_STARTED: (source: string) =>
    trackEvent({
      action: 'demo_request_started',
      category: 'conversion',
      label: source,
      custom_parameters: { conversion_step: 'form_opened' },
    }),

  DEMO_REQUEST_SUBMITTED: (formData: any) =>
    trackEvent({
      action: 'demo_request_submitted',
      category: 'conversion',
      label: 'beta_form',
      value: 1,
      custom_parameters: {
        conversion_step: 'form_submitted',
        company_size: formData.company_size,
        role: formData.role,
      },
    }),

  CTA_CLICKED: (buttonText: string, location: string) =>
    trackEvent({
      action: 'cta_clicked',
      category: 'engagement',
      label: `${buttonText}_${location}`,
      custom_parameters: { button_text: buttonText, page_section: location },
    }),

  SECTION_VIEWED: (sectionName: string) =>
    trackEvent({
      action: 'section_viewed',
      category: 'engagement',
      label: sectionName,
      custom_parameters: { content_type: 'landing_section' },
    }),

  SCROLL_MILESTONE: (percentage: number) =>
    trackEvent({
      action: 'scroll_milestone',
      category: 'engagement',
      label: `${percentage}%`,
      value: percentage,
      custom_parameters: { scroll_percentage: percentage },
    }),

  PAGE_PERFORMANCE: (metrics: any) =>
    trackEvent({
      action: 'page_performance',
      category: 'technical',
      label: 'core_web_vitals',
      custom_parameters: metrics,
    }),
}

// Hook for scroll tracking
export function useScrollTracking() {
  useEffect(() => {
    const milestones = [25, 50, 75, 90]
    const reached = new Set<number>()

    const handleScroll = () => {
      const scrollPercent = Math.round(
        (window.scrollY /
          (document.documentElement.scrollHeight - window.innerHeight)) *
          100
      )

      milestones.forEach(milestone => {
        if (scrollPercent >= milestone && !reached.has(milestone)) {
          reached.add(milestone)
          ConversionEvents.SCROLL_MILESTONE(milestone)
        }
      })
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])
}

// Hook for section visibility tracking
export function useSectionTracking(sectionName: string, threshold = 0.5) {
  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            ConversionEvents.SECTION_VIEWED(sectionName)
          }
        })
      },
      { threshold }
    )

    const element = document.getElementById(
      sectionName.toLowerCase().replace(/\s+/g, '-')
    )
    if (element) {
      observer.observe(element)
    }

    return () => observer.disconnect()
  }, [sectionName, threshold])
}

// Performance monitoring
export function usePerformanceTracking() {
  useEffect(() => {
    if (typeof window === 'undefined') return

    // Core Web Vitals tracking
    const trackWebVitals = () => {
      // FCP, LCP, FID, CLS tracking would go here
      // For now, basic performance timing
      const perfData = window.performance.getEntriesByType('navigation')[0] as any

      if (perfData) {
        ConversionEvents.PAGE_PERFORMANCE({
          load_time: Math.round(perfData.loadEventEnd - perfData.fetchStart),
          dom_content_loaded: Math.round(
            perfData.domContentLoadedEventEnd - perfData.fetchStart
          ),
          first_paint: Math.round(perfData.responseEnd - perfData.fetchStart),
          page_size: new Blob([document.documentElement.outerHTML]).size,
        })
      }
    }

    // Track after page load
    if (document.readyState === 'complete') {
      setTimeout(trackWebVitals, 1000)
    } else {
      window.addEventListener('load', () => setTimeout(trackWebVitals, 1000))
    }
  }, [])
}

// Main Analytics Provider Component
export default function Analytics() {
  useScrollTracking()
  usePerformanceTracking()

  useEffect(() => {
    // Track page view
    trackEvent({
      action: 'page_view',
      category: 'navigation',
      label: 'landing_page',
      custom_parameters: {
        page_title: document.title,
        page_path: window.location.pathname,
      },
    })
  }, [])

  return null // This component doesn't render anything
}
