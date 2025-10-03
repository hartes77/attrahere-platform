import { NextRequest, NextResponse } from 'next/server'

interface AnalyticsData {
  action: string
  category: string
  label?: string
  value?: number
  custom_parameters?: Record<string, any>
  timestamp: string
  url: string
  user_agent: string
  referrer: string
}

export async function POST(request: NextRequest) {
  try {
    const data: AnalyticsData = await request.json()

    // Basic validation
    if (!data.action || !data.category) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 })
    }

    // Get client IP and additional metadata
    const clientIP =
      request.headers.get('x-forwarded-for') ||
      request.headers.get('x-real-ip') ||
      'unknown'

    const enrichedData = {
      ...data,
      client_ip: clientIP,
      country: request.headers.get('cf-ipcountry') || 'unknown',
      user_id: generateUserHash(clientIP, data.user_agent),
      session_id: request.headers.get('x-session-id') || 'unknown',
    }

    // Log for development
    if (process.env.NODE_ENV === 'development') {
      console.log('📊 Analytics Event Received:', enrichedData)
    }

    // In production, you would send this to your analytics service
    // For now, we'll just log it
    await logAnalyticsEvent(enrichedData)

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Analytics API Error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}

// Simple hash function for user identification (privacy-friendly)
function generateUserHash(ip: string, userAgent: string): string {
  const data = `${ip}-${userAgent}`
  let hash = 0
  for (let i = 0; i < data.length; i++) {
    const char = data.charCodeAt(i)
    hash = (hash << 5) - hash + char
    hash = hash & hash // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36)
}

// Log analytics event (replace with your preferred analytics service)
async function logAnalyticsEvent(data: any) {
  // Here you would integrate with:
  // - Google Analytics 4
  // - Mixpanel
  // - Amplitude
  // - PostHog
  // - Custom database

  // For demo purposes, just console log in development
  if (process.env.NODE_ENV === 'development') {
    console.log('🎯 Analytics Event Processed:', {
      timestamp: data.timestamp,
      action: data.action,
      category: data.category,
      label: data.label,
      user_id: data.user_id,
      conversion_funnel: getConversionFunnelStage(data.action),
    })
  }

  // In production, store in your analytics backend
  // Example integrations:

  // Google Analytics 4
  /*
  if (process.env.GA_MEASUREMENT_ID) {
    await fetch(`https://www.google-analytics.com/mp/collect?measurement_id=${process.env.GA_MEASUREMENT_ID}&api_secret=${process.env.GA_API_SECRET}`, {
      method: 'POST',
      body: JSON.stringify({
        client_id: data.user_id,
        events: [{
          name: data.action,
          parameters: {
            event_category: data.category,
            event_label: data.label,
            value: data.value,
            ...data.custom_parameters
          }
        }]
      })
    })
  }
  */

  // PostHog
  /*
  if (process.env.POSTHOG_API_KEY) {
    await fetch('https://app.posthog.com/capture/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: process.env.POSTHOG_API_KEY,
        event: data.action,
        properties: {
          distinct_id: data.user_id,
          ...data.custom_parameters
        }
      })
    })
  }
  */
}

// Classify events into conversion funnel stages
function getConversionFunnelStage(action: string): string {
  const funnelStages = {
    page_view: 'awareness',
    section_viewed: 'interest',
    cta_clicked: 'consideration',
    demo_request_started: 'intent',
    demo_request_submitted: 'conversion',
    scroll_milestone: 'engagement',
  }

  return funnelStages[action as keyof typeof funnelStages] || 'other'
}

export async function GET() {
  return NextResponse.json({
    message: 'Analytics API is running',
    status: 'healthy',
  })
}
