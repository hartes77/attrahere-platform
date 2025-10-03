// lib/analytics-api.ts
/**
 * Analytics API Client for connecting to FastAPI backend
 * Handles authentication, data fetching, and error handling
 */

export interface AnalyticsConfig {
  baseUrl: string
  apiKey: string
}

export interface UsageMetrics {
  period_days: number
  total_analyses: number
  total_users: number
  total_patterns: number
  avg_quality_score: number
  avg_processing_time_ms: number
  daily_trend: Array<{
    date: string
    analyses: number
    active_users: number
    patterns_found: number
    avg_quality: number
  }>
}

export interface PatternAnalytics {
  period_days: number
  top_patterns: Array<{
    pattern_type: string
    total_occurrences: number
    avg_confidence: number
    severity_distribution: Record<string, number>
    users_affected: number
  }>
}

export interface QualityTrends {
  period_days: number
  quality_trends: Array<{
    date: string
    avg_quality: number
    avg_complexity: number
    patterns_per_analysis: number
  }>
}

export interface PerformanceMetrics {
  period_days: number
  avg_response_time_ms: number
  p95_response_time_ms: number
  error_rate_percent: number
  total_requests: number
  peak_hour: number
}

export interface UserStats {
  user_id: string
  email: string
  name: string
  organization: string
  total_analyses: number
  current_month_analyses: number
  max_monthly_analyses: number
  remaining_analyses: number
}

export interface UserLeaderboard {
  period_days: number
  leaderboard: Array<{
    name: string
    email: string
    total_analyses: number
    patterns_found: number
    avg_quality_score: number
  }>
}

export class AnalyticsAPIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: any
  ) {
    super(message)
    this.name = 'AnalyticsAPIError'
  }
}

export class AnalyticsAPIClient {
  private config: AnalyticsConfig

  constructor(config: AnalyticsConfig) {
    this.config = config
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`
    
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.config.apiKey}`,
      ...options.headers,
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => null)
        throw new AnalyticsAPIError(
          errorData?.detail || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          errorData
        )
      }

      return await response.json()
    } catch (error) {
      if (error instanceof AnalyticsAPIError) {
        throw error
      }
      
      // Network or parsing errors
      throw new AnalyticsAPIError(
        `Failed to connect to analytics API: ${error instanceof Error ? error.message : 'Unknown error'}`
      )
    }
  }

  /**
   * Get usage analytics for the authenticated user's organization
   */
  async getUsageAnalytics(days: number = 30): Promise<UsageMetrics> {
    return this.request<UsageMetrics>(`/api/v1/analytics/usage?days=${days}`)
  }

  /**
   * Get pattern analytics showing top anti-patterns
   */
  async getPatternAnalytics(days: number = 30, limit: number = 10): Promise<PatternAnalytics> {
    return this.request<PatternAnalytics>(`/api/v1/analytics/patterns?days=${days}&limit=${limit}`)
  }

  /**
   * Get code quality trends over time
   */
  async getQualityTrends(days: number = 30): Promise<QualityTrends> {
    return this.request<QualityTrends>(`/api/v1/analytics/quality-trends?days=${days}`)
  }

  /**
   * Get performance analytics for the platform
   */
  async getPerformanceMetrics(days: number = 7): Promise<PerformanceMetrics> {
    return this.request<PerformanceMetrics>(`/api/v1/analytics/performance?days=${days}`)
  }

  /**
   * Get user leaderboard for the organization
   */
  async getUserLeaderboard(days: number = 30, limit: number = 10): Promise<UserLeaderboard> {
    return this.request<UserLeaderboard>(`/api/v1/analytics/leaderboard?days=${days}&limit=${limit}`)
  }

  /**
   * Get current user statistics
   */
  async getUserStats(): Promise<UserStats> {
    return this.request<UserStats>('/api/v1/users/me')
  }

  /**
   * Create a new user (admin endpoint)
   */
  async createUser(data: {
    email: string
    name: string
    organization_slug?: string
  }): Promise<{
    user_id: string
    email: string
    name: string
    api_key: string
    organization: string
  }> {
    return this.request('/api/v1/users', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  }

  /**
   * Test API connection and authentication
   */
  async healthCheck(): Promise<{ status: string }> {
    return this.request<{ status: string }>('/health')
  }
}

// Singleton instance for the app
let analyticsClient: AnalyticsAPIClient | null = null

/**
 * Get or create the analytics client instance
 */
export function getAnalyticsClient(): AnalyticsAPIClient {
  if (!analyticsClient) {
    const config: AnalyticsConfig = {
      baseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000',
      apiKey: process.env.NEXT_PUBLIC_API_KEY || 'default-api-key',
    }
    
    analyticsClient = new AnalyticsAPIClient(config)
  }
  
  return analyticsClient
}

// React Query hooks for easier integration
export const analyticsQueryKeys = {
  usage: (days: number) => ['analytics', 'usage', days] as const,
  patterns: (days: number, limit: number) => ['analytics', 'patterns', days, limit] as const,
  quality: (days: number) => ['analytics', 'quality', days] as const,
  performance: (days: number) => ['analytics', 'performance', days] as const,
  leaderboard: (days: number, limit: number) => ['analytics', 'leaderboard', days, limit] as const,
  userStats: () => ['analytics', 'user-stats'] as const,
}

// Utility functions
export function formatMetricValue(value: number, type: 'percentage' | 'duration' | 'count' = 'count'): string {
  switch (type) {
    case 'percentage':
      return `${(value * 100).toFixed(1)}%`
    case 'duration':
      return value < 1000 ? `${value.toFixed(0)}ms` : `${(value / 1000).toFixed(1)}s`
    case 'count':
      return new Intl.NumberFormat().format(Math.round(value))
    default:
      return value.toString()
  }
}

export function getPatternSeverityColor(severity: string): string {
  const colors = {
    low: 'text-green-600 bg-green-100',
    medium: 'text-yellow-600 bg-yellow-100', 
    high: 'text-orange-600 bg-orange-100',
    critical: 'text-red-600 bg-red-100',
  }
  return colors[severity as keyof typeof colors] || 'text-gray-600 bg-gray-100'
}

export function calculateTrend(data: Array<{ date: string; value: number }>): 'up' | 'down' | 'stable' {
  if (data.length < 2) return 'stable'
  
  const recent = data.slice(-7) // Last 7 data points
  const older = data.slice(-14, -7) // Previous 7 data points
  
  const recentAvg = recent.reduce((sum, item) => sum + item.value, 0) / recent.length
  const olderAvg = older.reduce((sum, item) => sum + item.value, 0) / older.length
  
  const change = (recentAvg - olderAvg) / olderAvg
  
  if (change > 0.05) return 'up'
  if (change < -0.05) return 'down'
  return 'stable'
}