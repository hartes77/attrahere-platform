// lib/analytics-hooks.ts
/**
 * React Query hooks for analytics data fetching
 * Provides optimized caching and error handling for analytics endpoints
 */

'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  getAnalyticsClient, 
  analyticsQueryKeys,
  UsageMetrics,
  PatternAnalytics,
  QualityTrends,
  PerformanceMetrics,
  UserLeaderboard,
  UserStats,
  AnalyticsAPIError
} from './analytics-api'

// Usage Analytics Hook
export function useUsageAnalytics(days: number = 30) {
  const client = getAnalyticsClient()
  
  return useQuery({
    queryKey: analyticsQueryKeys.usage(days),
    queryFn: () => client.getUsageAnalytics(days),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: (failureCount, error) => {
      if (error instanceof AnalyticsAPIError && error.status === 401) {
        return false // Don't retry auth errors
      }
      return failureCount < 3
    },
  })
}

// Pattern Analytics Hook
export function usePatternAnalytics(days: number = 30, limit: number = 10) {
  const client = getAnalyticsClient()
  
  return useQuery({
    queryKey: analyticsQueryKeys.patterns(days, limit),
    queryFn: () => client.getPatternAnalytics(days, limit),
    staleTime: 5 * 60 * 1000,
    retry: (failureCount, error) => {
      if (error instanceof AnalyticsAPIError && error.status === 401) {
        return false
      }
      return failureCount < 3
    },
  })
}

// Quality Trends Hook
export function useQualityTrends(days: number = 30) {
  const client = getAnalyticsClient()
  
  return useQuery({
    queryKey: analyticsQueryKeys.quality(days),
    queryFn: () => client.getQualityTrends(days),
    staleTime: 5 * 60 * 1000,
    retry: (failureCount, error) => {
      if (error instanceof AnalyticsAPIError && error.status === 401) {
        return false
      }
      return failureCount < 3
    },
  })
}

// Performance Metrics Hook
export function usePerformanceMetrics(days: number = 7) {
  const client = getAnalyticsClient()
  
  return useQuery({
    queryKey: analyticsQueryKeys.performance(days),
    queryFn: () => client.getPerformanceMetrics(days),
    staleTime: 2 * 60 * 1000, // 2 minutes for performance data
    retry: (failureCount, error) => {
      if (error instanceof AnalyticsAPIError && error.status === 401) {
        return false
      }
      return failureCount < 3
    },
  })
}

// User Leaderboard Hook
export function useUserLeaderboard(days: number = 30, limit: number = 10) {
  const client = getAnalyticsClient()
  
  return useQuery({
    queryKey: analyticsQueryKeys.leaderboard(days, limit),
    queryFn: () => client.getUserLeaderboard(days, limit),
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: (failureCount, error) => {
      if (error instanceof AnalyticsAPIError && error.status === 401) {
        return false
      }
      return failureCount < 3
    },
  })
}

// User Stats Hook
export function useUserStats() {
  const client = getAnalyticsClient()
  
  return useQuery({
    queryKey: analyticsQueryKeys.userStats(),
    queryFn: () => client.getUserStats(),
    staleTime: 1 * 60 * 1000, // 1 minute
    retry: (failureCount, error) => {
      if (error instanceof AnalyticsAPIError && error.status === 401) {
        return false
      }
      return failureCount < 3
    },
  })
}

// Create User Mutation Hook
export function useCreateUser() {
  const client = getAnalyticsClient()
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: {
      email: string
      name: string
      organization_slug?: string
    }) => client.createUser(data),
    onSuccess: () => {
      // Invalidate relevant queries to refresh data
      queryClient.invalidateQueries({ 
        queryKey: ['analytics', 'leaderboard'] 
      })
      queryClient.invalidateQueries({ 
        queryKey: ['analytics', 'usage'] 
      })
    },
  })
}

// Health Check Hook
export function useHealthCheck() {
  const client = getAnalyticsClient()
  
  return useQuery({
    queryKey: ['analytics', 'health'],
    queryFn: () => client.healthCheck(),
    staleTime: 30 * 1000, // 30 seconds
    retry: false, // Don't retry health checks
  })
}

// Combined Dashboard Data Hook
export function useDashboardData(days: number = 30) {
  const usage = useUsageAnalytics(days)
  const patterns = usePatternAnalytics(days)
  const quality = useQualityTrends(days)
  const performance = usePerformanceMetrics(7) // Always 7 days for performance
  const leaderboard = useUserLeaderboard(days)
  const userStats = useUserStats()

  return {
    usage,
    patterns,
    quality,
    performance,
    leaderboard,
    userStats,
    
    // Computed states
    isLoading: usage.isLoading || patterns.isLoading || quality.isLoading || 
               performance.isLoading || leaderboard.isLoading || userStats.isLoading,
    
    isError: usage.isError || patterns.isError || quality.isError || 
             performance.isError || leaderboard.isError || userStats.isError,
    
    error: usage.error || patterns.error || quality.error || 
           performance.error || leaderboard.error || userStats.error,
    
    isSuccess: usage.isSuccess && patterns.isSuccess && quality.isSuccess && 
               performance.isSuccess && leaderboard.isSuccess && userStats.isSuccess,
  }
}

// Polling hook for real-time updates
export function useRealTimeAnalytics(enabled: boolean = false, intervalMs: number = 30000) {
  const usage = useUsageAnalytics(1) // Last 24 hours
  const performance = usePerformanceMetrics(1)
  
  // Enable refetching at intervals when component is visible
  return useQuery({
    queryKey: ['analytics', 'realtime'],
    queryFn: async () => {
      const [usageData, perfData] = await Promise.all([
        usage.refetch(),
        performance.refetch(),
      ])
      
      return {
        usage: usageData.data,
        performance: perfData.data,
        timestamp: new Date().toISOString(),
      }
    },
    enabled,
    refetchInterval: intervalMs,
    refetchIntervalInBackground: false,
  })
}

// Custom hook for error handling
export function useAnalyticsError(error: any) {
  if (!error) return null
  
  if (error instanceof AnalyticsAPIError) {
    switch (error.status) {
      case 401:
        return {
          type: 'auth',
          message: 'Authentication failed. Please check your API key.',
          action: 'Login again or contact support.',
        }
      case 403:
        return {
          type: 'permission',
          message: 'Access denied. You may have exceeded your usage limits.',
          action: 'Upgrade your plan or contact support.',
        }
      case 429:
        return {
          type: 'rate_limit',
          message: 'Rate limit exceeded. Please wait before making more requests.',
          action: 'Try again in a few minutes.',
        }
      case 500:
        return {
          type: 'server',
          message: 'Server error. Our team has been notified.',
          action: 'Please try again later.',
        }
      default:
        return {
          type: 'api',
          message: error.message,
          action: 'Check your connection and try again.',
        }
    }
  }
  
  return {
    type: 'network',
    message: 'Network error. Please check your connection.',
    action: 'Verify your internet connection and try again.',
  }
}