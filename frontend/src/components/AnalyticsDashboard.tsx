'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { 
  BarChart, 
  LineChart, 
  PieChart, 
  TrendingUp, 
  TrendingDown, 
  Users, 
  Activity, 
  CheckCircle,
  AlertTriangle,
  Clock,
  Zap
} from 'lucide-react'
import { 
  useDashboardData, 
  useAnalyticsError
} from '@/lib/analytics-hooks'
import { 
  formatMetricValue,
  getPatternSeverityColor,
  calculateTrend
} from '@/lib/analytics-api'

interface MetricCardProps {
  title: string
  value: number | string
  change?: number
  format?: 'count' | 'percentage' | 'duration'
  icon: React.ReactNode
  description?: string
}

function MetricCard({ title, value, change, format = 'count', icon, description }: MetricCardProps) {
  const trendIcon = change && change > 0 ? <TrendingUp className="h-4 w-4 text-green-500" /> : 
                   change && change < 0 ? <TrendingDown className="h-4 w-4 text-red-500" /> : null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">
          {typeof value === 'number' ? formatMetricValue(value, format) : value}
        </div>
        {change !== undefined && (
          <div className="flex items-center space-x-1 text-xs text-muted-foreground">
            {trendIcon}
            <span>{change > 0 ? '+' : ''}{change.toFixed(1)}% from last period</span>
          </div>
        )}
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
      </CardContent>
    </Card>
  )
}

interface ErrorDisplayProps {
  error: any
}

function ErrorDisplay({ error }: ErrorDisplayProps) {
  const errorInfo = useAnalyticsError(error)
  
  if (!errorInfo) return null

  return (
    <Card className="border-red-200 bg-red-50">
      <CardHeader>
        <CardTitle className="text-red-800 flex items-center space-x-2">
          <AlertTriangle className="h-5 w-5" />
          <span>Unable to Load Analytics</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-red-700 mb-2">{errorInfo.message}</p>
        <p className="text-red-600 text-sm">{errorInfo.action}</p>
        {errorInfo.type === 'auth' && (
          <Button variant="outline" size="sm" className="mt-3">
            Update API Key
          </Button>
        )}
      </CardContent>
    </Card>
  )
}

interface LoadingSkeletonProps {
  count?: number
}

function LoadingSkeleton({ count = 4 }: LoadingSkeletonProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i}>
          <CardHeader className="space-y-0 pb-2">
            <div className="h-4 bg-gray-200 rounded animate-pulse" />
          </CardHeader>
          <CardContent>
            <div className="h-8 bg-gray-200 rounded animate-pulse mb-2" />
            <div className="h-3 bg-gray-200 rounded animate-pulse w-3/4" />
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

interface TimeRangeSelectorProps {
  value: number
  onChange: (days: number) => void
}

function TimeRangeSelector({ value, onChange }: TimeRangeSelectorProps) {
  const ranges = [
    { label: '7 days', value: 7 },
    { label: '30 days', value: 30 },
    { label: '90 days', value: 90 },
  ]

  return (
    <div className="flex space-x-2">
      {ranges.map((range) => (
        <Button
          key={range.value}
          variant={value === range.value ? 'default' : 'outline'}
          size="sm"
          onClick={() => onChange(range.value)}
        >
          {range.label}
        </Button>
      ))}
    </div>
  )
}

export default function AnalyticsDashboard() {
  const [timeRange, setTimeRange] = useState(30)
  const { usage, patterns, quality, performance, leaderboard, userStats, isLoading, isError, error } = 
    useDashboardData(timeRange)

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h2 className="text-3xl font-bold tracking-tight">Analytics Dashboard</h2>
          <LoadingSkeleton count={1} />
        </div>
        <LoadingSkeleton count={4} />
      </div>
    )
  }

  if (isError) {
    return (
      <div className="space-y-6">
        <h2 className="text-3xl font-bold tracking-tight">Analytics Dashboard</h2>
        <ErrorDisplay error={error} />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Analytics Dashboard</h2>
          <p className="text-muted-foreground">
            Comprehensive insights into your code quality platform usage
          </p>
        </div>
        <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
      </div>

      {/* User Stats Banner */}
      {userStats.data && (
        <Card className="bg-gradient-to-r from-blue-50 to-indigo-50">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg">{userStats.data.name}</h3>
                <p className="text-sm text-muted-foreground">{userStats.data.email}</p>
                <p className="text-sm text-muted-foreground">{userStats.data.organization}</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold">{userStats.data.current_month_analyses}</div>
                <div className="text-sm text-muted-foreground">
                  of {formatMetricValue(userStats.data.max_monthly_analyses)} analyses this month
                </div>
                <div className="text-xs text-green-600">
                  {userStats.data.remaining_analyses} remaining
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Analyses"
          value={usage.data?.total_analyses || 0}
          icon={<Activity className="h-4 w-4 text-muted-foreground" />}
          description={`Across ${usage.data?.period_days || timeRange} days`}
        />
        <MetricCard
          title="Active Users"
          value={usage.data?.total_users || 0}
          icon={<Users className="h-4 w-4 text-muted-foreground" />}
          description="Unique users in period"
        />
        <MetricCard
          title="Patterns Detected"
          value={usage.data?.total_patterns || 0}
          icon={<CheckCircle className="h-4 w-4 text-muted-foreground" />}
          description="Code issues identified"
        />
        <MetricCard
          title="Avg Response Time"
          value={performance.data?.avg_response_time_ms || 0}
          format="duration"
          icon={<Zap className="h-4 w-4 text-muted-foreground" />}
          description="API response performance"
        />
      </div>

      {/* Quality Metrics */}
      <div className="grid gap-4 md:grid-cols-2">
        <MetricCard
          title="Code Quality Score"
          value={usage.data?.avg_quality_score || 0}
          format="percentage"
          icon={<TrendingUp className="h-4 w-4 text-muted-foreground" />}
          description="Average across all analyses"
        />
        <MetricCard
          title="Processing Time"
          value={usage.data?.avg_processing_time_ms || 0}
          format="duration"
          icon={<Clock className="h-4 w-4 text-muted-foreground" />}
          description="Average analysis duration"
        />
      </div>

      {/* Top Patterns Section */}
      <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Most Common Anti-Patterns</CardTitle>
              <CardDescription>
                Top code quality issues detected in the last {timeRange} days
              </CardDescription>
            </CardHeader>
            <CardContent>
              {patterns.data?.top_patterns.map((pattern, index) => (
                <div key={pattern.pattern_type} className="flex items-center justify-between py-3 border-b last:border-b-0">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium">{index + 1}.</span>
                      <span className="font-medium">{pattern.pattern_type.replace(/_/g, ' ')}</span>
                    </div>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className="text-sm text-muted-foreground">
                        {formatMetricValue(pattern.total_occurrences)} occurrences
                      </span>
                      <span className="text-sm text-muted-foreground">•</span>
                      <span className="text-sm text-muted-foreground">
                        {pattern.users_affected} users affected
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="text-sm text-muted-foreground">
                      {formatMetricValue(pattern.avg_confidence, 'percentage')} confidence
                    </div>
                    {Object.entries(pattern.severity_distribution).map(([severity, count]) => (
                      <Badge key={severity} variant="outline" className={getPatternSeverityColor(severity)}>
                        {severity}: {count}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

        {/* Performance Section */}
        <div className="grid gap-4 md:grid-cols-2">
          <MetricCard
            title="Error Rate"
            value={performance.data?.error_rate_percent || 0}
            format="percentage"
            icon={<AlertTriangle className="h-4 w-4 text-muted-foreground" />}
            description="API error percentage"
          />
          <MetricCard
            title="P95 Response Time"
            value={performance.data?.p95_response_time_ms || 0}
            format="duration"
            icon={<Clock className="h-4 w-4 text-muted-foreground" />}
            description="95th percentile latency"
          />
        </div>
      </div>
    </div>
  )
}