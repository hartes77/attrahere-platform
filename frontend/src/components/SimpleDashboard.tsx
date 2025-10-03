'use client'

import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
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

function MetricCard({ 
  title, 
  value, 
  icon, 
  description 
}: {
  title: string
  value: string | number
  icon: React.ReactNode
  description?: string
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
      </CardContent>
    </Card>
  )
}

export default function SimpleDashboard() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Analytics Dashboard</h2>
          <p className="text-muted-foreground">
            ML Code Quality Platform Analytics
          </p>
        </div>
      </div>

      {/* Demo Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Analyses"
          value="1,234"
          icon={<Activity className="h-4 w-4 text-muted-foreground" />}
          description="Analyses performed this month"
        />
        <MetricCard
          title="Active Users"
          value="42"
          icon={<Users className="h-4 w-4 text-muted-foreground" />}
          description="Users active this month"
        />
        <MetricCard
          title="Patterns Detected"
          value="3,567"
          icon={<CheckCircle className="h-4 w-4 text-muted-foreground" />}
          description="Code issues identified"
        />
        <MetricCard
          title="Avg Response Time"
          value="245ms"
          icon={<Zap className="h-4 w-4 text-muted-foreground" />}
          description="API response performance"
        />
      </div>

      {/* Quality Metrics */}
      <div className="grid gap-4 md:grid-cols-2">
        <MetricCard
          title="Code Quality Score"
          value="87.3%"
          icon={<TrendingUp className="h-4 w-4 text-muted-foreground" />}
          description="Average across all analyses"
        />
        <MetricCard
          title="Processing Time"
          value="1.2s"
          icon={<Clock className="h-4 w-4 text-muted-foreground" />}
          description="Average analysis duration"
        />
      </div>

      {/* Demo Charts Section */}
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Most Common Anti-Patterns</CardTitle>
            <CardDescription>
              Top code quality issues detected
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { name: 'Magic Numbers', count: 245, severity: 'medium' },
                { name: 'God Classes', count: 189, severity: 'high' },
                { name: 'Long Parameter Lists', count: 156, severity: 'low' },
                { name: 'Dead Code', count: 123, severity: 'medium' },
                { name: 'Duplicate Code', count: 98, severity: 'high' }
              ].map((pattern, index) => (
                <div key={pattern.name} className="flex items-center justify-between py-3 border-b last:border-b-0">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="font-medium">{index + 1}.</span>
                      <span className="font-medium">{pattern.name}</span>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {pattern.count} occurrences
                    </div>
                  </div>
                  <Badge 
                    variant="outline" 
                    className={
                      pattern.severity === 'high' ? 'text-red-600 bg-red-100' :
                      pattern.severity === 'medium' ? 'text-yellow-600 bg-yellow-100' :
                      'text-green-600 bg-green-100'
                    }
                  >
                    {pattern.severity}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Performance Section */}
        <div className="grid gap-4 md:grid-cols-2">
          <MetricCard
            title="Error Rate"
            value="0.12%"
            icon={<AlertTriangle className="h-4 w-4 text-muted-foreground" />}
            description="API error percentage"
          />
          <MetricCard
            title="P95 Response Time"
            value="450ms"
            icon={<Clock className="h-4 w-4 text-muted-foreground" />}
            description="95th percentile latency"
          />
        </div>

        {/* Connection Status */}
        <Card>
          <CardHeader>
            <CardTitle>System Status</CardTitle>
            <CardDescription>
              Current system health and connectivity
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span>Frontend Server</span>
                <Badge className="bg-green-100 text-green-800">Online</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>Backend API</span>
                <Badge className="bg-green-100 text-green-800">Connected</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>Database</span>
                <Badge className="bg-yellow-100 text-yellow-800">Setup Required</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span>Analytics Pipeline</span>
                <Badge className="bg-blue-100 text-blue-800">Ready</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Integration Guide */}
        <Card>
          <CardHeader>
            <CardTitle>Next Steps</CardTitle>
            <CardDescription>
              Complete your analytics setup
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm">Frontend application running</span>
              </div>
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm">Backend API connected</span>
              </div>
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-sm">API key configured</span>
              </div>
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-yellow-500" />
                <span className="text-sm">PostgreSQL database setup</span>
              </div>
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-yellow-500" />
                <span className="text-sm">Live data connection</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}