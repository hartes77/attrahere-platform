'use client'

import React, { useState, useEffect } from 'react'
import { apiClient, AuditLog } from '@/lib/api'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import {
  BarChart3,
  FileText,
  Settings,
  LogOut,
  Users,
  Shield,
  AlertCircle,
  User as UserIcon,
  Activity,
  Clock,
  Globe,
  Eye,
  RefreshCw,
} from 'lucide-react'

export default function AuditLogsPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [isAdmin, setIsAdmin] = useState(false)
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [loadingLogs, setLoadingLogs] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [limit, setLimit] = useState(100)

  useEffect(() => {
    const checkAuth = async () => {
      if (!apiClient.isAuthenticated()) {
        setIsAuthenticated(false)
        setIsLoading(false)
        return
      }

      try {
        const currentUser = await apiClient.getMe()
        setIsAuthenticated(true)
        setIsAdmin(currentUser.role === 'admin')

        if (currentUser.role === 'admin') {
          await loadAuditLogs()
        }
      } catch (error) {
        console.error('Auth check failed:', error)
        setIsAuthenticated(false)
      }
      setIsLoading(false)
    }

    checkAuth()
  }, [])

  const loadAuditLogs = async () => {
    setLoadingLogs(true)
    setError(null)
    try {
      const logsData = await apiClient.getAuditLogs(limit)
      setLogs(logsData)
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to load audit logs')
    }
    setLoadingLogs(false)
  }

  const handleLogout = async () => {
    await apiClient.logout()
    setIsAuthenticated(false)
  }

  const getActionColor = (action: string) => {
    if (action.includes('LOGIN')) return 'text-green-400'
    if (action.includes('LOGOUT')) return 'text-blue-400'
    if (action.includes('CREATED') || action.includes('STARTED')) return 'text-sky-400'
    if (action.includes('DELETED') || action.includes('FAILED')) return 'text-red-400'
    if (action.includes('FEEDBACK')) return 'text-yellow-400'
    return 'text-slate-400'
  }

  const getActionIcon = (action: string) => {
    if (action.includes('LOGIN')) return <UserIcon className="h-4 w-4 text-green-400" />
    if (action.includes('LOGOUT')) return <LogOut className="h-4 w-4 text-blue-400" />
    if (action.includes('USER_CREATED'))
      return <Users className="h-4 w-4 text-sky-400" />
    if (action.includes('USER_DELETED'))
      return <Users className="h-4 w-4 text-red-400" />
    if (action.includes('ANALYSIS'))
      return <Activity className="h-4 w-4 text-purple-400" />
    if (action.includes('FEEDBACK'))
      return <FileText className="h-4 w-4 text-yellow-400" />
    return <Eye className="h-4 w-4 text-slate-400" />
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString(),
      relative: getRelativeTime(date),
    }
  }

  const getRelativeTime = (date: Date) => {
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / (1000 * 60))
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  const formatDetails = (details: Record<string, any>) => {
    if (!details || Object.keys(details).length === 0) return null
    return JSON.stringify(details, null, 2)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-300">Loading...</p>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
        <div className="text-center max-w-2xl">
          <Shield className="h-24 w-24 text-sky-500 mx-auto mb-8" />
          <h1 className="text-4xl font-bold text-sky-500 mb-6">Access Required</h1>
          <p className="text-slate-300 text-lg mb-8">
            Please log in to access the audit logs
          </p>
          <Link href="/login">
            <Button className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-lg rounded-xl">
              Go to Login
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
        <div className="text-center max-w-2xl">
          <AlertCircle className="h-24 w-24 text-red-500 mx-auto mb-8" />
          <h1 className="text-4xl font-bold text-red-500 mb-6">Access Denied</h1>
          <p className="text-slate-300 text-lg mb-8">
            Only administrators can access audit logs
          </p>
          <Link href="/">
            <Button className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-lg rounded-xl">
              Back to Dashboard
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-slate-900 text-slate-100">
      {/* Sidebar */}
      <aside className="w-64 section-elevated p-8 flex flex-col">
        <div className="mb-10">
          <h1 className="text-2xl font-bold text-sky-500 mb-2">Attrahere</h1>
          <div className="w-12 h-1 bg-sky-500 rounded-full"></div>
        </div>

        <nav className="space-y-3 flex-1">
          <Link href="/" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <BarChart3 className="h-5 w-5" />
                <span>Dashboard</span>
              </div>
            </div>
          </Link>
          <Link href="/analyze" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <UserIcon className="h-5 w-5" />
                <span>Analyze</span>
              </div>
            </div>
          </Link>
          <Link href="/admin/users" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <Users className="h-5 w-5" />
                <span>Users</span>
              </div>
            </div>
          </Link>
          <div className="card-relief-strong text-sky-500 px-5 py-4 rounded-xl border-sky-500/30">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5" />
              <span className="font-semibold">Audit Logs</span>
            </div>
          </div>
          <Link href="/settings" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <Settings className="h-5 w-5" />
                <span>Settings</span>
              </div>
            </div>
          </Link>
        </nav>

        <Button
          onClick={handleLogout}
          className="mt-auto bg-red-600 hover:bg-red-700 text-white flex items-center gap-2 button-relief"
        >
          <LogOut className="h-4 w-4" />
          Logout
        </Button>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-10 space-y-10">
        {/* Header Section */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-slate-100 mb-2">Audit Logs</h1>
              <p className="text-slate-300 text-lg">Security and activity monitoring</p>
            </div>
            <div className="flex items-center gap-4">
              <select
                value={limit}
                onChange={e => setLimit(Number(e.target.value))}
                className="px-4 py-2 bg-slate-800 text-slate-100 rounded-lg border border-slate-600 focus:outline-none focus:ring-2 focus:ring-sky-500"
              >
                <option value={50}>Last 50</option>
                <option value={100}>Last 100</option>
                <option value={500}>Last 500</option>
              </select>
              <Button
                onClick={loadAuditLogs}
                disabled={loadingLogs}
                className="bg-sky-500 hover:bg-sky-600 text-white px-6 py-2 rounded-xl flex items-center gap-2 button-relief"
              >
                <RefreshCw className={`h-4 w-4 ${loadingLogs ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="section-elevated p-6">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-6 w-6 text-red-500 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-red-500 font-bold text-lg">Error</p>
                <p className="text-slate-300">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-500 hover:text-red-400 text-2xl font-bold px-3"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Audit Logs */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold text-slate-100 mb-2">
                Activity Log ({logs.length})
              </h2>
              <p className="text-slate-300">
                Real-time security and user activity monitoring
              </p>
            </div>
            {loadingLogs && (
              <div className="w-6 h-6 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
            )}
          </div>

          <div className="space-y-4">
            {logs.length === 0 ? (
              <div className="text-center py-12">
                <FileText className="h-16 w-16 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400 text-lg">No audit logs found</p>
                <p className="text-slate-500">
                  Activity will appear here as users interact with the platform
                </p>
              </div>
            ) : (
              logs.map(log => {
                const timestamp = formatTimestamp(log.timestamp)
                return (
                  <div
                    key={log.id}
                    className="card-relief p-6 hover:scale-101 transition-all duration-200"
                  >
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 mt-1">
                        {getActionIcon(log.action)}
                      </div>

                      <div className="flex-1">
                        <div className="flex items-start justify-between mb-2">
                          <div>
                            <span className="text-slate-100 font-medium">
                              {log.username}
                            </span>
                            <span className="text-slate-400 mx-2">•</span>
                            <span
                              className={`font-medium ${getActionColor(log.action)}`}
                            >
                              {log.action.replace('_', ' ')}
                            </span>
                          </div>
                          <div className="text-right text-sm">
                            <div className="text-slate-300">{timestamp.relative}</div>
                            <div className="text-slate-500 text-xs">
                              {timestamp.date} {timestamp.time}
                            </div>
                          </div>
                        </div>

                        <div className="text-slate-400 text-sm mb-2">
                          <span>Resource: </span>
                          <code className="text-slate-300 bg-slate-800 px-2 py-1 rounded text-xs">
                            {log.resource}
                          </code>
                        </div>

                        <div className="flex items-center gap-4 text-xs text-slate-500">
                          <div className="flex items-center gap-1">
                            <Globe className="h-3 w-3" />
                            <span>{log.ip_address}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            <span>ID: {log.id.slice(0, 8)}...</span>
                          </div>
                        </div>

                        {formatDetails(log.details) && (
                          <details className="mt-3">
                            <summary className="cursor-pointer text-slate-400 hover:text-slate-300 text-sm">
                              View Details
                            </summary>
                            <pre className="mt-2 p-3 bg-slate-800 rounded text-xs text-slate-300 overflow-x-auto">
                              {formatDetails(log.details)}
                            </pre>
                          </details>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
