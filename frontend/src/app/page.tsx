'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api'
import {
  BarChart3,
  User,
  FileText,
  Settings,
  Rocket,
  LogOut,
  Code2,
  Activity,
  Users,
  Shield,
  Brain,
  FolderOpen,
} from 'lucide-react'
import AttrahereLogo from '@/components/AttrahereLogo'

export default function HomePage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [currentUser, setCurrentUser] = useState<any>(null)

  useEffect(() => {
    const checkAuth = async () => {
      if (!apiClient.isAuthenticated()) {
        setIsAuthenticated(false)
        setIsLoading(false)
        return
      }

      try {
        const user = await apiClient.getMe()
        setIsAuthenticated(true)
        setCurrentUser(user)
      } catch (error) {
        console.error('Auth check failed:', error)
        // Clear authentication state on error
        apiClient.logout()
        setIsAuthenticated(false)
        setCurrentUser(null)
      }
      setIsLoading(false)
    }
    checkAuth()
  }, [])

  const handleLogout = () => {
    apiClient.logout()
    setIsAuthenticated(false)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-300">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    // Redirect to landing page for non-authenticated users
    if (typeof window !== 'undefined') {
      window.location.href = '/landing'
    }
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-300">Redirecting to landing page...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-slate-900 text-slate-100">
      {/* Sidebar con effetti rilievo */}
      <aside className="w-64 section-elevated p-8 flex flex-col">
        <div className="mb-10">
          <h1 className="text-2xl font-bold text-sky-500 mb-2">Attrahere</h1>
          <div className="w-12 h-1 bg-sky-500 rounded-full"></div>
        </div>

        <nav className="space-y-3 flex-1">
          <div className="card-relief-strong text-sky-500 px-5 py-4 rounded-xl border-sky-500/30">
            <div className="flex items-center gap-3">
              <BarChart3 className="h-5 w-5" />
              <span className="font-semibold">Dashboard</span>
            </div>
          </div>

          {/* Analysis Section - Available to Admin and Researcher */}
          {currentUser &&
            (currentUser.role === 'admin' || currentUser.role === 'researcher') && (
              <>
                <div className="pt-4 pb-2">
                  <div className="text-slate-500 text-xs uppercase tracking-wide font-medium px-3">
                    Code Analysis
                  </div>
                </div>
                <Link href="/analyze" className="block">
                  <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
                    <div className="flex items-center gap-3">
                      <Code2 className="h-5 w-5" />
                      <span>Single File Analysis</span>
                    </div>
                  </div>
                </Link>
                <Link href="/analyze-project" className="block">
                  <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
                    <div className="flex items-center gap-3">
                      <FolderOpen className="h-5 w-5" />
                      <span>Project Analysis</span>
                    </div>
                  </div>
                </Link>
                <Link href="/analyze-natural" className="block">
                  <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200 bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
                    <div className="flex items-center gap-3">
                      <Brain className="h-5 w-5 text-purple-400" />
                      <span className="text-purple-300">Natural Language Analysis</span>
                      <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded-full">
                        🧠 NEW
                      </span>
                    </div>
                  </div>
                </Link>
              </>
            )}

          <Link href="/patterns" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <FileText className="h-5 w-5" />
                <span>Patterns</span>
              </div>
            </div>
          </Link>

          {/* Admin Only Section */}
          {currentUser?.role === 'admin' && (
            <>
              <div className="pt-4 pb-2">
                <div className="text-slate-500 text-xs uppercase tracking-wide font-medium px-3">
                  Admin Panel
                </div>
              </div>
              <Link href="/admin/users" className="block">
                <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
                  <div className="flex items-center gap-3">
                    <Users className="h-5 w-5" />
                    <span>User Management</span>
                  </div>
                </div>
              </Link>
              <Link href="/admin/audit" className="block">
                <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
                  <div className="flex items-center gap-3">
                    <Shield className="h-5 w-5" />
                    <span>Audit Logs</span>
                  </div>
                </div>
              </Link>
            </>
          )}

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

      {/* Main Content con sezioni in rilievo */}
      <main className="flex-1 p-10 space-y-10">
        {/* Header Section */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-slate-100 mb-2">Dashboard</h1>
              <p className="text-slate-300 text-lg">
                Monitor your ML code quality metrics
              </p>
            </div>
            <div className="text-right">
              <p className="text-slate-400 text-sm">Last updated</p>
              <p className="text-slate-300 font-medium">Just now</p>
            </div>
          </div>
        </div>

        {/* KPI Cards con effetti rilievo */}
        <div className="section-elevated p-8">
          <h2 className="text-slate-100 text-2xl font-bold mb-6">Analytics Overview</h2>
          <div className="grid grid-cols-3 gap-8">
            <div className="kpi-card">
              <p className="text-slate-400 text-sm mb-3 uppercase tracking-wide">
                Total Analyses
              </p>
              <p className="text-slate-100 text-5xl font-bold mb-2">123</p>
              <div className="flex items-center justify-center gap-2 text-green-500 text-sm">
                <span>↗</span>
                <span>+12% this month</span>
              </div>
            </div>
            <div className="kpi-card">
              <p className="text-slate-400 text-sm mb-3 uppercase tracking-wide">
                Success Rate
              </p>
              <p className="text-slate-100 text-5xl font-bold mb-2">87%</p>
              <div className="flex items-center justify-center gap-2 text-green-500 text-sm">
                <span>↗</span>
                <span>+5% this month</span>
              </div>
            </div>
            <div className="kpi-card">
              <p className="text-slate-400 text-sm mb-3 uppercase tracking-wide">
                Top Pattern
              </p>
              <p className="text-slate-100 text-3xl font-bold mb-2">Singleton</p>
              <div className="text-slate-300 text-sm">Most detected pattern</div>
            </div>
          </div>
        </div>

        {/* CTA Section in rilievo */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-slate-100 text-2xl font-bold mb-2">
                Ready to Analyze?
              </h2>
              <p className="text-slate-300 mb-3">
                Upload your Python ML code and detect anti-patterns instantly
              </p>
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-purple-400" />
                <span className="text-purple-300 text-sm font-medium">
                  NEW: Natural Language Analysis with Gemma AI
                </span>
                <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded-full">
                  🧠 AI
                </span>
              </div>
            </div>
            <div className="flex gap-4">
              <Link href="/analyze">
                <Button className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-lg rounded-xl flex items-center gap-3 font-bold button-relief">
                  <Code2 className="h-5 w-5" />
                  Analyze Code
                </Button>
              </Link>
              <Link href="/analyze-natural">
                <Button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-4 text-lg rounded-xl flex items-center gap-3 font-bold button-relief">
                  <Brain className="h-5 w-5" />
                  Ask Gemma AI
                </Button>
              </Link>
            </div>
          </div>
        </div>

        {/* Detected Patterns Section con rilievo */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-slate-100 text-3xl font-bold">
              Recent Pattern Detections
            </h2>
            <Link
              href="/patterns"
              className="text-sky-500 hover:text-sky-400 font-medium"
            >
              View all patterns →
            </Link>
          </div>

          <div className="grid gap-8 md:grid-cols-2">
            {/* Factory Method Pattern Card */}
            <div className="card-relief-strong p-8 hover:scale-105 transition-all duration-300">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-slate-100 text-2xl font-bold mb-2">
                    Factory Method
                  </h3>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <span className="text-slate-400 text-sm">Medium Priority</span>
                  </div>
                </div>
                <div className="text-slate-400 text-sm text-right">
                  <div>Detected 3 times</div>
                  <div>Today</div>
                </div>
              </div>
              <p className="text-slate-300 leading-relaxed">
                Create objects by passing them on to another object
              </p>
            </div>

            {/* Singleton Pattern Card */}
            <div className="card-relief-strong p-8 hover:scale-105 transition-all duration-300">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-slate-100 text-2xl font-bold mb-2">Singleton</h3>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span className="text-slate-400 text-sm">Low Priority</span>
                  </div>
                </div>
                <div className="text-slate-400 text-sm text-right">
                  <div>Detected 8 times</div>
                  <div>This week</div>
                </div>
              </div>
              <p className="text-slate-300 leading-relaxed">
                Restrict instantiation of a class to one instance
              </p>
            </div>
          </div>
        </div>

        {/* Footer with GDPR/Privacy Links */}
        <footer className="section-elevated p-8 mt-12 border-t border-slate-700/50">
          <div className="max-w-6xl mx-auto">
            <div className="flex flex-col md:flex-row justify-between items-center gap-4">
              <div className="text-slate-400 text-sm">
                © 2025 Attrahere - ML Code Quality Platform
              </div>
              <div className="flex items-center gap-6 text-sm">
                <a
                  href="/privacy-policy"
                  className="text-slate-500 hover:text-sky-400 transition-colors"
                >
                  Privacy Policy
                </a>
                <a
                  href="/terms-of-service"
                  className="text-slate-500 hover:text-sky-400 transition-colors"
                >
                  Terms of Service
                </a>
                <a
                  href="/gdpr-rights"
                  className="text-slate-500 hover:text-sky-400 transition-colors"
                >
                  GDPR Rights
                </a>
                <a
                  href="/settings"
                  className="text-slate-500 hover:text-sky-400 transition-colors"
                >
                  Data Settings
                </a>
              </div>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}
