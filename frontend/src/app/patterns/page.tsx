'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { apiClient } from '@/lib/api'
import {
  ArrowLeft,
  AlertTriangle,
  Info,
  Search,
  Filter,
  Download,
  BarChart3,
  FileText,
  Code2,
  Brain,
  Clock,
  CheckCircle,
  Users,
  Shield,
  Settings,
  LogOut,
} from 'lucide-react'

interface MLPattern {
  id: string
  name: string
  description: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  category: string
  detectedCount: number
  lastDetected: string
  examples: string[]
  recommendation: string
}

export default function PatternsPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [currentUser, setCurrentUser] = useState<any>(null)
  const [patterns, setPatterns] = useState<MLPattern[]>([])
  const [filteredPatterns, setFilteredPatterns] = useState<MLPattern[]>([])
  const [searchTerm, setSearchTerm] = useState('')
  const [severityFilter, setSeverityFilter] = useState<string>('all')

  // Mock data for patterns (in real app, fetch from API)
  const mockPatterns: MLPattern[] = [
    {
      id: '1',
      name: 'Data Leakage',
      description: 'Information from the future accidentally used to train the model',
      severity: 'critical',
      category: 'Data Processing',
      detectedCount: 8,
      lastDetected: '2 hours ago',
      examples: [
        'scaler.fit(X) before train_test_split()',
        'Using future data in features',
      ],
      recommendation:
        'Apply preprocessing only on training data, then transform test data',
    },
    {
      id: '2',
      name: 'Magic Numbers',
      description: 'Hard-coded numerical values without explanation',
      severity: 'medium',
      category: 'Code Quality',
      detectedCount: 15,
      lastDetected: '1 day ago',
      examples: ['threshold = 0.73625', 'epochs = 42'],
      recommendation:
        'Define constants with meaningful names and document their origin',
    },
    {
      id: '3',
      name: 'Missing Random Seed',
      description: 'Non-reproducible results due to missing random state',
      severity: 'high',
      category: 'Reproducibility',
      detectedCount: 6,
      lastDetected: '3 hours ago',
      examples: ['train_test_split() without random_state', 'model.fit() without seed'],
      recommendation: 'Set random_state parameter for reproducible experiments',
    },
    {
      id: '4',
      name: 'Class Imbalance Ignored',
      description: 'Model trained on imbalanced data without addressing the issue',
      severity: 'high',
      category: 'Data Quality',
      detectedCount: 4,
      lastDetected: '5 hours ago',
      examples: ['99% negative, 1% positive class', 'No class_weight parameter'],
      recommendation: 'Use class balancing techniques or appropriate metrics',
    },
    {
      id: '5',
      name: 'Overfitting Warning',
      description: 'Model shows signs of overfitting to training data',
      severity: 'medium',
      category: 'Model Performance',
      detectedCount: 12,
      lastDetected: '30 minutes ago',
      examples: ['Training acc: 99%, Validation acc: 65%', 'No regularization'],
      recommendation: 'Add regularization, reduce model complexity, or get more data',
    },
    {
      id: '6',
      name: 'Unsafe Deserialization',
      description: 'pickle.load() used without validation - security risk',
      severity: 'critical',
      category: 'Security',
      detectedCount: 2,
      lastDetected: '1 day ago',
      examples: ['pickle.load(file)', 'joblib.load() from untrusted source'],
      recommendation:
        'Use safe serialization formats like JSON or validate pickle sources',
    },
  ]

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
        setPatterns(mockPatterns)
        setFilteredPatterns(mockPatterns)
      } catch (error) {
        console.error('Auth check failed:', error)
        apiClient.logout()
        setIsAuthenticated(false)
        setCurrentUser(null)
      }
      setIsLoading(false)
    }
    checkAuth()
  }, [])

  useEffect(() => {
    let filtered = patterns

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(
        pattern =>
          pattern.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          pattern.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
          pattern.category.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    // Filter by severity
    if (severityFilter !== 'all') {
      filtered = filtered.filter(pattern => pattern.severity === severityFilter)
    }

    setFilteredPatterns(filtered)
  }, [searchTerm, severityFilter, patterns])

  const handleLogout = () => {
    apiClient.logout()
    setIsAuthenticated(false)
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500'
      case 'high':
        return 'bg-orange-500'
      case 'medium':
        return 'bg-yellow-500'
      case 'low':
        return 'bg-green-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="h-4 w-4 text-red-500" />
      case 'high':
        return <AlertTriangle className="h-4 w-4 text-orange-500" />
      case 'medium':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'low':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      default:
        return <Info className="h-4 w-4" />
    }
  }

  const exportPatterns = async () => {
    try {
      const csvContent =
        'data:text/csv;charset=utf-8,' +
        'Pattern,Severity,Category,Description,Detected Count,Last Detected\n' +
        filteredPatterns
          .map(
            p =>
              `"${p.name}","${p.severity}","${p.category}","${p.description}","${p.detectedCount}","${p.lastDetected}"`
          )
          .join('\n')

      const encodedUri = encodeURI(csvContent)
      const link = document.createElement('a')
      link.setAttribute('href', encodedUri)
      link.setAttribute('download', 'attrahere-patterns-report.csv')
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-300">Loading patterns...</p>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-12">
        <div className="text-center max-w-2xl">
          <FileText className="h-24 w-24 text-sky-500 mx-auto mb-8" />
          <h1 className="text-4xl font-bold text-sky-500 mb-4">Access Required</h1>
          <p className="text-slate-300 text-xl mb-8">
            Please log in to view ML patterns.
          </p>
          <Link href="/login">
            <Button className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-xl rounded-xl">
              Login
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

          {/* Analysis Section */}
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
                <Link href="/analyze-natural" className="block">
                  <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200 bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
                    <div className="flex items-center gap-3">
                      <Brain className="h-5 w-5 text-purple-400" />
                      <span className="text-purple-300">Natural Language Analysis</span>
                    </div>
                  </div>
                </Link>
              </>
            )}

          <div className="card-relief-strong text-sky-500 px-5 py-4 rounded-xl border-sky-500/30">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5" />
              <span className="font-semibold">Patterns</span>
            </div>
          </div>

          {/* Admin Section */}
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

      {/* Main Content */}
      <main className="flex-1 p-10 space-y-10">
        {/* Header Section */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-slate-100 mb-2">ML Patterns</h1>
              <p className="text-slate-300 text-lg">
                Detected anti-patterns and code quality issues
              </p>
            </div>
            <div className="text-right">
              <p className="text-slate-400 text-sm">Total Patterns</p>
              <p className="text-slate-100 text-3xl font-bold">
                {filteredPatterns.length}
              </p>
            </div>
          </div>
        </div>

        {/* Filters and Search */}
        <div className="section-elevated p-8">
          <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-4 w-4" />
              <Input
                placeholder="Search patterns..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="pl-10 bg-slate-800 border-slate-600 text-slate-100"
              />
            </div>
            <div className="flex gap-4">
              <select
                value={severityFilter}
                onChange={e => setSeverityFilter(e.target.value)}
                className="px-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-slate-100"
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
              <Button
                onClick={exportPatterns}
                className="bg-green-600 hover:bg-green-700 text-white flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Export CSV
              </Button>
            </div>
          </div>
        </div>

        {/* Patterns Grid */}
        <div className="grid gap-6">
          {filteredPatterns.map(pattern => (
            <div key={pattern.id} className="section-elevated p-8">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    {getSeverityIcon(pattern.severity)}
                    <h3 className="text-2xl font-bold text-slate-100">
                      {pattern.name}
                    </h3>
                    <span className="text-xs bg-slate-700 text-slate-300 px-2 py-1 rounded">
                      {pattern.category}
                    </span>
                  </div>
                  <p className="text-slate-300 mb-4">{pattern.description}</p>

                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="text-slate-100 font-semibold mb-2">Examples:</h4>
                      <ul className="text-slate-400 space-y-1">
                        {pattern.examples.map((example, idx) => (
                          <li
                            key={idx}
                            className="text-sm font-mono bg-slate-800 px-2 py-1 rounded"
                          >
                            {example}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4 className="text-slate-100 font-semibold mb-2">
                        Recommendation:
                      </h4>
                      <p className="text-slate-400 text-sm">{pattern.recommendation}</p>
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div
                    className={`w-4 h-4 rounded-full ${getSeverityColor(pattern.severity)} mb-2`}
                  ></div>
                  <p className="text-slate-400 text-sm capitalize">
                    {pattern.severity}
                  </p>
                  <p className="text-slate-300 font-semibold">
                    {pattern.detectedCount}x
                  </p>
                  <p className="text-slate-500 text-xs">{pattern.lastDetected}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredPatterns.length === 0 && (
          <div className="section-elevated p-8 text-center">
            <FileText className="h-16 w-16 text-slate-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-300 mb-2">
              No patterns found
            </h3>
            <p className="text-slate-400">
              Try adjusting your search or filter criteria.
            </p>
          </div>
        )}
      </main>
    </div>
  )
}
