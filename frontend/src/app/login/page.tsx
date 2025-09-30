'use client'

import React, { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { apiClient } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { AlertCircle } from 'lucide-react'

export default function LoginPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = useState(true)
  const [email, setEmail] = useState('admin')
  const [password, setPassword] = useState('admin123')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  useEffect(() => {
    // Check if already authenticated, redirect to dashboard
    const checkAuth = async () => {
      if (apiClient.isAuthenticated()) {
        router.push('/')
        return
      }
      setIsLoading(false)
    }

    checkAuth()
  }, [router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    setError(null)

    try {
      await apiClient.login(email, password)
      router.push('/')
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Login failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-textSecondary">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-8">
      <div className="w-full max-w-md">
        {/* Attrahere Header */}
        <div className="text-center mb-12 section-elevated p-8">
          <h1 className="text-4xl font-bold text-sky-500 mb-4">Attrahere</h1>
          <p className="text-slate-300">The GitHub of ML Code Quality</p>
        </div>

        {/* Login Form Card */}
        <div className="section-elevated p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Error Message */}
            {error && (
              <div className="card-relief bg-red-900/30 border border-red-500/20 p-4 flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-red-500 flex-shrink-0" />
                <span className="text-sm text-red-500">{error}</span>
              </div>
            )}

            {/* Email Input */}
            <div className="space-y-2">
              <input
                type="text"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="Username"
                autoComplete="username"
                required
                className="w-full px-4 py-4 text-lg code-editor-relief bg-slate-800 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent transition-all"
              />
            </div>

            {/* Password Input */}
            <div className="space-y-2">
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Password"
                autoComplete="current-password"
                required
                className="w-full px-4 py-4 text-lg code-editor-relief bg-slate-800 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent transition-all"
              />
            </div>

            {/* Login Button */}
            <Button
              type="submit"
              disabled={isSubmitting}
              className="w-full bg-sky-500 hover:bg-sky-600 text-white py-4 text-lg font-semibold rounded-xl button-relief"
            >
              {isSubmitting ? 'Logging in...' : 'Log in'}
            </Button>
          </form>

          {/* Legal and Help Links */}
          <div className="text-center mt-6 space-y-3">
            <a
              href="#"
              className="text-slate-400 hover:text-sky-500 text-sm transition-colors block"
            >
              Forgot password?
            </a>
            <div className="flex justify-center gap-4 text-xs">
              <a
                href="/privacy-policy"
                className="text-slate-500 hover:text-sky-400 transition-colors"
              >
                Privacy Policy
              </a>
              <span className="text-slate-600">•</span>
              <a
                href="/terms-of-service"
                className="text-slate-500 hover:text-sky-400 transition-colors"
              >
                Terms of Service
              </a>
              <span className="text-slate-600">•</span>
              <a
                href="/gdpr-rights"
                className="text-slate-500 hover:text-sky-400 transition-colors"
              >
                GDPR Rights
              </a>
            </div>
          </div>
        </div>

        {/* Demo Credentials */}
        <div className="text-center mt-8">
          <div className="card-relief p-4 bg-green-900/20 border-green-500/20">
            <p className="text-green-500 font-medium text-sm mb-1">Demo Credentials</p>
            <p className="text-slate-300 text-sm font-mono">admin / admin123</p>
          </div>
        </div>
      </div>
    </div>
  )
}
