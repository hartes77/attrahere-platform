'use client'

import React, { useEffect } from 'react'
import { useRouter } from 'next/navigation'

interface PrivateBetaGuardProps {
  children: React.ReactNode
}

export function PrivateBetaGuard({ children }: PrivateBetaGuardProps) {
  const router = useRouter()

  // Check if development bypass is active (ONLY in development)
  const isDevelopmentBypassActive = process.env.NEXT_PUBLIC_BETA_GUARD_BYPASS === 'true'

  // TODO: Add proper authentication logic here
  // const isUserAuthenticated = checkUserAuthentication()

  // If development bypass is active, show the page content
  if (isDevelopmentBypassActive) {
    return <>{children}</>
  }

  useEffect(() => {
    // Redirect all protected pages to landing during private beta
    router.push('/')
  }, [router])

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center">
      <div className="text-center">
        <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-slate-300">Redirecting to Private Beta...</p>
      </div>
    </div>
  )
}
