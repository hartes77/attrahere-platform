'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Cookie, Settings, Shield, X } from 'lucide-react'

export default function CookieConsentBanner() {
  const [isVisible, setIsVisible] = useState(false)
  const [showDetails, setShowDetails] = useState(false)

  useEffect(() => {
    // Check if user has already given consent
    const consent = localStorage.getItem('ml_platform_cookie_consent')
    if (!consent) {
      // Show banner after a short delay
      setTimeout(() => setIsVisible(true), 1000)
    }
  }, [])

  const handleAcceptAll = () => {
    localStorage.setItem('ml_platform_cookie_consent', 'all')
    localStorage.setItem('ml_platform_consent_date', new Date().toISOString())
    setIsVisible(false)
  }

  const handleAcceptEssential = () => {
    localStorage.setItem('ml_platform_cookie_consent', 'essential')
    localStorage.setItem('ml_platform_consent_date', new Date().toISOString())
    setIsVisible(false)
  }

  const handleDecline = () => {
    localStorage.setItem('ml_platform_cookie_consent', 'declined')
    localStorage.setItem('ml_platform_consent_date', new Date().toISOString())
    setIsVisible(false)
  }

  if (!isVisible) return null

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 bg-slate-900/95 backdrop-blur-sm border-t border-slate-700">
      <div className="max-w-6xl mx-auto">
        <div className="card-relief p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-3 flex-1">
              <Cookie className="h-6 w-6 text-sky-500 mt-1 flex-shrink-0" />

              <div className="flex-1">
                <h3 className="text-lg font-semibold text-slate-100 mb-2">
                  Cookie & Privacy Consent
                </h3>

                <p className="text-slate-300 text-sm mb-3 leading-relaxed">
                  We use cookies and similar technologies to enhance your experience,
                  analyze usage patterns, and support our research. By continuing to use
                  this platform, you consent to our use of cookies in accordance with
                  our{' '}
                  <a
                    href="/privacy-policy"
                    className="text-sky-400 hover:text-sky-300 underline"
                  >
                    Privacy Policy
                  </a>
                  .
                </p>

                {!showDetails ? (
                  <Button
                    variant="link"
                    size="sm"
                    onClick={() => setShowDetails(true)}
                    className="p-0 h-auto text-sky-400 hover:text-sky-300"
                  >
                    <Settings className="h-3 w-3 mr-1" />
                    Cookie Settings & Details
                  </Button>
                ) : (
                  <div className="bg-slate-800/50 p-4 rounded-lg text-sm space-y-3 border border-slate-700">
                    <div>
                      <h4 className="font-semibold text-slate-100 mb-2">
                        Cookie Categories:
                      </h4>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-slate-300 font-medium">
                              Essential Cookies
                            </span>
                            <p className="text-slate-400 text-xs">
                              Required for basic functionality
                            </p>
                          </div>
                          <span className="text-green-400 text-xs font-semibold">
                            Always On
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-slate-300 font-medium">
                              Analytics Cookies
                            </span>
                            <p className="text-slate-400 text-xs">
                              Help us understand how you use the platform
                            </p>
                          </div>
                          <span className="text-slate-400 text-xs">Optional</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-slate-300 font-medium">
                              Research Cookies
                            </span>
                            <p className="text-slate-400 text-xs">
                              Support academic research (anonymized)
                            </p>
                          </div>
                          <span className="text-slate-400 text-xs">Optional</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 pt-2 border-t border-slate-600">
                      <Shield className="h-4 w-4 text-green-500" />
                      <span className="text-xs text-slate-300">
                        GDPR Compliant • No personal data sold • Withdraw consent
                        anytime
                      </span>
                    </div>

                    <Button
                      variant="link"
                      size="sm"
                      onClick={() => setShowDetails(false)}
                      className="p-0 h-auto text-sky-400 hover:text-sky-300"
                    >
                      Hide Details
                    </Button>
                  </div>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-2 flex-shrink-0">
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDecline}
                  className="text-xs border-slate-600 text-slate-400 hover:text-slate-300"
                >
                  Decline All
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleAcceptEssential}
                  className="text-xs border-slate-600 text-slate-300 hover:text-slate-100"
                >
                  Essential Only
                </Button>
                <Button
                  size="sm"
                  onClick={handleAcceptAll}
                  className="bg-sky-600 hover:bg-sky-700 text-xs text-white"
                >
                  Accept All
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsVisible(false)}
                  className="h-8 w-8 p-0 text-slate-400 hover:text-slate-300"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="text-center">
                <a
                  href="/gdpr-rights"
                  className="text-xs text-slate-500 hover:text-sky-400 transition-colors"
                >
                  Manage GDPR Rights
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
