'use client'

import React, { useState } from 'react'
import {
  ArrowLeft,
  Shield,
  Download,
  Trash2,
  Edit,
  Eye,
  UserCheck,
  AlertCircle,
  CheckCircle,
  Clock,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

export default function GDPRRightsPage() {
  const [activeRequest, setActiveRequest] = useState<string | null>(null)
  const [showConfirmation, setShowConfirmation] = useState(false)

  const gdprRights = [
    {
      id: 'access',
      title: 'Right of Access',
      article: 'Art. 15',
      description: 'Request a copy of all personal data we hold about you',
      icon: Eye,
      color: 'blue',
      actionText: 'Download My Data',
      estimatedTime: 'Immediate download',
    },
    {
      id: 'rectification',
      title: 'Right to Rectification',
      article: 'Art. 16',
      description: 'Correct inaccurate or incomplete personal data',
      icon: Edit,
      color: 'green',
      actionText: 'Update My Info',
      estimatedTime: 'Immediate update',
    },
    {
      id: 'erasure',
      title: 'Right to Erasure',
      article: 'Art. 17',
      description: 'Request deletion of your personal data ("Right to be Forgotten")',
      icon: Trash2,
      color: 'red',
      actionText: 'Delete My Data',
      estimatedTime: 'Within 30 days',
    },
    {
      id: 'portability',
      title: 'Right to Data Portability',
      article: 'Art. 20',
      description: 'Receive your personal data in machine-readable format',
      icon: Download,
      color: 'purple',
      actionText: 'Export My Data',
      estimatedTime: 'Within 24 hours',
    },
    {
      id: 'object',
      title: 'Right to Object',
      article: 'Art. 21',
      description: 'Object to processing based on legitimate interests',
      icon: AlertCircle,
      color: 'orange',
      actionText: 'Object to Processing',
      estimatedTime: 'Immediate effect',
    },
    {
      id: 'withdraw',
      title: 'Withdraw Consent',
      article: 'Art. 7(3)',
      description: 'Withdraw consent for research data collection at any time',
      icon: UserCheck,
      color: 'yellow',
      actionText: 'Manage Consent',
      estimatedTime: 'Immediate effect',
    },
  ]

  const handleRightRequest = (rightId: string) => {
    setActiveRequest(rightId)
    if (rightId === 'erasure') {
      setShowConfirmation(true)
    } else {
      // Simulate processing for other rights
      setTimeout(() => {
        setActiveRequest(null)
        alert(
          `${gdprRights.find(r => r.id === rightId)?.title} request submitted successfully!`
        )
      }, 1000)
    }
  }

  const confirmErasure = () => {
    setShowConfirmation(false)
    setActiveRequest(null)
    alert(
      'Data erasure request submitted. You will receive confirmation within 30 days.'
    )
  }

  const getColorClasses = (color: string) => {
    const colors = {
      blue: 'bg-blue-900/20 border-blue-500/20 text-blue-400',
      green: 'bg-green-900/20 border-green-500/20 text-green-400',
      red: 'bg-red-900/20 border-red-500/20 text-red-400',
      purple: 'bg-purple-900/20 border-purple-500/20 text-purple-400',
      orange: 'bg-orange-900/20 border-orange-500/20 text-orange-400',
      yellow: 'bg-yellow-900/20 border-yellow-500/20 text-yellow-400',
    }
    return colors[color as keyof typeof colors] || colors.blue
  }

  return (
    <div className="min-h-screen bg-background">
      <main className="container mx-auto px-6 py-12 max-w-6xl">
        {/* Header */}
        <div className="mb-8">
          <Button
            variant="outline"
            className="mb-6"
            onClick={() => window.history.back()}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>

          <div className="section-elevated p-8">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="w-8 h-8 text-sky-500" />
              <h1 className="text-3xl font-bold text-slate-100">GDPR Rights</h1>
            </div>
            <p className="text-slate-300 text-lg">
              Exercise your data protection rights under the General Data Protection
              Regulation
            </p>
            <p className="text-slate-400 text-sm mt-2">
              All requests are processed in accordance with GDPR requirements and
              timelines
            </p>
          </div>
        </div>

        {/* Quick Info */}
        <div className="card-relief p-6 mb-8">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center">
              <CheckCircle className="w-8 h-8 text-green-500 mx-auto mb-2" />
              <h3 className="text-slate-100 font-semibold">GDPR Compliant</h3>
              <p className="text-slate-400 text-sm">
                Fully compliant with EU data protection laws
              </p>
            </div>
            <div className="text-center">
              <Clock className="w-8 h-8 text-sky-500 mx-auto mb-2" />
              <h3 className="text-slate-100 font-semibold">Timely Processing</h3>
              <p className="text-slate-400 text-sm">
                All requests processed within legal timeframes
              </p>
            </div>
            <div className="text-center">
              <Shield className="w-8 h-8 text-purple-500 mx-auto mb-2" />
              <h3 className="text-slate-100 font-semibold">Secure & Private</h3>
              <p className="text-slate-400 text-sm">
                Your data is protected with enterprise security
              </p>
            </div>
          </div>
        </div>

        {/* GDPR Rights Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {gdprRights.map(right => {
            const Icon = right.icon
            const colorClasses = getColorClasses(right.color)
            const isProcessing = activeRequest === right.id

            return (
              <Card
                key={right.id}
                className="card-relief hover:scale-105 transition-all duration-300"
              >
                <CardHeader>
                  <div className={`p-3 rounded-lg ${colorClasses} w-fit mb-3`}>
                    <Icon className="w-6 h-6" />
                  </div>
                  <CardTitle className="text-slate-100 text-lg">
                    {right.title}
                  </CardTitle>
                  <CardDescription className="text-slate-400">
                    {right.article} • {right.estimatedTime}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-300 text-sm mb-4 leading-relaxed">
                    {right.description}
                  </p>
                  <Button
                    onClick={() => handleRightRequest(right.id)}
                    disabled={isProcessing}
                    className="w-full"
                    variant={right.id === 'erasure' ? 'destructive' : 'default'}
                  >
                    {isProcessing ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                        Processing...
                      </>
                    ) : (
                      <>
                        <Icon className="w-4 h-4 mr-2" />
                        {right.actionText}
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>

        {/* Data Categories Overview */}
        <div className="card-relief p-8 mb-8">
          <h2 className="text-2xl font-bold text-slate-100 mb-6">Data We Process</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-slate-100 mb-3">
                Account Data
              </h3>
              <ul className="text-slate-300 space-y-2 text-sm">
                <li>• Username and email address</li>
                <li>• Account settings and preferences</li>
                <li>• Login timestamps and activity</li>
                <li>• User role and permissions</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-100 mb-3">
                Analysis Data
              </h3>
              <ul className="text-slate-300 space-y-2 text-sm">
                <li>• Code analysis requests (anonymized)</li>
                <li>• Pattern detection results</li>
                <li>• User feedback and ratings</li>
                <li>• Usage analytics (opt-in)</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Legal Basis Information */}
        <div className="card-relief p-8 mb-8">
          <h2 className="text-2xl font-bold text-slate-100 mb-6">
            Legal Basis for Processing
          </h2>
          <div className="space-y-4">
            <div className="p-4 bg-sky-900/20 border border-sky-500/20 rounded-lg">
              <h3 className="text-sky-400 font-semibold mb-2">
                Contract Performance (Art. 6(1)(b))
              </h3>
              <p className="text-slate-300 text-sm">
                Account information and basic service functionality necessary to provide
                our service
              </p>
            </div>
            <div className="p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
              <h3 className="text-green-400 font-semibold mb-2">
                Consent (Art. 6(1)(a))
              </h3>
              <p className="text-slate-300 text-sm">
                Research data collection, analytics, and optional features - you can
                withdraw anytime
              </p>
            </div>
            <div className="p-4 bg-purple-900/20 border border-purple-500/20 rounded-lg">
              <h3 className="text-purple-400 font-semibold mb-2">
                Legitimate Interests (Art. 6(1)(f))
              </h3>
              <p className="text-slate-300 text-sm">
                Security monitoring, fraud prevention, and system performance
                optimization
              </p>
            </div>
          </div>
        </div>

        {/* Contact for Help */}
        <div className="card-relief p-8">
          <h2 className="text-2xl font-bold text-slate-100 mb-6">Need Help?</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <h3 className="text-slate-100 font-semibold mb-2">
                Data Protection Officer
              </h3>
              <p className="text-slate-300 text-sm mb-2">dpo@mlcodeplatform.com</p>
              <p className="text-slate-400 text-xs">For GDPR compliance questions</p>
            </div>
            <div className="text-center">
              <h3 className="text-slate-100 font-semibold mb-2">Privacy Team</h3>
              <p className="text-slate-300 text-sm mb-2">privacy@mlcodeplatform.com</p>
              <p className="text-slate-400 text-xs">For general privacy questions</p>
            </div>
            <div className="text-center">
              <h3 className="text-slate-100 font-semibold mb-2">Support Team</h3>
              <p className="text-slate-300 text-sm mb-2">support@mlcodeplatform.com</p>
              <p className="text-slate-400 text-xs">For technical assistance</p>
            </div>
          </div>
        </div>

        {/* Confirmation Modal */}
        {showConfirmation && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="card-relief-strong p-8 max-w-md mx-4">
              <div className="text-center">
                <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-slate-100 mb-4">
                  Confirm Data Erasure
                </h3>
                <p className="text-slate-300 mb-6">
                  This action will permanently delete all your personal data. This
                  cannot be undone. Are you sure you want to proceed?
                </p>
                <div className="flex gap-4">
                  <Button
                    variant="outline"
                    onClick={() => setShowConfirmation(false)}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={confirmErasure}
                    className="flex-1"
                  >
                    Confirm Deletion
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
