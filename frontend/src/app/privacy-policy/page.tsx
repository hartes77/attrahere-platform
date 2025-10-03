'use client'

import React from 'react'
import { ArrowLeft, Shield, Lock, Eye, Download, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function PrivacyPolicyPage() {
  return (
    <div className="min-h-screen bg-background">
      <main className="container mx-auto px-6 py-12 max-w-4xl">
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
              <h1 className="text-3xl font-bold text-slate-100">Privacy Policy</h1>
            </div>
            <p className="text-slate-300 text-lg">
              Attrahere - ML Code Quality Platform - GDPR Compliant Privacy Policy
            </p>
            <p className="text-slate-400 text-sm mt-2">
              Last Updated: September 13, 2025 | Effective Date: September 13, 2025
            </p>
          </div>
        </div>

        {/* Quick Links */}
        <div className="card-relief p-6 mb-8">
          <h2 className="text-slate-100 text-lg font-semibold mb-4">Quick Links</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <a
              href="#data-collection"
              className="text-sky-400 hover:text-sky-300 text-sm"
            >
              Data Collection
            </a>
            <a href="#gdpr-rights" className="text-sky-400 hover:text-sky-300 text-sm">
              GDPR Rights
            </a>
            <a
              href="#data-security"
              className="text-sky-400 hover:text-sky-300 text-sm"
            >
              Data Security
            </a>
            <a href="#contact" className="text-sky-400 hover:text-sky-300 text-sm">
              Contact Us
            </a>
          </div>
        </div>

        {/* Main Content */}
        <div className="space-y-8">
          {/* Introduction */}
          <div className="card-relief p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-4">Introduction</h2>
            <p className="text-slate-300 leading-relaxed mb-4">
              This Privacy Policy describes how Attrahere - ML Code Quality Platform
              ("we," "our," or "us") collects, uses, and shares your personal
              information when you use our machine learning code analysis platform (the
              "Service").
            </p>
            <p className="text-slate-300 leading-relaxed">
              We are committed to protecting your privacy and ensuring GDPR compliance
              for all users, including those in the European Union.
            </p>
            <div className="mt-6 p-4 bg-sky-900/20 border border-sky-500/20 rounded-lg">
              <p className="text-sky-300 font-semibold">Data Controller</p>
              <p className="text-slate-300">Attrahere - ML Code Quality Platform</p>
              <p className="text-slate-400 text-sm">
                Contact: privacy@attrahere.com | DPO: dpo@attrahere.com
              </p>
            </div>
          </div>

          {/* Data Collection */}
          <div className="card-relief p-8" id="data-collection">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Information We Collect
            </h2>

            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-3 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-sky-500" />
                  Account Information
                </h3>
                <ul className="text-slate-300 space-y-2 ml-7">
                  <li>• Username and email address for account creation</li>
                  <li>• Password hash (never stored in plain text)</li>
                  <li>• User role (admin, researcher, developer)</li>
                  <li>• Account creation timestamp and last login</li>
                </ul>
                <p className="text-slate-400 text-sm mt-2">
                  <strong>Legal Basis:</strong> Contract performance (GDPR Art. 6(1)(b))
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-3 flex items-center gap-2">
                  <Lock className="w-5 h-5 text-sky-500" />
                  Analysis Data
                </h3>
                <ul className="text-slate-300 space-y-2 ml-7">
                  <li>• Natural language requests (e.g., "check for data leakage")</li>
                  <li>• Code snippets (first 1,000 characters only)</li>
                  <li>• File paths (anonymized for privacy)</li>
                  <li>• Analysis results and pattern detections</li>
                </ul>
                <p className="text-slate-400 text-sm mt-2">
                  <strong>Legal Basis:</strong> Consent (GDPR Art. 6(1)(a)) - Only with
                  your explicit consent
                </p>
              </div>
            </div>
          </div>

          {/* GDPR Rights */}
          <div className="card-relief p-8" id="gdpr-rights">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Your Rights Under GDPR
            </h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
                  <h3 className="text-green-400 font-semibold mb-2">
                    Right of Access (Art. 15)
                  </h3>
                  <p className="text-slate-300 text-sm">
                    Request a copy of all personal data we hold about you
                  </p>
                </div>

                <div className="p-4 bg-blue-900/20 border border-blue-500/20 rounded-lg">
                  <h3 className="text-blue-400 font-semibold mb-2">
                    Right to Rectification (Art. 16)
                  </h3>
                  <p className="text-slate-300 text-sm">
                    Correct inaccurate or incomplete personal data
                  </p>
                </div>

                <div className="p-4 bg-red-900/20 border border-red-500/20 rounded-lg">
                  <h3 className="text-red-400 font-semibold mb-2">
                    Right to Erasure (Art. 17)
                  </h3>
                  <p className="text-slate-300 text-sm">
                    Request deletion of your personal data ("Right to be Forgotten")
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="p-4 bg-purple-900/20 border border-purple-500/20 rounded-lg">
                  <h3 className="text-purple-400 font-semibold mb-2">
                    Right to Data Portability (Art. 20)
                  </h3>
                  <p className="text-slate-300 text-sm">
                    Receive your personal data in machine-readable format
                  </p>
                </div>

                <div className="p-4 bg-yellow-900/20 border border-yellow-500/20 rounded-lg">
                  <h3 className="text-yellow-400 font-semibold mb-2">
                    Right to Object (Art. 21)
                  </h3>
                  <p className="text-slate-300 text-sm">
                    Object to processing based on legitimate interests
                  </p>
                </div>

                <div className="p-4 bg-orange-900/20 border border-orange-500/20 rounded-lg">
                  <h3 className="text-orange-400 font-semibold mb-2">
                    Withdraw Consent (Art. 7)
                  </h3>
                  <p className="text-slate-300 text-sm">
                    Withdraw consent for research data collection at any time
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Data Security */}
          <div className="card-relief p-8" id="data-security">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Data Protection and Security
            </h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-4">
                  Technical Safeguards
                </h3>
                <ul className="text-slate-300 space-y-2">
                  <li>• Encryption in transit (HTTPS/TLS)</li>
                  <li>• Encryption at rest (AES-256)</li>
                  <li>• Access controls with role-based permissions</li>
                  <li>• Regular security audits and monitoring</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-4">
                  Privacy by Design
                </h3>
                <ul className="text-slate-300 space-y-2">
                  <li>• Automatic PII detection and removal</li>
                  <li>• Data anonymization before research use</li>
                  <li>• Minimal data collection</li>
                  <li>• Purpose limitation</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Contact */}
          <div className="card-relief p-8" id="contact">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Contact Information
            </h2>

            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <h3 className="text-slate-100 font-semibold mb-2">
                  General Privacy Questions
                </h3>
                <p className="text-slate-300 text-sm mb-1">privacy@attrahere.com</p>
                <p className="text-slate-400 text-xs">Response time: Within 72 hours</p>
              </div>

              <div>
                <h3 className="text-slate-100 font-semibold mb-2">
                  Data Protection Officer
                </h3>
                <p className="text-slate-300 text-sm mb-1">dpo@attrahere.com</p>
                <p className="text-slate-400 text-xs">GDPR compliance matters</p>
              </div>

              <div>
                <h3 className="text-slate-100 font-semibold mb-2">GDPR Requests</h3>
                <p className="text-slate-300 text-sm mb-1">gdpr@attrahere.com</p>
                <p className="text-slate-400 text-xs">
                  Processing time: Within 30 days
                </p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="card-relief p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Exercise Your Rights
            </h2>
            <div className="flex flex-wrap gap-4">
              <Button className="bg-sky-600 hover:bg-sky-700">
                <Download className="w-4 h-4 mr-2" />
                Download My Data
              </Button>
              <Button
                variant="outline"
                className="border-red-500 text-red-400 hover:bg-red-900/20"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Request Data Deletion
              </Button>
              <Button variant="outline">
                <Shield className="w-4 h-4 mr-2" />
                Manage Consent
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
