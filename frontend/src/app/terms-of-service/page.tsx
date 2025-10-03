'use client'

import React from 'react'
import {
  ArrowLeft,
  FileText,
  Users,
  GraduationCap,
  Shield,
  AlertTriangle,
} from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function TermsOfServicePage() {
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
              <FileText className="w-8 h-8 text-sky-500" />
              <h1 className="text-3xl font-bold text-slate-100">Terms of Service</h1>
            </div>
            <p className="text-slate-300 text-lg">
              Attrahere - ML Code Quality Platform - Academic & Enterprise Terms
            </p>
            <p className="text-slate-400 text-sm mt-2">
              Last Updated: September 13, 2025 | Effective Date: September 13, 2025
            </p>
          </div>
        </div>

        {/* Quick Links */}
        <div className="card-relief p-6 mb-8">
          <h2 className="text-slate-100 text-lg font-semibold mb-4">
            Quick Navigation
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <a
              href="#acceptable-use"
              className="text-sky-400 hover:text-sky-300 text-sm"
            >
              Acceptable Use
            </a>
            <a href="#academic-use" className="text-sky-400 hover:text-sky-300 text-sm">
              Academic Use
            </a>
            <a
              href="#intellectual-property"
              className="text-sky-400 hover:text-sky-300 text-sm"
            >
              IP Rights
            </a>
            <a href="#termination" className="text-sky-400 hover:text-sky-300 text-sm">
              Termination
            </a>
          </div>
        </div>

        <div className="space-y-8">
          {/* Introduction */}
          <div className="card-relief p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-4">
              Acceptance of Terms
            </h2>
            <p className="text-slate-300 leading-relaxed mb-4">
              By accessing or using Attrahere - ML Code Quality Platform (the
              "Service"), you agree to be bound by these Terms of Service ("Terms"). If
              you do not agree to these Terms, please do not use the Service.
            </p>
            <p className="text-slate-300 leading-relaxed">
              These Terms constitute a legally binding agreement between you and
              Attrahere - ML Code Quality Platform ("we," "us," or "Company").
            </p>
          </div>

          {/* Service Description */}
          <div className="card-relief p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Description of Service
            </h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-3 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-sky-500" />
                  Platform Overview
                </h3>
                <ul className="text-slate-300 space-y-2">
                  <li>• AI-powered code analysis service</li>
                  <li>• ML anti-pattern detection</li>
                  <li>• Natural language code assessment</li>
                  <li>• Academic research support</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-3 flex items-center gap-2">
                  <GraduationCap className="w-5 h-5 text-sky-500" />
                  Academic Focus
                </h3>
                <ul className="text-slate-300 space-y-2">
                  <li>• Computer science education</li>
                  <li>• ML research and development</li>
                  <li>• Academic publications support</li>
                  <li>• Student learning enhancement</li>
                </ul>
              </div>
            </div>
          </div>

          {/* User Accounts */}
          <div className="card-relief p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              User Accounts and Eligibility
            </h2>

            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-3 flex items-center gap-2">
                  <Users className="w-5 h-5 text-sky-500" />
                  Account Types
                </h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
                    <h4 className="text-green-400 font-semibold mb-2">Admin</h4>
                    <p className="text-slate-300 text-sm">
                      Full platform access and user management
                    </p>
                  </div>
                  <div className="p-4 bg-blue-900/20 border border-blue-500/20 rounded-lg">
                    <h4 className="text-blue-400 font-semibold mb-2">Researcher</h4>
                    <p className="text-slate-300 text-sm">
                      Analysis tools and data export capabilities
                    </p>
                  </div>
                  <div className="p-4 bg-purple-900/20 border border-purple-500/20 rounded-lg">
                    <h4 className="text-purple-400 font-semibold mb-2">Developer</h4>
                    <p className="text-slate-300 text-sm">
                      Basic analysis features and feedback tools
                    </p>
                  </div>
                  <div className="p-4 bg-orange-900/20 border border-orange-500/20 rounded-lg">
                    <h4 className="text-orange-400 font-semibold mb-2">Student</h4>
                    <p className="text-slate-300 text-sm">
                      Educational access through institutions
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Acceptable Use */}
          <div className="card-relief p-8" id="acceptable-use">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Acceptable Use Policy
            </h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-green-400 mb-3">
                  ✅ Permitted Uses
                </h3>
                <ul className="text-slate-300 space-y-2">
                  <li>• Analyze your own code for quality improvement</li>
                  <li>• Conduct legitimate academic research</li>
                  <li>• Learn ML best practices and anti-patterns</li>
                  <li>• Collaborate on educational projects</li>
                  <li>• Export your data for research purposes</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-red-400 mb-3">
                  ❌ Prohibited Uses
                </h3>
                <ul className="text-slate-300 space-y-2">
                  <li>• Upload malicious code or harm the platform</li>
                  <li>• Analyze code without proper authorization</li>
                  <li>• Share sensitive or proprietary code</li>
                  <li>• Reverse-engineer analysis algorithms</li>
                  <li>• Commercial use without authorization</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Academic Use */}
          <div className="card-relief p-8" id="academic-use">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Academic Use and Research
            </h2>

            <div className="space-y-6">
              <div className="p-6 bg-sky-900/20 border border-sky-500/20 rounded-lg">
                <h3 className="text-sky-400 font-semibold mb-3">
                  Educational Institution Partnerships
                </h3>
                <p className="text-slate-300 mb-4">
                  We offer special terms for educational institutions with reduced fees,
                  bulk student access, and research collaboration opportunities.
                </p>
                <ul className="text-slate-300 space-y-1 text-sm">
                  <li>• Reduced or waived fees for academic use</li>
                  <li>• Bulk student access with institutional agreements</li>
                  <li>• Research collaboration opportunities</li>
                  <li>• Data processing agreements for compliance</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-slate-100 mb-3">
                  Research Publications
                </h3>
                <p className="text-slate-300 mb-3">
                  When using our platform for research:
                </p>
                <ul className="text-slate-300 space-y-2">
                  <li>• Cite the platform appropriately in publications</li>
                  <li>• Follow academic integrity guidelines</li>
                  <li>• Respect data protection requirements</li>
                  <li>• Share findings with academic community when appropriate</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Intellectual Property */}
          <div className="card-relief p-8" id="intellectual-property">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Intellectual Property Rights
            </h2>

            <div className="space-y-6">
              <div className="p-4 border border-slate-600 rounded-lg">
                <h3 className="text-slate-100 font-semibold mb-2">
                  Your Code and Data
                </h3>
                <ul className="text-slate-300 space-y-1 text-sm">
                  <li>• You retain ownership of all code you submit</li>
                  <li>• You grant us limited license to analyze your code</li>
                  <li>• You can delete your data at any time</li>
                  <li>• We do not claim ownership of your IP</li>
                </ul>
              </div>

              <div className="p-4 border border-slate-600 rounded-lg">
                <h3 className="text-slate-100 font-semibold mb-2">
                  Platform Technology
                </h3>
                <ul className="text-slate-300 space-y-1 text-sm">
                  <li>• Analysis algorithms are our proprietary property</li>
                  <li>• ML models and detection systems are protected</li>
                  <li>• Academic research should cite appropriately</li>
                  <li>• Commercial applications require separate licensing</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Service Availability */}
          <div className="card-relief p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Service Availability and Support
            </h2>

            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center p-4 bg-green-900/20 border border-green-500/20 rounded-lg">
                <h3 className="text-green-400 font-semibold mb-2">Basic Tier</h3>
                <p className="text-slate-300 text-sm">
                  Free for educational and non-commercial use
                </p>
              </div>
              <div className="text-center p-4 bg-blue-900/20 border border-blue-500/20 rounded-lg">
                <h3 className="text-blue-400 font-semibold mb-2">Research Tier</h3>
                <p className="text-slate-300 text-sm">
                  Enhanced features for academic institutions
                </p>
              </div>
              <div className="text-center p-4 bg-purple-900/20 border border-purple-500/20 rounded-lg">
                <h3 className="text-purple-400 font-semibold mb-2">Enterprise Tier</h3>
                <p className="text-slate-300 text-sm">
                  Commercial licensing available separately
                </p>
              </div>
            </div>
          </div>

          {/* Disclaimers */}
          <div className="card-relief p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-6 flex items-center gap-2">
              <AlertTriangle className="w-6 h-6 text-yellow-500" />
              Disclaimers and Limitations
            </h2>

            <div className="space-y-4">
              <div className="p-4 bg-yellow-900/20 border border-yellow-500/20 rounded-lg">
                <h3 className="text-yellow-400 font-semibold mb-2">
                  Service Disclaimers
                </h3>
                <ul className="text-slate-300 space-y-1 text-sm">
                  <li>• "As-is" service provision without warranties</li>
                  <li>• No guarantee of accuracy in analysis results</li>
                  <li>• Educational tool not suitable for production decisions</li>
                  <li>• Continuous improvement may affect consistency</li>
                </ul>
              </div>

              <div className="p-4 bg-red-900/20 border border-red-500/20 rounded-lg">
                <h3 className="text-red-400 font-semibold mb-2">
                  Academic Use Limitations
                </h3>
                <ul className="text-slate-300 space-y-1 text-sm">
                  <li>• Results are for educational purposes only</li>
                  <li>• Human review is recommended for all results</li>
                  <li>• Critical applications should use multiple methods</li>
                  <li>• Professional judgment should supplement analysis</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Termination */}
          <div className="card-relief p-8" id="termination">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">Termination</h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-slate-100 font-semibold mb-3">
                  Termination by You
                </h3>
                <p className="text-slate-300 mb-3">
                  You may terminate your account at any time by:
                </p>
                <ul className="text-slate-300 space-y-1 text-sm">
                  <li>• Using the account deletion feature</li>
                  <li>• Submitting a GDPR erasure request</li>
                  <li>• Contacting our support team</li>
                  <li>• Simply stopping use of the Service</li>
                </ul>
              </div>

              <div>
                <h3 className="text-slate-100 font-semibold mb-3">
                  Effects of Termination
                </h3>
                <p className="text-slate-300 mb-3">Upon termination:</p>
                <ul className="text-slate-300 space-y-1 text-sm">
                  <li>• Access to Service is immediately revoked</li>
                  <li>• Data deletion procedures are initiated</li>
                  <li>• Backup data removed within 30 days</li>
                  <li>• Research data may be retained per agreements</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Contact */}
          <div className="card-relief p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">
              Contact Information
            </h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-slate-100 font-semibold mb-2">General Inquiries</h3>
                <p className="text-slate-300 text-sm mb-1">support@attrahere.com</p>
                <p className="text-slate-400 text-xs">https://attrahere.com</p>
              </div>

              <div>
                <h3 className="text-slate-100 font-semibold mb-2">
                  Academic Partnerships
                </h3>
                <p className="text-slate-300 text-sm mb-1">academic@attrahere.com</p>
                <p className="text-slate-400 text-xs">Research Portal: /research</p>
              </div>
            </div>
          </div>

          {/* Acceptance */}
          <div className="card-relief-strong p-8 border-2 border-sky-500/30">
            <div className="text-center">
              <Shield className="w-12 h-12 text-sky-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-slate-100 mb-4">
                Agreement Acknowledgment
              </h2>
              <p className="text-slate-300 leading-relaxed">
                <strong>
                  By using Attrahere - ML Code Quality Platform, you acknowledge that
                  you have read, understood, and agree to be bound by these Terms of
                  Service.
                </strong>
              </p>
              <p className="text-slate-400 text-sm mt-4">
                For the most current version: https://attrahere.com/terms
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
