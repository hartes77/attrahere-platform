'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import AttrahereLogo from '@/components/AttrahereLogo'
import BetaRequestForm from '@/components/BetaRequestForm'
import {
  Brain,
  Code2,
  Shield,
  Zap,
  GitBranch,
  Rocket,
  ChevronRight,
  Mail,
  Star,
  TrendingUp,
  Users,
  Award,
  ArrowRight,
} from 'lucide-react'

export default function LandingPage() {
  const [showFullForm, setShowFullForm] = useState(false)

  const handleShowForm = () => {
    setShowFullForm(true)
    // Smooth scroll to form after a brief delay to let it render
    setTimeout(() => {
      document.getElementById('beta-form')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
    }, 100)
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Navigation Header */}
      <nav className="w-full py-6 px-8 border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <AttrahereLogo size="md" showText={true} className="text-sky-500" />
          <div className="flex items-center gap-6">
            <Link
              href="/login"
              className="text-slate-300 hover:text-sky-400 transition-colors"
            >
              Accedi
            </Link>
            <Button className="bg-sky-500 hover:bg-sky-600 text-white button-relief">
              Richiedi Demo
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="mb-8">
            <AttrahereLogo size="xl" showText={false} className="justify-center" />
          </div>

          <h1 className="text-7xl font-bold mb-6 bg-gradient-to-r from-sky-400 to-blue-500 bg-clip-text text-transparent">
            Attrahere
          </h1>

          <div className="max-w-4xl mx-auto mb-8">
            <h2 className="text-4xl font-bold text-slate-200 mb-6 leading-tight">
              La Soluzione Definitiva al{' '}
              <span className="text-red-400">Debito Tecnico 2.0</span>
            </h2>

            <p className="text-xl text-slate-300 leading-relaxed mb-4">
              Qualità e Sicurezza per il Codice scritto dall'AI. La prima piattaforma
              conversazionale che comprende, insegna e migliora la qualità del codice ML
              attraverso intelligenza artificiale.
            </p>

            <p className="text-lg text-slate-400 leading-relaxed">
              Il 73% del codice ML generato da AI tools funziona ma contiene
              anti-pattern critici che causano data leakage, sprechi GPU e vulnerabilità
              di sicurezza da milioni di euro.
            </p>
          </div>

          {/* CTA Quick Actions */}
          <div className="mb-12">
            <Button
              onClick={handleShowForm}
              className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-12 py-6 text-xl font-bold button-relief mr-4"
            >
              <Mail className="h-6 w-6 mr-3" />
              Richiedi Demo Esclusiva
            </Button>
            <span className="text-slate-400">oppure</span>
            <Button
              variant="outline"
              className="ml-4 border-slate-600 text-slate-300 hover:bg-slate-800 px-8 py-6 text-lg font-bold button-relief"
              asChild
            >
              <Link href="/login">Prova Subito</Link>
            </Button>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <Button
              size="lg"
              className="bg-sky-500 hover:bg-sky-600 text-white px-10 py-6 text-xl font-bold button-relief"
              asChild
            >
              <Link href="/login">
                <Rocket className="h-6 w-6 mr-3" />
                Prova Subito Gratis
                <ChevronRight className="h-6 w-6 ml-2" />
              </Link>
            </Button>

            <Button
              variant="outline"
              size="lg"
              className="border-slate-600 text-slate-300 hover:bg-slate-800 px-10 py-6 text-xl font-bold button-relief"
            >
              <Brain className="h-6 w-6 mr-3" />
              Guarda la Demo AI
            </Button>
          </div>
        </div>
      </section>

      {/* Beta Request Form Modal */}
      {showFullForm && (
        <section id="beta-form" className="py-20 px-8">
          <div className="max-w-4xl mx-auto">
            <BetaRequestForm
              onSuccess={() => setShowFullForm(false)}
              className="max-w-4xl mx-auto"
            />
          </div>
        </section>
      )}

      {/* Features Grid */}
      <section className="py-20 px-8 bg-slate-800/50">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-slate-100 mb-16">
            Perché Attrahere è Rivoluzionario
          </h2>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="card-relief-strong p-8 text-center hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-purple-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Brain className="h-8 w-8 text-purple-400" />
              </div>
              <h3 className="text-xl font-bold text-slate-100 mb-4">
                AI Conversazionale
              </h3>
              <p className="text-slate-300 leading-relaxed">
                "Controlla questo codice per data leakage" → Analisi mirata automatica.
                Prima piattaforma che capisce il linguaggio naturale.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="card-relief-strong p-8 text-center hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-red-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Shield className="h-8 w-8 text-red-400" />
              </div>
              <h3 className="text-xl font-bold text-slate-100 mb-4">
                Debito Tecnico 2.0
              </h3>
              <p className="text-slate-300 leading-relaxed">
                Risolve il nuovo debito tecnico generato da AI tools. Rileva 17+ ML
                anti-pattern con 90-95% accuratezza.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="card-relief-strong p-8 text-center hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-green-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <TrendingUp className="h-8 w-8 text-green-400" />
              </div>
              <h3 className="text-xl font-bold text-slate-100 mb-4">
                AI Auto-Migliorante
              </h3>
              <p className="text-slate-300 leading-relaxed">
                Sistema di apprendimento continuo che migliora automaticamente con il
                feedback degli utenti. Più lo usi, più diventa preciso.
              </p>
            </div>

            {/* Feature 4 */}
            <div className="card-relief-strong p-8 text-center hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-blue-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Code2 className="h-8 w-8 text-blue-400" />
              </div>
              <h3 className="text-xl font-bold text-slate-100 mb-4">
                Multi-Layer Analysis
              </h3>
              <p className="text-slate-300 leading-relaxed">
                AST + AI + Rules Engine combinati. Analysis statica, semantica e pattern
                recognition in un'unica piattaforma.
              </p>
            </div>

            {/* Feature 5 */}
            <div className="card-relief-strong p-8 text-center hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-yellow-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <GitBranch className="h-8 w-8 text-yellow-400" />
              </div>
              <h3 className="text-xl font-bold text-slate-100 mb-4">
                Enterprise Ready
              </h3>
              <p className="text-slate-300 leading-relaxed">
                JWT auth, RBAC, audit logging, GDPR compliance. Production-ready con
                enterprise security dal primo giorno.
              </p>
            </div>

            {/* Feature 6 */}
            <div className="card-relief-strong p-8 text-center hover:scale-105 transition-all duration-300">
              <div className="w-16 h-16 bg-cyan-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Zap className="h-8 w-8 text-cyan-400" />
              </div>
              <h3 className="text-xl font-bold text-slate-100 mb-4">
                Real-Time Analysis
              </h3>
              <p className="text-slate-300 leading-relaxed">
                Analisi in tempo reale per IDE integration. Feedback istantaneo mentre
                scrivi codice ML.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Social Proof */}
      <section className="py-20 px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-slate-100 mb-12">
            Già Scelto da Team Innovativi
          </h2>

          <div className="grid md:grid-cols-3 gap-8 mb-16">
            <div className="kpi-card">
              <div className="flex items-center justify-center mb-4">
                <Users className="h-8 w-8 text-sky-400" />
              </div>
              <p className="text-4xl font-bold text-slate-100 mb-2">500+</p>
              <p className="text-slate-400">Sviluppatori ML</p>
            </div>

            <div className="kpi-card">
              <div className="flex items-center justify-center mb-4">
                <Star className="h-8 w-8 text-yellow-400" />
              </div>
              <p className="text-4xl font-bold text-slate-100 mb-2">90%</p>
              <p className="text-slate-400">Pattern Accuracy</p>
            </div>

            <div className="kpi-card">
              <div className="flex items-center justify-center mb-4">
                <Award className="h-8 w-8 text-purple-400" />
              </div>
              <p className="text-4xl font-bold text-slate-100 mb-2">50+</p>
              <p className="text-slate-400">Università Partner</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-8 bg-gradient-to-r from-purple-900/50 to-blue-900/50">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-5xl font-bold text-slate-100 mb-6">
            Pronto a Rivoluzionare il Tuo Codice ML?
          </h2>

          <p className="text-xl text-slate-300 mb-8 leading-relaxed">
            Unisciti alla rivoluzione dell'AI conversazionale per la qualità del codice.
            Sii tra i primi a sperimentare il futuro dello sviluppo ML.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <Button
              size="lg"
              className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-12 py-8 text-2xl font-bold button-relief"
              asChild
            >
              <Link href="/login">
                <Brain className="h-8 w-8 mr-4" />
                Inizia Subito Gratis
                <ArrowRight className="h-8 w-8 ml-4" />
              </Link>
            </Button>
          </div>

          <p className="text-slate-400 mt-6">
            ✅ Nessuna carta di credito richiesta • ✅ Setup in 2 minuti • ✅ Supporto
            24/7
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-8 border-t border-slate-700/50">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-8">
            <div className="flex items-center gap-4">
              <AttrahereLogo size="sm" showText={true} />
              <span className="text-slate-400">
                Making AI code safer, one pattern at a time.
              </span>
            </div>

            <div className="flex items-center gap-8 text-sm">
              <Link
                href="/privacy-policy"
                className="text-slate-500 hover:text-sky-400 transition-colors"
              >
                Privacy Policy
              </Link>
              <Link
                href="/terms-of-service"
                className="text-slate-500 hover:text-sky-400 transition-colors"
              >
                Terms of Service
              </Link>
              <Link
                href="/login"
                className="text-slate-500 hover:text-sky-400 transition-colors"
              >
                Login
              </Link>
            </div>
          </div>

          <div className="text-center text-slate-500 text-sm mt-8 pt-8 border-t border-slate-700/30">
            © 2025 Attrahere. All rights reserved. Built with ❤️ for the ML Engineering
            Community.
          </div>
        </div>
      </footer>
    </div>
  )
}
