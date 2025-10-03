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
              Stai lanciando modelli ML?{' '}
              <span className="text-red-400">Scopri se hai data leakage.</span>
            </h2>

            <p className="text-xl text-slate-300 leading-relaxed mb-4">
              Stiamo costruendo Attrahere - la prima piattaforma di ML Code Quality. 
              Inserisci la tua email per accedere alla beta.
            </p>

            <div className="flex flex-wrap justify-center gap-8 text-sm text-slate-400 mb-6">
              <div className="flex items-center gap-2">
                <span className="text-green-400">✓</span>
                <span>Già testato su 5.289 file HuggingFace</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-400">✓</span>
                <span>Zero falsi positivi</span>
              </div>
            </div>
          </div>

          {/* CTA Quick Actions */}
          <div className="mb-12">
            <Button
              onClick={handleShowForm}
              className="bg-sky-500 hover:bg-sky-600 text-white px-12 py-6 text-xl font-bold button-relief mr-4"
            >
              <Rocket className="h-6 w-6 mr-3" />
              🚀 ENTRA IN BETA WAITLIST
            </Button>
            <Button
              variant="outline"
              className="ml-4 border-slate-600 text-slate-300 hover:bg-slate-800 px-8 py-6 text-lg font-bold button-relief"
            >
              <Brain className="h-6 w-6 mr-3" />
              🎬 VEDI DEMO
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

      {/* Problem Section */}
      <section className="py-20 px-8 bg-red-900/20">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-slate-100 mb-12">
            Il 68% dei modelli ML fallisce in production per data leakage
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <div className="card-relief-strong p-8">
              <div className="text-3xl font-bold text-red-400 mb-2">€500k-€2M</div>
              <div className="text-slate-300">costo medio per decisioni sbagliate</div>
            </div>
            <div className="card-relief-strong p-8">
              <div className="text-3xl font-bold text-red-400 mb-2">50-100 ore/mese</div>
              <div className="text-slate-300">perse in debugging</div>
            </div>
            <div className="card-relief-strong p-8">
              <div className="text-3xl font-bold text-red-400 mb-2">Errori subtili</div>
              <div className="text-slate-300">che sfuggono al code review</div>
            </div>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-20 px-8 bg-slate-800/50">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-slate-100 mb-16">
            Da due settimane di debugging a due minuti di scan
          </h2>
          
          <div className="bg-slate-900 rounded-lg p-8 mb-12 max-w-3xl mx-auto">
            <div className="text-slate-400 mb-2"># Come funzionerà:</div>
            <div className="text-sky-400 mb-4">$ attrahere scan tuo_modello.py</div>
            <div className="text-slate-400 mb-2"></div>
            <div className="text-sky-400">🔍 Analisi completata in 1.2s</div>
            <div className="text-green-400">✅ 0 data leakage trovati</div>
            <div className="text-yellow-400">⚠️  1 potential target contamination</div>
            <div className="text-sky-400">📊 Report dettagliato generato</div>
          </div>
          
          <div className="text-center">
            <p className="text-xl text-sky-400 font-bold">Stiamo costruendo - Entra nella beta founders</p>
          </div>
        </div>
      </section>

      {/* Validation Section */}
      <section className="py-20 px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-slate-100 mb-12">
            La tecnologia è già validata - Il prodotto sta arrivando
          </h2>

          <div className="card-relief-strong p-8 max-w-4xl mx-auto">
            <h3 className="text-2xl font-bold text-slate-100 mb-6">TECNOLOGIA TESTATA SU:</h3>
            <div className="grid md:grid-cols-2 gap-6 text-left">
              <div className="flex items-center gap-3">
                <span className="text-green-400">•</span>
                <span className="text-slate-300">HuggingFace Transformers (5.289 file)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-green-400">•</span>
                <span className="text-slate-300">Competizioni Kaggle (Porto Seguro 2° posto)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-green-400">•</span>
                <span className="text-slate-300">Codebase enterprise (400k+ LOC)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-green-400">•</span>
                <span className="text-slate-300">Zero falsi positivi dimostrati</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Founder Section */}
      <section className="py-20 px-8 bg-gradient-to-r from-purple-900/30 to-blue-900/30">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-slate-100 mb-8">
            Sto costruendo Attrahere e cerco il CTO per scalare
          </h2>
          
          <p className="text-xl text-slate-300 mb-8 leading-relaxed">
            Sono Jean Piroddi. Ho passato 1.5 anni a sviluppare la tecnologia core di Attrahere. 
            Ora cerco il CTO co-founder per costruire insieme la piattaforma enterprise.
          </p>
          
          <Button
            size="lg"
            className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-12 py-6 text-xl font-bold button-relief"
          >
            🤝 SONO UN CTO - VOGLIO PARLARNE
          </Button>
        </div>
      </section>

      {/* Beta Waitlist Section */}
      <section className="py-20 px-8 bg-slate-800/50">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-slate-100 mb-8">
            Stiamo selezionando i primi 10 beta tester
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="flex items-center gap-3 text-slate-300">
              <span className="text-green-400">✅</span>
              <span>Accesso prioritario alla piattaforma</span>
            </div>
            <div className="flex items-center gap-3 text-slate-300">
              <span className="text-green-400">✅</span>
              <span>Influence il prodotto con il tuo feedback</span>
            </div>
            <div className="flex items-center gap-3 text-slate-300">
              <span className="text-green-400">✅</span>
              <span>Prezzo founding members garantito</span>
            </div>
            <div className="flex items-center gap-3 text-slate-300">
              <span className="text-green-400">✅</span>
              <span>Supporto diretto con il founder</span>
            </div>
          </div>
          
          <div className="card-relief-strong p-8">
            <Button
              onClick={handleShowForm}
              size="lg"
              className="bg-sky-500 hover:bg-sky-600 text-white px-12 py-6 text-xl font-bold button-relief mb-4"
            >
              <Mail className="h-6 w-6 mr-3" />
              ENTRA IN WAITLIST
            </Button>
            <p className="text-slate-400 text-sm">
              Inviteremo i beta tester per fase - Priorità a team ML enterprise
            </p>
          </div>
        </div>
      </section>


      {/* Final CTA Section */}
      <section className="py-20 px-8 bg-gradient-to-r from-sky-900/50 to-blue-900/50">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-5xl font-bold text-slate-100 mb-6">
            Il tuo codice ML è production-ready?
          </h2>

          <p className="text-xl text-slate-300 mb-8 leading-relaxed">
            Scoprilo in 5 minuti. Entra nella beta di Attrahere.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
            <Button
              onClick={handleShowForm}
              size="lg"
              className="bg-sky-500 hover:bg-sky-600 text-white px-12 py-8 text-2xl font-bold button-relief"
            >
              <Mail className="h-8 w-8 mr-4" />
              🎬 GUARDA LA DEMO LIVE
            </Button>
            <Button
              onClick={handleShowForm}
              variant="outline"
              size="lg"
              className="border-slate-600 text-slate-300 hover:bg-slate-800 px-12 py-8 text-2xl font-bold button-relief"
            >
              <Rocket className="h-8 w-8 mr-4" />
              💼 SCARICA IL WHITEPAPER
            </Button>
          </div>

          <p className="text-slate-400 mt-6">
            Pronto per deployment production immediato • Architecture enterprise già validata • Zero configurazione necessaria
          </p>
        </div>
      </section>

      {/* Footer Strategico */}
      <footer className="py-12 px-8 border-t border-slate-700/50">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8 mb-8">
            <div className="text-center">
              <h4 className="text-lg font-bold text-slate-100 mb-4">📧 Beta Waitlist</h4>
              <p className="text-slate-400 mb-4">Per utenti finali</p>
              <Button
                onClick={handleShowForm}
                className="bg-sky-500 hover:bg-sky-600 text-white px-6 py-3 button-relief"
              >
                Entra in Lista
              </Button>
            </div>

            <div className="text-center">
              <h4 className="text-lg font-bold text-slate-100 mb-4">🤝 CTO Application</h4>
              <p className="text-slate-400 mb-4">Per co-founder</p>
              <Button
                variant="outline"
                className="border-purple-500 text-purple-400 hover:bg-purple-900/20 px-6 py-3 button-relief"
              >
                Candidati
              </Button>
            </div>

            <div className="text-center">
              <h4 className="text-lg font-bold text-slate-100 mb-4">📞 Investor Deck</h4>
              <p className="text-slate-400 mb-4">Per incubatori/investitori</p>
              <Button
                variant="outline"
                className="border-green-500 text-green-400 hover:bg-green-900/20 px-6 py-3 button-relief"
              >
                Richiedi Info
              </Button>
            </div>
          </div>

          <div className="flex flex-col md:flex-row justify-between items-center gap-8 pt-8 border-t border-slate-700/30">
            <div className="flex items-center gap-4">
              <AttrahereLogo size="sm" showText={true} />
              <span className="text-slate-400">
                Making ML code bulletproof, one pattern at a time.
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
            </div>
          </div>

          <div className="text-center text-slate-500 text-sm mt-6">
            © 2025 Attrahere. Beta in costruzione. Built with ❤️ for the ML Engineering Community.
          </div>
        </div>
      </footer>
    </div>
  )
}
