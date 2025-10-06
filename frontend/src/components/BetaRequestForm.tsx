'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Mail, Check, AlertCircle, Building, User, MessageSquare } from 'lucide-react'

interface BetaRequestFormProps {
  onSuccess?: () => void
  className?: string
}

interface FormData {
  email: string
  company: string
  name: string
  message: string
}

export default function BetaRequestForm({
  onSuccess,
  className = '',
}: BetaRequestFormProps) {
  const [formData, setFormData] = useState<FormData>({
    email: '',
    company: '',
    name: '',
    message: '',
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(false)
  const [error, setError] = useState('')

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }))
    // Clear error when user starts typing
    if (error) setError('')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    setError('')

    try {
      // Validate required fields
      if (!formData.email || !formData.name) {
        throw new Error('Email e nome sono obbligatori')
      }

      // Email validation
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      if (!emailRegex.test(formData.email)) {
        throw new Error('Inserisci un email valida')
      }

      // Make actual API call to backend
      const response = await fetch(
        'https://backend-c3bxijb07-jean-piroddis-projects.vercel.app/api/v1/demo-requests',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: formData.email,
            name: formData.name,
            company: formData.company || '',
            message: formData.message || '',
          }),
        }
      )

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.message || "Errore durante l'invio della richiesta")
      }

      const result = await response.json()
      console.log('Demo request submitted successfully:', result)

      setIsSubmitted(true)
      if (onSuccess) {
        onSuccess()
      }

      // Reset form
      setFormData({
        email: '',
        company: '',
        name: '',
        message: '',
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : "Errore durante l'invio")
    } finally {
      setIsSubmitting(false)
    }
  }

  if (isSubmitted) {
    return (
      <div className={`section-elevated p-10 text-center ${className}`}>
        <div className="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
          <Check className="h-10 w-10 text-white" />
        </div>
        <h3 className="text-2xl font-bold text-green-400 mb-4">
          🎉 Richiesta Inviata con Successo!
        </h3>
        <p className="text-slate-300 text-lg mb-4">
          Grazie per il tuo interesse in Attrahere!
        </p>
        <p className="text-slate-400">
          Il nostro team ti contatterà entro <strong>24 ore</strong> per programmare una
          demo esclusiva e discutere come Attrahere può rivoluzionare la qualità del tuo
          codice ML.
        </p>
        <div className="mt-8 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <p className="text-blue-300 text-sm">
            💡 Nel frattempo, seguici per rimanere aggiornato sulle novità dell'AI
            conversazionale
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className={`section-elevated p-10 ${className}`}>
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-slate-100 mb-4">
          🚀 Richiedi l'Accesso alla Beta Privata
        </h3>
        <p className="text-slate-300 text-lg">
          Sii tra i primi a sperimentare l'AI conversazionale per la qualità del codice
          ML
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6 max-w-2xl mx-auto">
        {/* Error Message */}
        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-300">
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Name Field */}
        <div>
          <label className="block text-slate-300 font-medium mb-2">
            <User className="h-4 w-4 inline mr-2" />
            Nome e Cognome *
          </label>
          <Input
            type="text"
            placeholder="Es. Mario Rossi"
            value={formData.name}
            onChange={e => handleInputChange('name', e.target.value)}
            required
            className="bg-slate-800 border-slate-600 text-slate-100 placeholder-slate-400 text-lg py-4"
          />
        </div>

        {/* Email Field */}
        <div>
          <label className="block text-slate-300 font-medium mb-2">
            <Mail className="h-4 w-4 inline mr-2" />
            Email Aziendale *
          </label>
          <Input
            type="email"
            placeholder="mario.rossi@azienda.com"
            value={formData.email}
            onChange={e => handleInputChange('email', e.target.value)}
            required
            className="bg-slate-800 border-slate-600 text-slate-100 placeholder-slate-400 text-lg py-4"
          />
        </div>

        {/* Company Field */}
        <div>
          <label className="block text-slate-300 font-medium mb-2">
            <Building className="h-4 w-4 inline mr-2" />
            Azienda/Università
          </label>
          <Input
            type="text"
            placeholder="Es. Tech Startup SRL"
            value={formData.company}
            onChange={e => handleInputChange('company', e.target.value)}
            className="bg-slate-800 border-slate-600 text-slate-100 placeholder-slate-400 text-lg py-4"
          />
        </div>

        {/* Message Field */}
        <div>
          <label className="block text-slate-300 font-medium mb-2">
            <MessageSquare className="h-4 w-4 inline mr-2" />
            Descrivici il tuo use case (opzionale)
          </label>
          <Textarea
            placeholder="Es. Abbiamo un team di 10 ML engineers e generiamo molto codice con ChatGPT/Copilot. Vorremmo automatizzare il quality check..."
            value={formData.message}
            onChange={e => handleInputChange('message', e.target.value)}
            rows={4}
            className="bg-slate-800 border-slate-600 text-slate-100 placeholder-slate-400 text-base resize-none"
          />
        </div>

        {/* Submit Button */}
        <Button
          type="submit"
          disabled={isSubmitting}
          className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-6 text-xl font-bold button-relief disabled:opacity-50"
        >
          {isSubmitting ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-3" />
              Inviando Richiesta...
            </>
          ) : (
            <>
              <Mail className="h-6 w-6 mr-3" />
              Richiedi Accesso alla Beta Privata
            </>
          )}
        </Button>

        {/* Privacy Note */}
        <p className="text-slate-400 text-sm text-center">
          Inviando questa richiesta accetti la nostra{' '}
          <a
            href="/privacy-policy"
            className="text-sky-400 hover:text-sky-300 underline"
          >
            Privacy Policy
          </a>
          . I tuoi dati saranno utilizzati solo per contattarti riguardo Attrahere.
        </p>
      </form>
    </div>
  )
}
