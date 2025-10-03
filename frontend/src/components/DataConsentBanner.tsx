'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { X, Info, Shield, Brain } from 'lucide-react'

interface DataConsentBannerProps {
  onAccept: () => void
  onDecline: () => void
}

const DataConsentBanner: React.FC<DataConsentBannerProps> = ({
  onAccept,
  onDecline,
}) => {
  const [isVisible, setIsVisible] = useState(false)
  const [showDetails, setShowDetails] = useState(false)

  useEffect(() => {
    // Check if user has already given consent
    const consent = localStorage.getItem('ml_platform_data_consent')
    if (!consent) {
      setIsVisible(true)
    }
  }, [])

  const handleAccept = () => {
    localStorage.setItem('ml_platform_data_consent', 'accepted')
    localStorage.setItem('ml_platform_consent_date', new Date().toISOString())
    setIsVisible(false)
    onAccept()
  }

  const handleDecline = () => {
    localStorage.setItem('ml_platform_data_consent', 'declined')
    localStorage.setItem('ml_platform_consent_date', new Date().toISOString())
    setIsVisible(false)
    onDecline()
  }

  if (!isVisible) return null

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 bg-background/95 backdrop-blur border-t">
      <Card className="max-w-4xl mx-auto">
        <CardContent className="p-6">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-3 flex-1">
              <div className="flex items-center gap-2 mt-1">
                <Brain className="h-5 w-5 text-primary" />
                <Shield className="h-4 w-4 text-green-600" />
              </div>

              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <h3 className="font-semibold text-lg">
                    Consenso per la Raccolta Dati di Ricerca
                  </h3>
                  <Badge variant="outline" className="text-xs">
                    <Info className="h-3 w-3 mr-1" />
                    Ricerca Accademica
                  </Badge>
                </div>

                <p className="text-sm text-muted-foreground mb-3">
                  Questa piattaforma raccoglie dati anonimi sulle tue interazioni per
                  migliorare il rilevamento dei pattern ML e supportare la ricerca
                  accademica. I tuoi dati contribuiranno al progresso della qualità del
                  codice ML.
                </p>

                {!showDetails ? (
                  <Button
                    variant="link"
                    size="sm"
                    onClick={() => setShowDetails(true)}
                    className="p-0 h-auto text-primary"
                  >
                    Mostra dettagli sui dati raccolti
                  </Button>
                ) : (
                  <div className="bg-muted p-3 rounded-lg text-sm space-y-2">
                    <p className="font-medium">Dati raccolti:</p>
                    <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                      <li>Richieste in linguaggio naturale (anonimizzate)</li>
                      <li>Frammenti di codice analizzati (primi 1000 caratteri)</li>
                      <li>Pattern rilevati e punteggi di confidenza</li>
                      <li>Durata dell'analisi e timestamp</li>
                      <li>Feedback sui pattern (thumbs up/down)</li>
                    </ul>
                    <div className="flex items-center gap-2 mt-2 pt-2 border-t">
                      <Shield className="h-4 w-4 text-green-600" />
                      <span className="text-xs font-medium">
                        Nessun dato personale identificabile viene raccolto
                      </span>
                    </div>
                    <Button
                      variant="link"
                      size="sm"
                      onClick={() => setShowDetails(false)}
                      className="p-0 h-auto text-primary"
                    >
                      Nascondi dettagli
                    </Button>
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleDecline}
                className="text-xs"
              >
                Rifiuta
              </Button>
              <Button size="sm" onClick={handleAccept} className="bg-primary text-xs">
                Accetto
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsVisible(false)}
                className="h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default DataConsentBanner
