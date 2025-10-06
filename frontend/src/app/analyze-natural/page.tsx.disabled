'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { apiClient } from '@/lib/api'
import {
  Brain,
  Code,
  FileText,
  Zap,
  AlertCircle,
  CheckCircle,
  Clock,
  ThumbsUp,
  ThumbsDown,
  MessageCircle,
  ArrowLeft,
  Copy,
} from 'lucide-react'
import DataConsentBanner from '@/components/DataConsentBanner'
import Link from 'next/link'
import { PrivateBetaGuard } from '@/components/PrivateBetaGuard'

interface NaturalAnalysisResult {
  id: string
  patterns: Array<{
    id: string
    name: string
    severity: 'critical' | 'high' | 'medium' | 'low'
    confidence: number
    reasoning: string
    suggestion: string
    line: number
    source: string
  }>
  summary: {
    total_patterns: number
    severity_counts: Record<string, number>
    overall_score: number
  }
  natural_language: {
    original_request: string
    interpreted_intent: string
    confidence: number
    intelligent_insights: Record<string, any>
  }
}

function NaturalAnalysisPageContent() {
  const [naturalRequest, setNaturalRequest] = useState('')
  const [code, setCode] = useState('')
  const [projectPath, setProjectPath] = useState('')
  const [analysisMode, setAnalysisMode] = useState<'code' | 'project'>('code')

  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [results, setResults] = useState<NaturalAnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<Set<string>>(new Set())
  const [dataConsent, setDataConsent] = useState<string | null>(null)
  const [copiedPatterns, setCopiedPatterns] = useState<Set<string>>(new Set())

  React.useEffect(() => {
    // Check consent status on component mount
    const consent = localStorage.getItem('ml_platform_data_consent')
    setDataConsent(consent)
  }, [])

  const submitFeedback = async (
    patternId: string,
    feedbackType: 'correct' | 'incorrect' | 'missing',
    rating?: number,
    comment?: string
  ) => {
    if (!taskId) return

    try {
      await apiClient.post('/api/v1/research/feedback', {
        pattern_id: patternId,
        task_id: taskId,
        feedback_type: feedbackType,
        rating: rating,
        comment: comment,
      })

      setFeedbackSubmitted(prev => new Set([...prev, patternId]))
    } catch (error) {
      console.error('Failed to submit feedback:', error)
    }
  }

  const handleConsentAccept = () => {
    setDataConsent('accepted')
  }

  const handleConsentDecline = () => {
    setDataConsent('declined')
  }

  const exampleRequests = {
    code: [
      'controlla questo codice per data leakage e memory leak',
      'trova problemi di overfitting nel training loop',
      'analizza per magic numbers e hardcoded parameters',
      'verifica la sicurezza di questo script ML',
    ],
    project: [
      'analizza tutto il progetto per inconsistenze nel preprocessing',
      'trova problemi di performance in questo repository PyTorch',
      'controlla cross-file per pattern di data leakage',
      'audit completo della pipeline di training',
    ],
  }

  const handleAnalyze = async () => {
    if (!naturalRequest.trim()) {
      setError('Inserisci una richiesta in linguaggio naturale')
      return
    }

    if (analysisMode === 'code' && !code.trim()) {
      setError('Inserisci il codice da analizzare')
      return
    }

    if (analysisMode === 'project' && !projectPath.trim()) {
      setError('Inserisci il path del progetto')
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setResults(null)
    setProgress(0)

    try {
      // Start natural language analysis
      const payload = {
        request: naturalRequest,
        ...(analysisMode === 'code' ? { code } : { project_path: projectPath }),
      }

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/analyze-natural`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      )

      if (!response.ok) {
        throw new Error('Analysis failed')
      }

      const data = await response.json()
      setTaskId(data.task_id)

      // Poll for results
      await pollTaskStatus(data.task_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
      setIsAnalyzing(false)
    }
  }

  const pollTaskStatus = async (taskId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/v1/tasks/${taskId}/status`
        )
        const status = await response.json()

        setProgress(status.progress || 0)

        if (status.status === 'completed' && status.successful) {
          setResults(status.result)
          setIsAnalyzing(false)
          clearInterval(pollInterval)
        } else if (status.status === 'failed') {
          setError(status.error || 'Analysis failed')
          setIsAnalyzing(false)
          clearInterval(pollInterval)
        }
      } catch (err) {
        setError('Failed to get analysis status')
        setIsAnalyzing(false)
        clearInterval(pollInterval)
      }
    }, 1000)

    // Timeout after 5 minutes
    setTimeout(() => {
      clearInterval(pollInterval)
      if (isAnalyzing) {
        setError('Analysis timeout')
        setIsAnalyzing(false)
      }
    }, 300000)
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'destructive'
      case 'high':
        return 'destructive'
      case 'medium':
        return 'default'
      case 'low':
        return 'secondary'
      default:
        return 'outline'
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      case 'high':
        return <AlertCircle className="h-4 w-4 text-orange-500" />
      case 'medium':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'low':
        return <CheckCircle className="h-4 w-4 text-blue-500" />
      default:
        return <AlertCircle className="h-4 w-4" />
    }
  }

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Back Button */}
      <div className="mb-6">
        <Link href="/analyze">
          <Button variant="outline" className="flex items-center gap-2">
            <ArrowLeft className="h-4 w-4" />
            Torna alla Dashboard
          </Button>
        </Link>
      </div>

      {/* Header */}
      <div className="flex items-center space-x-4">
        <Brain className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Natural Language Analysis</h1>
          <p className="text-muted-foreground">
            Analizza codice ML usando richieste in linguaggio naturale con Gemma AI
          </p>
        </div>
      </div>

      {/* Analysis Mode Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="h-5 w-5" />
            <span>Modalità Analisi</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <Button
              variant={analysisMode === 'code' ? 'default' : 'outline'}
              onClick={() => setAnalysisMode('code')}
              className="h-20 flex flex-col space-y-2"
            >
              <Code className="h-6 w-6" />
              <span>Singolo File</span>
            </Button>
            <Button
              variant={analysisMode === 'project' ? 'default' : 'outline'}
              onClick={() => setAnalysisMode('project')}
              className="h-20 flex flex-col space-y-2"
            >
              <FileText className="h-6 w-6" />
              <span>Progetto Completo</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Natural Language Request */}
      <Card>
        <CardHeader>
          <CardTitle>Richiesta in Linguaggio Naturale</CardTitle>
          <CardDescription>
            Descrivi cosa vuoi analizzare usando linguaggio naturale
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Es: controlla questo codice per data leakage e memory leak..."
            value={naturalRequest}
            onChange={e => setNaturalRequest(e.target.value)}
            className="min-h-[100px]"
          />

          {/* Example requests */}
          <div>
            <p className="text-sm font-medium mb-2">Esempi ({analysisMode}):</p>
            <div className="flex flex-wrap gap-2">
              {exampleRequests[analysisMode].map((example, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => setNaturalRequest(example)}
                  className="text-xs"
                >
                  {example}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Code Input or Project Path */}
      <Card>
        <CardHeader>
          <CardTitle>
            {analysisMode === 'code' ? 'Codice da Analizzare' : 'Path Progetto'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {analysisMode === 'code' ? (
            <Textarea
              placeholder="Incolla il tuo codice Python qui..."
              value={code}
              onChange={e => setCode(e.target.value)}
              className="min-h-[300px] font-mono text-sm"
            />
          ) : (
            <Input
              placeholder="/path/to/your/project"
              value={projectPath}
              onChange={e => setProjectPath(e.target.value)}
            />
          )}
        </CardContent>
      </Card>

      {/* Analyze Button */}
      <div className="flex justify-center">
        <Button
          onClick={handleAnalyze}
          disabled={isAnalyzing}
          size="lg"
          className="w-full max-w-md"
        >
          {isAnalyzing ? (
            <>
              <Brain className="mr-2 h-4 w-4 animate-spin" />
              Analizzando con Gemma...
            </>
          ) : (
            <>
              <Brain className="mr-2 h-4 w-4" />
              Analizza con Gemma AI
            </>
          )}
        </Button>
      </div>

      {/* Progress */}
      {isAnalyzing && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progresso Analisi</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
              <p className="text-xs text-muted-foreground text-center">
                Gemma sta interpretando la richiesta e analizzando il codice...
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error */}
      {error && (
        <Card className="border-destructive">
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2 text-destructive">
              <AlertCircle className="h-4 w-4" />
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-6">
          {/* Intent Interpretation */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="h-5 w-5 text-primary" />
                <span>Interpretazione Gemma</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium">Richiesta Originale:</p>
                  <p className="text-sm text-muted-foreground">
                    {results.natural_language?.original_request || naturalRequest}
                  </p>
                </div>
                <div>
                  <p className="text-sm font-medium">Intent Rilevato:</p>
                  <Badge variant="outline">
                    {results.natural_language?.interpreted_intent || 'code_analysis'}
                  </Badge>
                </div>
              </div>
              <div>
                <p className="text-sm font-medium">Confidence:</p>
                <div className="flex items-center space-x-2">
                  <Progress
                    value={(results.natural_language?.confidence || 0.5) * 100}
                    className="flex-1"
                  />
                  <span className="text-sm">
                    {Math.round((results.natural_language?.confidence || 0.5) * 100)}%
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Analysis Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Risultati Analisi</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-4 gap-4 text-center">
                <div>
                  <p className="text-2xl font-bold text-red-500">
                    {results.summary?.by_severity?.critical || 0}
                  </p>
                  <p className="text-sm text-muted-foreground">Critical</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-orange-500">
                    {results.summary?.by_severity?.high || 0}
                  </p>
                  <p className="text-sm text-muted-foreground">High</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-yellow-500">
                    {results.summary?.by_severity?.medium || 0}
                  </p>
                  <p className="text-sm text-muted-foreground">Medium</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-blue-500">
                    {results.summary?.by_severity?.low || 0}
                  </p>
                  <p className="text-sm text-muted-foreground">Low</p>
                </div>
              </div>

              <div className="text-center">
                <p className="text-3xl font-bold">
                  {Math.round(results.summary?.overall_score || 50)}/100
                </p>
                <p className="text-muted-foreground">Quality Score</p>
              </div>
            </CardContent>
          </Card>

          {/* Pattern Details */}
          {results.patterns && results.patterns.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Pattern Rilevati ({results.patterns.length})</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {(results.patterns || []).map((pattern, index) => (
                  <Card key={pattern.id} className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3 flex-1">
                        {getSeverityIcon(pattern.severity)}
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <h4 className="font-medium">{pattern.name}</h4>
                            <Badge variant={getSeverityColor(pattern.severity)}>
                              {pattern.severity}
                            </Badge>
                            <Badge variant="outline">
                              {Math.round(pattern.confidence * 100)}% confident
                            </Badge>
                          </div>

                          <p className="text-sm text-muted-foreground mb-3">
                            {pattern.reasoning}
                          </p>

                          {pattern.suggestion && (
                            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 p-4 rounded-lg mb-3">
                              <div className="flex items-start space-x-3">
                                <div className="flex-shrink-0">
                                  <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                                    <span className="text-white text-sm font-bold">
                                      💡
                                    </span>
                                  </div>
                                </div>
                                <div className="flex-1">
                                  <h4 className="text-sm font-semibold text-blue-900 mb-1">
                                    🔧 Suggested Fix
                                  </h4>
                                  <p className="text-sm text-blue-800 leading-relaxed">
                                    {pattern.suggestion}
                                  </p>
                                  <div className="mt-3 flex items-center justify-between">
                                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                      <CheckCircle className="w-3 h-3 mr-1" />
                                      Quick Fix Available
                                    </span>
                                    {copiedPatterns.has(pattern.id) ? (
                                      <span className="inline-flex items-center px-3 py-1 rounded text-xs font-medium bg-green-100 text-green-700 border border-green-300">
                                        <CheckCircle className="w-3 h-3 mr-1" />✅
                                        Copied!
                                      </span>
                                    ) : (
                                      <Button
                                        size="sm"
                                        onClick={() => {
                                          navigator.clipboard.writeText(
                                            pattern.suggestion
                                          )
                                          setCopiedPatterns(
                                            prev => new Set([...prev, pattern.id])
                                          )
                                          setTimeout(() => {
                                            setCopiedPatterns(prev => {
                                              const newSet = new Set(prev)
                                              newSet.delete(pattern.id)
                                              return newSet
                                            })
                                          }, 3000)
                                        }}
                                        className="bg-blue-500 hover:bg-blue-600 text-white text-xs px-3 py-1"
                                      >
                                        <Copy className="w-3 h-3 mr-1" />
                                        📋 Copy Fixed Code
                                      </Button>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </div>
                          )}

                          <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                            <span>Line: {pattern.line}</span>
                            <span>Source: {pattern.source}</span>
                          </div>
                        </div>
                      </div>

                      {/* Feedback Buttons */}
                      <div className="mt-3 pt-2 border-t border-gray-100">
                        <div className="flex items-center justify-between">
                          <p className="text-xs text-gray-500">🔍 Accurate?</p>
                          {feedbackSubmitted.has(pattern.id) ? (
                            <Badge className="bg-green-100 text-green-700 border-green-200 text-xs px-2 py-1">
                              <CheckCircle className="h-3 w-3 mr-1" />
                              Thanks!
                            </Badge>
                          ) : (
                            <div className="flex items-center space-x-2">
                              <Button
                                size="sm"
                                onClick={() => submitFeedback(pattern.id, 'correct')}
                                className="bg-green-500 hover:bg-green-600 text-white px-2 py-1 text-xs h-7"
                              >
                                <ThumbsUp className="h-3 w-3 mr-1" />
                                👍
                              </Button>

                              <Button
                                size="sm"
                                onClick={() => submitFeedback(pattern.id, 'incorrect')}
                                className="bg-red-500 hover:bg-red-600 text-white px-2 py-1 text-xs h-7"
                              >
                                <ThumbsDown className="h-3 w-3 mr-1" />
                                👎
                              </Button>

                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => {
                                  const comment = prompt('💬 Add comment:')
                                  if (comment !== null) {
                                    submitFeedback(
                                      pattern.id,
                                      'missing',
                                      undefined,
                                      comment || undefined
                                    )
                                  }
                                }}
                                className="border-blue-200 text-blue-600 hover:bg-blue-50 px-2 py-1 text-xs h-7"
                              >
                                <MessageCircle className="h-3 w-3" />
                              </Button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Intelligent Insights */}
          {results.natural_language?.intelligent_insights && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5 text-primary" />
                  <span>Insight Intelligenti</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-sm bg-muted p-4 rounded overflow-x-auto">
                  {JSON.stringify(
                    results.natural_language?.intelligent_insights || {},
                    null,
                    2
                  )}
                </pre>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Data Consent Banner */}
      {dataConsent !== 'accepted' && dataConsent !== 'declined' && (
        <DataConsentBanner
          onAccept={handleConsentAccept}
          onDecline={handleConsentDecline}
        />
      )}
    </div>
  )
}

export default function NaturalAnalysisPage() {
  return (
    <PrivateBetaGuard>
      <NaturalAnalysisPageContent />
    </PrivateBetaGuard>
  )
}
