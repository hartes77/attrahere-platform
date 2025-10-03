'use client'

import React, { useState, useEffect } from 'react'
import { MLPattern, UserFeedback } from '@/types/ml-analysis'
import { apiClient } from '@/lib/api'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import {
  BarChart3,
  User,
  FileText,
  Settings,
  LogOut,
  Code2,
  Play,
  AlertCircle,
  Users,
  Shield,
  CheckCircle,
  Brain,
  Lightbulb,
  TrendingUp,
  FolderOpen,
} from 'lucide-react'

const SAMPLE_CODE = `import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# ❌ Data leakage: preprocessing before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ❌ Missing random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# ❌ Magic numbers in PyTorch model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)  # ❌ Magic numbers
        self.fc2 = nn.Linear(256, 10)   # ❌ Magic numbers

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = NeuralNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ❌ Magic learning rate

# ❌ GPU memory leak potential
for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # ❌ Missing .detach() - can cause memory leaks

# Train sklearn model too
sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, y_train)

# Evaluate
score = sklearn_model.score(X_test, y_test)
print(f"Accuracy: {score}")`

export default function AnalyzePage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [currentUser, setCurrentUser] = useState<any>(null)
  const [code, setCode] = useState(SAMPLE_CODE)
  const [patterns, setPatterns] = useState<MLPattern[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<Set<string>>(new Set())
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null)
  const [hasAnalyzed, setHasAnalyzed] = useState(false)
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null)

  // Code analysis helper function
  const analyzeCodeContext = (code: string) => {
    const lines = code.toLowerCase().split('\n')
    const codeText = code.toLowerCase()

    // Library detection
    const libraries = {
      ml: {
        sklearn: /from sklearn|import sklearn/.test(codeText),
        tensorflow: /import tensorflow|from tensorflow/.test(codeText),
        torch: /import torch|from torch/.test(codeText),
        pandas: /import pandas|from pandas/.test(codeText),
        numpy: /import numpy|from numpy/.test(codeText),
        keras: /import keras|from keras/.test(codeText),
        xgboost: /import xgboost|from xgboost/.test(codeText),
        lightgbm: /import lightgbm|from lightgbm/.test(codeText),
      },
      web: {
        flask: /from flask|import flask/.test(codeText),
        django: /from django|import django/.test(codeText),
        fastapi: /from fastapi|import fastapi/.test(codeText),
        streamlit: /import streamlit/.test(codeText),
      },
      data: {
        sqlalchemy: /from.*sqlalchemy|import.*sqlalchemy/.test(codeText),
        sqlite: /sqlite/.test(codeText),
        postgres: /postgres|psycopg/.test(codeText),
        mysql: /mysql/.test(codeText),
      },
      general: {
        os: /import os/.test(codeText),
        sys: /import sys/.test(codeText),
        json: /import json/.test(codeText),
        requests: /import requests/.test(codeText),
      },
    }

    // ML operations detection
    const mlOperations = {
      training: /\.fit\(|\.train\(|\.fit_transform\(/.test(codeText),
      prediction: /\.predict\(|\.predict_proba\(/.test(codeText),
      preprocessing: /standardscaler|normalizer|labelencoder/.test(codeText),
      splitting: /train_test_split/.test(codeText),
      evaluation: /\.score\(|accuracy_score|confusion_matrix/.test(codeText),
      crossValidation: /cross_val_score|gridsearchcv/.test(codeText),
    }

    // Code type classification
    const codeTypes = []
    const detectedLibraries = []
    const mlFeatures = []

    // Classify code type
    if (Object.values(libraries.ml).some(Boolean)) {
      codeTypes.push('Machine Learning')
      if (libraries.ml.sklearn) detectedLibraries.push('scikit-learn')
      if (libraries.ml.tensorflow) detectedLibraries.push('TensorFlow')
      if (libraries.ml.torch) detectedLibraries.push('PyTorch')
      if (libraries.ml.pandas) detectedLibraries.push('Pandas')
      if (libraries.ml.numpy) detectedLibraries.push('NumPy')
    }

    if (Object.values(libraries.web).some(Boolean)) {
      codeTypes.push('Web Application')
      if (libraries.web.flask) detectedLibraries.push('Flask')
      if (libraries.web.django) detectedLibraries.push('Django')
      if (libraries.web.fastapi) detectedLibraries.push('FastAPI')
    }

    if (Object.values(libraries.data).some(Boolean)) {
      codeTypes.push('Database Operations')
      if (libraries.data.sqlalchemy) detectedLibraries.push('SQLAlchemy')
      if (libraries.data.sqlite) detectedLibraries.push('SQLite')
    }

    if (codeTypes.length === 0) {
      codeTypes.push('General Python')
    }

    // Detect ML operations
    if (mlOperations.training) mlFeatures.push('Model Training')
    if (mlOperations.prediction) mlFeatures.push('Prediction')
    if (mlOperations.preprocessing) mlFeatures.push('Data Preprocessing')
    if (mlOperations.splitting) mlFeatures.push('Data Splitting')
    if (mlOperations.evaluation) mlFeatures.push('Model Evaluation')

    return {
      codeTypes,
      detectedLibraries,
      mlFeatures,
      isMLCode: Object.values(libraries.ml).some(Boolean) || mlFeatures.length > 0,
      isWebApp: Object.values(libraries.web).some(Boolean),
      hasDatabase: Object.values(libraries.data).some(Boolean),
    }
  }

  useEffect(() => {
    const checkAuth = async () => {
      if (!apiClient.isAuthenticated()) {
        setIsAuthenticated(false)
        setIsLoading(false)
        return
      }

      try {
        const user = await apiClient.getMe()
        setIsAuthenticated(true)
        setCurrentUser(user)
      } catch (error) {
        console.error('Auth check failed:', error)
        setIsAuthenticated(false)
      }
      setIsLoading(false)
    }
    checkAuth()
  }, [])

  // Reset analysis state when code changes
  useEffect(() => {
    setHasAnalyzed(false)
    setPatterns([])
    setError(null)
  }, [code])

  const handleLogout = () => {
    apiClient.logout()
    setIsAuthenticated(false)
  }

  const handleAnalyze = async () => {
    if (!apiClient.isAuthenticated()) {
      setError('Please log in first')
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setPatterns([])
    setHasAnalyzed(true)

    try {
      const response = await apiClient.analyzeCode({
        code: code.trim(),
        file_path: 'analysis.py',
        language: 'python',
      })

      // Handle direct response from simple_api.py
      if (response.success && response.data) {
        // Save analysis_id for RLHF feedback
        if (response.analysis_id) {
          setCurrentAnalysisId(response.analysis_id)
        }

        const rawPatterns = response.data.patterns || []

        const transformedPatterns = rawPatterns.map((pattern: any, index: number) => ({
          id: pattern.id || `pattern-${index}`,
          type: pattern.type || 'unknown',
          severity: pattern.severity || 'medium',
          location: pattern.location || {
            file: 'analysis.py',
            startLine: pattern.line || 0,
            endLine: pattern.line || 0,
            startColumn: pattern.column || 0,
            endColumn: pattern.column || 0,
          },
          message: pattern.message || '',
          explanation: pattern.explanation || '',
          suggestedFix: pattern.suggested_fix || pattern.suggestedFix || '',
          confidence: Math.round((pattern.confidence || 0) * 100),
          context: {
            codeSnippet: pattern.code_snippet || pattern.context?.codeSnippet || '',
            fixSnippet: pattern.fix_snippet || pattern.context?.fixSnippet || '',
          },
          references: pattern.references || [],
        }))

        setPatterns(transformedPatterns)
      } else {
        // No patterns found
        setPatterns([])
      }
    } catch (error) {
      console.error('Analysis failed:', error)
      setError(error instanceof Error ? error.message : 'Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFeedback = async (feedback: UserFeedback) => {
    try {
      // Use new RLHF system for thumbs feedback
      if (
        currentAnalysisId &&
        (feedback.feedback_type === 'helpful' ||
          feedback.feedback_type === 'not_helpful')
      ) {
        const response = await apiClient.submitThumbsFeedback({
          analysis_id: currentAnalysisId,
          pattern_id: feedback.pattern_id ? parseInt(feedback.pattern_id) : undefined,
          is_helpful: feedback.feedback_type === 'helpful',
          user_id: currentUser?.id,
          comment: undefined,
        })

        // Show enhanced success message with learning info
        const feedbackType =
          feedback.feedback_type === 'helpful' ? 'positive' : 'negative'
        let message = `Thanks for your ${feedbackType} feedback!`

        if (response.learning_triggered) {
          message += ' 🧠 Your feedback is helping improve the AI!'
        }

        setFeedbackMessage(message)
      } else {
        // Fallback to old feedback system
        await apiClient.submitFeedback(feedback)

        const feedbackType =
          feedback.feedback_type === 'helpful' ? 'positive' : 'negative'
        setFeedbackMessage(`Thanks for your ${feedbackType} feedback!`)
      }

      // Add pattern ID to submitted feedback set
      setFeedbackSubmitted(prev => new Set(prev).add(feedback.pattern_id))

      // Clear message after 4 seconds (longer for learning message)
      setTimeout(() => {
        setFeedbackMessage(null)
      }, 4000)
    } catch (error) {
      console.error('Failed to submit feedback:', error)
      setFeedbackMessage('Failed to submit feedback. Please try again.')
      setTimeout(() => {
        setFeedbackMessage(null)
      }, 3000)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-300">Initializing...</p>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
        <div className="text-center max-w-2xl">
          <Code2 className="h-24 w-24 text-sky-500 mx-auto mb-8" />
          <h1 className="text-4xl font-bold text-sky-500 mb-6">ML Code Analyzer</h1>
          <p className="text-slate-300 text-lg mb-8">
            Please log in to analyze your machine learning code
          </p>
          <Link href="/login">
            <Button className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-lg rounded-xl">
              Go to Login
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-slate-900 text-slate-100">
      {/* Sidebar con effetti rilievo */}
      <aside className="w-64 section-elevated p-8 flex flex-col">
        <div className="mb-10">
          <h1 className="text-2xl font-bold text-sky-500 mb-2">Attrahere</h1>
          <div className="w-12 h-1 bg-sky-500 rounded-full"></div>
        </div>

        <nav className="space-y-3 flex-1">
          <Link href="/" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <BarChart3 className="h-5 w-5" />
                <span>Dashboard</span>
              </div>
            </div>
          </Link>

          <div className="card-relief-strong text-sky-500 px-5 py-4 rounded-xl border-sky-500/30">
            <div className="flex items-center gap-3">
              <User className="h-5 w-5" />
              <span className="font-semibold">Analyze</span>
            </div>
          </div>

          <Link href="/patterns" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <FileText className="h-5 w-5" />
                <span>Patterns</span>
              </div>
            </div>
          </Link>

          <Link href="/analyze-project" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <FolderOpen className="h-5 w-5" />
                <span>Project Analysis</span>
              </div>
            </div>
          </Link>

          <Link href="/roi-calculator" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <TrendingUp className="h-5 w-5" />
                <span>ROI Calculator</span>
              </div>
            </div>
          </Link>

          {/* Admin Only Section */}
          {currentUser?.role === 'admin' && (
            <>
              <div className="pt-4 pb-2">
                <div className="text-slate-500 text-xs uppercase tracking-wide font-medium px-3">
                  Admin Panel
                </div>
              </div>
              <Link href="/admin/users" className="block">
                <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
                  <div className="flex items-center gap-3">
                    <Users className="h-5 w-5" />
                    <span>User Management</span>
                  </div>
                </div>
              </Link>
              <Link href="/admin/audit" className="block">
                <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
                  <div className="flex items-center gap-3">
                    <Shield className="h-5 w-5" />
                    <span>Audit Logs</span>
                  </div>
                </div>
              </Link>
            </>
          )}

          <Link href="/settings" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <Settings className="h-5 w-5" />
                <span>Settings</span>
              </div>
            </div>
          </Link>
        </nav>

        <Button
          onClick={handleLogout}
          className="mt-auto bg-red-600 hover:bg-red-700 text-white flex items-center gap-2 button-relief"
        >
          <LogOut className="h-4 w-4" />
          Logout
        </Button>
      </aside>

      {/* Main Content con sezioni in rilievo */}
      <main className="flex-1 p-10 space-y-10">
        {/* Header Section */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-slate-100 mb-2">Code Analysis</h1>
              <p className="text-slate-300 text-lg">
                Analyze your Python ML code for anti-patterns
              </p>
            </div>
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-lg rounded-2xl flex items-center gap-3 font-bold button-relief"
            >
              <Play className="h-5 w-5" />
              {isAnalyzing ? 'Analyzing...' : 'Analyze Code'}
            </Button>
          </div>
        </div>

        {/* Feedback Message */}
        {feedbackMessage && (
          <div className="section-elevated p-4">
            <div className="flex items-center gap-3 text-green-400">
              <div className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
                <span className="text-white text-xs">✓</span>
              </div>
              <span>{feedbackMessage}</span>
            </div>
          </div>
        )}

        {/* Code Editor Section */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-slate-100">Python Code Editor</h2>
            <div className="flex items-center gap-3 text-slate-400 text-sm">
              <span>Language: Python</span>
              <span>•</span>
              <span>ML Focus</span>
            </div>
          </div>

          <div className="code-editor-relief p-6">
            <textarea
              value={code}
              onChange={e => setCode(e.target.value)}
              className="w-full h-80 bg-transparent text-slate-100 font-mono text-sm resize-none focus:outline-none placeholder:text-slate-400"
              placeholder="# Paste your Python ML code here...
# Example:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Your ML code here..."
            />
          </div>

          <div className="flex items-center justify-between mt-4 text-slate-400 text-sm">
            <span>
              {code.split('\n').length} lines • {code.length} characters
            </span>
            <span>Press Ctrl+A to select all</span>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="section-elevated p-6">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-6 w-6 text-red-500 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-red-500 font-bold text-lg">Analysis Failed</p>
                <p className="text-slate-300">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-500 hover:text-red-400 text-2xl font-bold px-3"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isAnalyzing && (
          <div className="section-elevated p-12 text-center">
            <div className="relative mb-8">
              <div className="w-16 h-16 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin mx-auto"></div>
              <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-r-green-500 rounded-full animate-pulse mx-auto"></div>
            </div>
            <h3 className="text-slate-100 text-2xl font-bold mb-2">
              Analyzing your code...
            </h3>
            <p className="text-slate-300 text-lg mb-4">
              Detecting ML anti-patterns and quality issues
            </p>
            <div className="max-w-md mx-auto">
              <div className="bg-slate-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-sky-500 h-full rounded-full animate-pulse"
                  style={{ width: '60%' }}
                ></div>
              </div>
              <p className="text-slate-400 text-sm mt-3">
                This may take up to 90 seconds
              </p>
            </div>
          </div>
        )}

        {/* Analysis Results - When patterns found */}
        {patterns.length > 0 && (
          <div className="section-elevated p-8">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h2 className="text-3xl font-bold text-slate-100 mb-2">
                  Detected Patterns
                </h2>
                <p className="text-slate-300">
                  Found {patterns.length} potential issue
                  {patterns.length !== 1 ? 's' : ''} in your code
                </p>
              </div>
              <div className="text-right">
                <div className="text-slate-400 text-sm">Analysis completed</div>
                <div className="text-slate-300 font-medium">Just now</div>
              </div>
            </div>

            <div className="grid gap-8">
              {patterns.map((pattern, index) => {
                const severityClass =
                  {
                    critical: 'pattern-card-critical',
                    high: 'pattern-card-high',
                    medium: 'pattern-card-medium',
                    low: 'pattern-card-low',
                  }[pattern.severity] || 'pattern-card-medium'

                return (
                  <div
                    key={pattern.id}
                    className={`${severityClass} p-8 hover:scale-102 transition-all duration-300`}
                  >
                    <div className="flex items-start justify-between mb-6">
                      <div>
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-slate-400 text-sm font-medium">
                            #{index + 1}
                          </span>
                          <h3 className="text-slate-100 text-2xl font-bold">
                            {pattern.type
                              .replace('_', ' ')
                              .replace(/\b\w/g, l => l.toUpperCase())}
                          </h3>
                        </div>
                        <div className="flex items-center gap-3">
                          <div
                            className={`w-3 h-3 rounded-full ${
                              pattern.severity === 'critical'
                                ? 'bg-red-500'
                                : pattern.severity === 'high'
                                  ? 'bg-orange-500'
                                  : pattern.severity === 'medium'
                                    ? 'bg-yellow-500'
                                    : 'bg-green-500'
                            }`}
                          ></div>
                          <span className="text-slate-400 text-sm uppercase tracking-wide">
                            {pattern.severity} Priority
                          </span>
                          {pattern.confidence && (
                            <>
                              <span className="text-slate-400">•</span>
                              <span className="text-slate-300 text-sm font-medium">
                                {pattern.confidence}% confidence
                              </span>
                            </>
                          )}
                        </div>
                      </div>

                      {pattern.location && (
                        <div className="text-slate-400 text-sm text-right">
                          <div className="font-mono">{pattern.location.file}</div>
                          <div>Line {pattern.location.startLine}</div>
                        </div>
                      )}
                    </div>

                    <p className="text-slate-300 text-lg leading-relaxed mb-6">
                      {pattern.explanation}
                    </p>

                    {pattern.context?.codeSnippet && (
                      <div className="code-editor-relief p-4 mb-6">
                        <div className="text-slate-400 text-xs mb-2 uppercase tracking-wide">
                          Code Snippet
                        </div>
                        <code className="text-sky-400 font-mono text-sm">
                          {pattern.context.codeSnippet}
                        </code>
                      </div>
                    )}

                    {pattern.suggestedFix && (
                      <div className="card-relief p-6 border-green-500/20">
                        <div className="flex items-start gap-3">
                          <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                            <span className="text-white text-xs font-bold">💡</span>
                          </div>
                          <div className="flex-1">
                            <p className="text-green-500 font-bold text-lg mb-2">
                              Suggested Fix:
                            </p>
                            <p className="text-slate-300 leading-relaxed mb-4">
                              {pattern.suggestedFix}
                            </p>

                            {pattern.context?.fixSnippet && (
                              <div className="code-editor-relief p-4">
                                <div className="text-green-400 text-xs mb-2 uppercase tracking-wide">
                                  Corrected Code
                                </div>
                                <code className="text-green-400 font-mono text-sm whitespace-pre-wrap">
                                  {pattern.context.fixSnippet}
                                </code>
                              </div>
                            )}

                            {pattern.references && pattern.references.length > 0 && (
                              <div className="mt-4">
                                <p className="text-slate-400 text-sm mb-2">
                                  Learn more:
                                </p>
                                <div className="space-y-1">
                                  {pattern.references.map(
                                    (ref: string, idx: number) => (
                                      <a
                                        key={idx}
                                        href={ref}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-sky-400 hover:text-sky-300 text-sm block underline"
                                      >
                                        {ref}
                                      </a>
                                    )
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}

                    {handleFeedback && (
                      <div className="flex items-center justify-end gap-4 mt-6 pt-6 border-t border-slate-700">
                        {feedbackSubmitted.has(pattern.id) ? (
                          <div className="flex items-center gap-2 text-green-400">
                            <div className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
                              <span className="text-white text-xs">✓</span>
                            </div>
                            <span className="text-sm">Feedback submitted</span>
                          </div>
                        ) : (
                          <>
                            <span className="text-slate-400 text-sm">
                              Was this helpful?
                            </span>
                            <div className="flex items-center gap-2">
                              <button
                                onClick={() =>
                                  handleFeedback({
                                    pattern_id: pattern.id,
                                    feedback_type: 'helpful',
                                    timestamp: new Date().toISOString(),
                                  })
                                }
                                className="card-relief px-4 py-2 text-green-500 hover:text-green-400 hover:border-green-500/30 hover:bg-green-500/10 transition-all"
                              >
                                👍 Yes
                              </button>
                              <button
                                onClick={() =>
                                  handleFeedback({
                                    pattern_id: pattern.id,
                                    feedback_type: 'not_helpful',
                                    timestamp: new Date().toISOString(),
                                  })
                                }
                                className="card-relief px-4 py-2 text-slate-400 hover:text-red-500 hover:border-red-500/30 hover:bg-red-500/10 transition-all"
                              >
                                👎 No
                              </button>
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Analysis Results - When no patterns found */}
        {!isAnalyzing &&
          patterns.length === 0 &&
          hasAnalyzed &&
          (() => {
            const codeContext = analyzeCodeContext(code)
            return (
              <div className="section-elevated p-8">
                <div className="text-center">
                  <div className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                    <CheckCircle className="h-10 w-10 text-green-500" />
                  </div>

                  <h2 className="text-3xl font-bold text-slate-100 mb-4">
                    Analysis Complete - No ML Issues Found
                  </h2>

                  <div className="max-w-3xl mx-auto space-y-6">
                    <p className="text-slate-300 text-lg">
                      {codeContext.isMLCode
                        ? 'Excellent! Your ML code follows best practices and no anti-patterns were detected.'
                        : 'Analysis completed successfully. This appears to be non-ML code, which explains why no ML-specific issues were found.'}
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 mt-8">
                      <div className="card-relief p-6">
                        <div className="flex items-center gap-3 mb-4">
                          <Brain className="h-6 w-6 text-blue-500" />
                          <h3 className="text-lg font-semibold text-slate-100">
                            Code Analysis Results
                          </h3>
                        </div>
                        <div className="space-y-3 text-sm">
                          <div>
                            <span className="text-slate-400">Code Type:</span>
                            <div className="text-slate-300 mt-1">
                              {codeContext.codeTypes.map((type, index) => (
                                <span
                                  key={type}
                                  className="inline-block bg-slate-700 px-2 py-1 rounded mr-2 mb-1"
                                >
                                  {type}
                                </span>
                              ))}
                            </div>
                          </div>

                          {codeContext.detectedLibraries.length > 0 && (
                            <div>
                              <span className="text-slate-400">Libraries Found:</span>
                              <div className="text-slate-300 mt-1">
                                {codeContext.detectedLibraries.map((lib, index) => (
                                  <span
                                    key={lib}
                                    className="inline-block bg-blue-700/30 text-blue-300 px-2 py-1 rounded mr-2 mb-1"
                                  >
                                    {lib}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {codeContext.mlFeatures.length > 0 && (
                            <div>
                              <span className="text-slate-400">ML Operations:</span>
                              <div className="text-slate-300 mt-1">
                                {codeContext.mlFeatures.map((feature, index) => (
                                  <span
                                    key={feature}
                                    className="inline-block bg-green-700/30 text-green-300 px-2 py-1 rounded mr-2 mb-1"
                                  >
                                    {feature}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="card-relief p-6">
                        <div className="flex items-center gap-3 mb-4">
                          <FileText className="h-6 w-6 text-yellow-500" />
                          <h3 className="text-lg font-semibold text-slate-100">
                            Analysis Explanation
                          </h3>
                        </div>
                        <ul className="text-slate-300 space-y-2 text-sm">
                          {codeContext.isMLCode ? (
                            <>
                              <li>• ✅ ML libraries properly imported</li>
                              <li>• ✅ No data leakage patterns detected</li>
                              <li>• ✅ No obvious anti-patterns found</li>
                              <li>• ✅ Code follows ML best practices</li>
                            </>
                          ) : (
                            <>
                              <li>• ❌ No ML libraries detected</li>
                              <li>• ❌ No machine learning operations found</li>
                              {codeContext.isWebApp && (
                                <li>• 🌐 Web application detected</li>
                              )}
                              {codeContext.hasDatabase && (
                                <li>• 🗄️ Database operations found</li>
                              )}
                              <li>
                                • ℹ️ This is{' '}
                                {codeContext.codeTypes.join(', ').toLowerCase()} code
                              </li>
                            </>
                          )}
                        </ul>
                      </div>
                    </div>

                    <div className="bg-slate-800/50 rounded-xl p-6 mt-8">
                      <div className="flex items-start gap-4">
                        <div className="w-8 h-8 bg-sky-500/20 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                          <Lightbulb className="h-4 w-4 text-sky-500" />
                        </div>
                        <div>
                          <h4 className="text-slate-100 font-semibold mb-2">
                            Want to Test the Analyzer?
                          </h4>
                          <p className="text-slate-300 text-sm mb-4">
                            Try pasting ML code with common issues to see Attrahere in
                            action:
                          </p>
                          <div className="bg-slate-900 rounded-lg p-4 font-mono text-xs text-slate-300 overflow-x-auto">
                            <div className="text-slate-500 mb-2">
                              # Example ML code with issues:
                            </div>
                            <div className="text-blue-400">
                              from sklearn.model_selection import train_test_split
                            </div>
                            <div className="text-blue-400">
                              from sklearn.preprocessing import StandardScaler
                            </div>
                            <div className="text-green-400">
                              <br />
                              # ❌ Data leakage: preprocessing before split
                              <br />
                              scaler = StandardScaler()
                              <br />
                              X_scaled = scaler.fit_transform(X)
                              <br />
                              <br />
                              # ❌ Missing random_state
                              <br />
                              X_train, X_test = train_test_split(X_scaled, y)
                            </div>
                          </div>
                          <button
                            onClick={() =>
                              setCode(`from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# ❌ Data leakage: preprocessing before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ❌ Missing random_state (not reproducible!)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# ❌ Magic numbers without explanation
model = RandomForestClassifier(n_estimators=73, max_depth=15)
model.fit(X_train, y_train)`)
                            }
                            className="mt-4 px-4 py-2 bg-sky-500 hover:bg-sky-600 text-white rounded-lg text-sm transition-colors"
                          >
                            Load Example Code
                          </button>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-center gap-6 mt-8 pt-6 border-t border-slate-700">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-slate-100">
                          {code.split('\n').length}
                        </div>
                        <div className="text-slate-400 text-sm">Lines Analyzed</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-slate-100">
                          {code.length}
                        </div>
                        <div className="text-slate-400 text-sm">Characters</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-500">0</div>
                        <div className="text-slate-400 text-sm">Issues Found</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )
          })()}
      </main>
    </div>
  )
}
