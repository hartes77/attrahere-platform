/**
 * Example Usage of Feature Flags in Frontend Components
 * ML Code Quality Platform - Enterprise Feature Management
 */

import React, { useEffect, useState } from 'react'
import {
  useFeatureFlags,
  FeatureGate,
  ConditionalFeature,
  isEnabled,
  isDemoMode,
  shouldShowBetaFeatures,
} from '@/lib/feature-flags'

// Example 1: Simple Feature Gate
export function AdvancedAnalysisButton() {
  return (
    <FeatureGate feature="advancedPatternDetection">
      <button className="btn-primary">🔍 Advanced Pattern Detection</button>
    </FeatureGate>
  )
}

// Example 2: Conditional Features with Fallback
export function ExportButtons() {
  return (
    <div className="export-section">
      <ConditionalFeature
        feature="pdfExport"
        enabled={<button className="btn-export">📄 Export PDF</button>}
        disabled={
          <button className="btn-disabled" disabled>
            📄 Export PDF (Coming Soon)
          </button>
        }
      />

      <ConditionalFeature
        feature="excelExport"
        enabled={<button className="btn-export">📊 Export Excel</button>}
        disabled={
          <span className="text-muted">Excel export available in Enterprise plan</span>
        }
      />
    </div>
  )
}

// Example 3: Demo Mode Adaptations
export function NavigationMenu() {
  const { flags } = useFeatureFlags()

  return (
    <nav className="main-nav">
      <a href="/analyze">Single File Analysis</a>

      <FeatureGate feature="projectWideAnalysis">
        <a href="/analyze-project">Project Analysis</a>
      </FeatureGate>

      <FeatureGate feature="conversationalAnalysis">
        <a href="/analyze-natural">
          🧠 AI Assistant
          {shouldShowBetaFeatures() && <span className="badge-beta">BETA</span>}
        </a>
      </FeatureGate>

      {/* Admin features only show when not in demo mode */}
      {!isDemoMode() && flags.userManagement && <a href="/admin">Admin Panel</a>}
    </nav>
  )
}

// Example 4: Complex Feature Logic
export function AnalysisConfigPanel() {
  const { flags, getFlag } = useFeatureFlags()

  const showAdvancedOptions = getFlag('enableExperimentalPatterns', false)
  const isTeachingModeEnabled = getFlag('teachingMode', true)

  return (
    <div className="config-panel">
      <h3>Analysis Configuration</h3>

      {/* Basic options always available */}
      <label>
        <input type="checkbox" />
        Enable ML Pattern Detection
      </label>

      {/* Advanced options only when feature enabled */}
      {showAdvancedOptions && (
        <details>
          <summary>🧪 Experimental Features</summary>
          <label>
            <input type="checkbox" />
            GPU Memory Analysis
          </label>
          <label>
            <input type="checkbox" />
            Advanced Data Flow Detection
          </label>
        </details>
      )}

      {/* Teaching mode explanations */}
      {isTeachingModeEnabled && (
        <div className="help-section">
          <h4>💡 What does this do?</h4>
          <p>
            ML Pattern Detection scans your code for common machine learning
            anti-patterns like data leakage, overfitting, and performance issues.
          </p>
        </div>
      )}
    </div>
  )
}

// Example 5: Dynamic Feature Flag Updates (Admin only)
export function AdminFeatureFlagPanel() {
  const { flags, syncWithBackend } = useFeatureFlags()
  const [isAdmin, setIsAdmin] = useState(false)

  // Check if user is admin (you'd get this from your auth system)
  useEffect(() => {
    // setIsAdmin(getCurrentUser()?.role === 'admin');
  }, [])

  const toggleFeature = async (featureName: string, newValue: boolean) => {
    try {
      const response = await fetch('/api/v1/feature-flags', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${getAuthToken()}`,
        },
        body: JSON.stringify({
          feature_name: featureName,
          value: newValue,
        }),
      })

      if (response.ok) {
        await syncWithBackend() // Refresh flags from backend
        alert(`Feature ${featureName} ${newValue ? 'enabled' : 'disabled'}`)
      }
    } catch (error) {
      console.error('Failed to update feature flag:', error)
    }
  }

  if (!isAdmin) return null

  return (
    <div className="admin-feature-panel">
      <h3>🔧 Feature Flag Management</h3>

      {Object.entries(flags).map(([key, value]) => (
        <div key={key} className="feature-toggle">
          <label>
            <input
              type="checkbox"
              checked={Boolean(value)}
              onChange={e => toggleFeature(key, e.target.checked)}
            />
            {key.replace(/([A-Z])/g, ' $1').toLowerCase()}
          </label>
        </div>
      ))}

      <button onClick={syncWithBackend} className="btn-refresh">
        🔄 Sync with Backend
      </button>
    </div>
  )
}

// Example 6: Demo Mode Dashboard
export function DemoDashboard() {
  const demoMode = isDemoMode()

  if (demoMode) {
    return (
      <div className="demo-banner">
        <h2>🎯 ML Code Quality Platform - Demo Mode</h2>
        <p>
          This is a demonstration of our Enterprise ML Code Quality Platform. Some
          features are disabled to focus on core capabilities.
        </p>
        <div className="demo-features">
          <FeatureGate feature="mlAnalysisEngine">✅ ML Pattern Detection</FeatureGate>
          <FeatureGate feature="gemmaAiIntegration">✅ AI-Powered Analysis</FeatureGate>
          <FeatureGate feature="authenticationSystem">
            ✅ Enterprise Security
          </FeatureGate>
          <span className="coming-soon">🚧 Advanced Reporting (Enterprise)</span>
        </div>
      </div>
    )
  }

  return (
    <div className="full-dashboard">
      {/* Full dashboard with all features */}
      <h1>ML Code Quality Platform</h1>
      {/* ... rest of dashboard */}
    </div>
  )
}

// Helper function (you'd implement this based on your auth system)
function getAuthToken(): string {
  return localStorage.getItem('auth_token') || ''
}

export default {
  AdvancedAnalysisButton,
  ExportButtons,
  NavigationMenu,
  AnalysisConfigPanel,
  AdminFeatureFlagPanel,
  DemoDashboard,
}
