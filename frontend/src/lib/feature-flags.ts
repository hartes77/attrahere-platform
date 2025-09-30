/**
 * Enterprise Feature Flags System - Frontend
 * Professional feature toggle management for ML Code Quality Platform
 */

import React from 'react'

export interface FeatureFlags {
  // Core Platform Features
  authenticationSystem: boolean
  mlAnalysisEngine: boolean
  projectWideAnalysis: boolean

  // Enterprise Features (All 4/4 completed!)
  auditLogging: boolean
  userManagement: boolean
  rateLimiting: boolean
  securityHeaders: boolean

  // Advanced AI Features
  gemmaAiIntegration: boolean
  conversationalAnalysis: boolean
  naturalLanguageQueries: boolean
  multiTurnConversations: boolean

  // Demo Mode Controls
  demoMode: boolean
  showIncompleteFeatures: boolean
  enableExperimentalPatterns: boolean
  showBetaBadges: boolean

  // Advanced Analysis Features
  advancedPatternDetection: boolean
  crossFileAnalysis: boolean
  aiPoweredSuggestions: boolean
  customRuleCreation: boolean

  // UI/UX Features
  darkTheme: boolean
  realTimeUpdates: boolean
  threeDReliefEffects: boolean
  accessibilityMode: boolean

  // Export & Reporting
  pdfExport: boolean
  excelExport: boolean
  customReports: boolean
  apiAnalytics: boolean

  // Integration Features
  githubIntegration: boolean
  vscodeExtension: boolean
  ciCdIntegration: boolean
  slackNotifications: boolean

  // Performance Features
  cachingEnabled: boolean
  parallelProcessing: boolean
  edgeTpuAcceleration: boolean

  // Educational Features
  teachingMode: boolean
  multiAudienceExplanations: boolean
  interactiveTutorials: boolean
  certificationTracking: boolean
}

// Default feature flags configuration
const DEFAULT_FLAGS: FeatureFlags = {
  // Core Platform Features
  authenticationSystem: true,
  mlAnalysisEngine: true,
  projectWideAnalysis: true,

  // Enterprise Features (All 4/4 completed!)
  auditLogging: true,
  userManagement: true,
  rateLimiting: true,
  securityHeaders: true,

  // Advanced AI Features
  gemmaAiIntegration: true,
  conversationalAnalysis: true,
  naturalLanguageQueries: true,
  multiTurnConversations: false, // Coming in Phase 2

  // Demo Mode Controls
  demoMode: false,
  showIncompleteFeatures: true,
  enableExperimentalPatterns: true,
  showBetaBadges: true,

  // Advanced Analysis Features
  advancedPatternDetection: true,
  crossFileAnalysis: true,
  aiPoweredSuggestions: true,
  customRuleCreation: false, // Coming soon

  // UI/UX Features
  darkTheme: true,
  realTimeUpdates: true,
  threeDReliefEffects: true,
  accessibilityMode: false,

  // Export & Reporting
  pdfExport: false, // Phase 3 feature
  excelExport: false, // Phase 3 feature
  customReports: false, // Enterprise feature
  apiAnalytics: true,

  // Integration Features
  githubIntegration: false, // Phase 5
  vscodeExtension: false, // Phase 5
  ciCdIntegration: false, // Phase 5
  slackNotifications: false, // Future

  // Performance Features
  cachingEnabled: true,
  parallelProcessing: true,
  edgeTpuAcceleration: false, // Hardware dependent

  // Educational Features
  teachingMode: true,
  multiAudienceExplanations: true,
  interactiveTutorials: false, // Future
  certificationTracking: false, // Educational partnership
}

export type FeatureEnvironment = 'demo' | 'development' | 'staging' | 'production'

class FeatureFlagManager {
  private flags: FeatureFlags
  private environment: FeatureEnvironment
  private apiUrl: string

  constructor(environment: FeatureEnvironment = 'development') {
    this.environment = environment
    this.apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    this.flags = this.initializeFlags()
  }

  private initializeFlags(): FeatureFlags {
    let flags = { ...DEFAULT_FLAGS }

    // Environment-specific overrides
    if (this.environment === 'demo') {
      flags = {
        ...flags,
        demoMode: true,
        showIncompleteFeatures: false,
        enableExperimentalPatterns: false,
        showBetaBadges: false,
        customRuleCreation: false,
        pdfExport: false,
        excelExport: false,
        customReports: false,
        githubIntegration: false,
        vscodeExtension: false,
        ciCdIntegration: false,
        interactiveTutorials: false,
        certificationTracking: false,
      }
    } else if (this.environment === 'production') {
      flags = {
        ...flags,
        demoMode: false,
        enableExperimentalPatterns: false,
        showBetaBadges: false,
        edgeTpuAcceleration: false,
      }
    }

    return flags
  }

  public isEnabled(featureName: keyof FeatureFlags): boolean {
    return this.flags[featureName]
  }

  public getFlag<K extends keyof FeatureFlags>(
    featureName: K,
    defaultValue?: FeatureFlags[K]
  ): FeatureFlags[K] {
    return this.flags[featureName] ?? defaultValue ?? false
  }

  public async syncWithBackend(): Promise<void> {
    try {
      const response = await fetch(`${this.apiUrl}/api/v1/feature-flags`)
      if (response.ok) {
        const backendFlags = await response.json()
        this.updateFlags(backendFlags)
      }
    } catch (error) {
      console.warn('Could not sync feature flags with backend:', error)
    }
  }

  private updateFlags(backendFlags: Record<string, unknown>): void {
    // Convert snake_case backend flags to camelCase frontend flags
    const camelCaseFlags: Partial<FeatureFlags> = {}

    for (const [key, value] of Object.entries(backendFlags)) {
      const camelCaseKey = this.snakeToCamel(key) as keyof FeatureFlags
      if (camelCaseKey in DEFAULT_FLAGS) {
        ;(camelCaseFlags as Record<string, unknown>)[camelCaseKey] = value
      }
    }

    this.flags = { ...this.flags, ...camelCaseFlags }
  }

  private snakeToCamel(str: string): string {
    return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())
  }

  public getAllFlags(): FeatureFlags {
    return { ...this.flags }
  }

  public getDemoConfig(): FeatureFlags {
    return {
      ...this.flags,
      demoMode: true,
      showIncompleteFeatures: false,
      enableExperimentalPatterns: false,
      showBetaBadges: false,
    }
  }

  public getProductionConfig(): FeatureFlags {
    return {
      ...this.flags,
      demoMode: false,
      enableExperimentalPatterns: false,
      edgeTpuAcceleration: false,
    }
  }
}

// Environment detection
function getEnvironment(): FeatureEnvironment {
  if (typeof window === 'undefined') return 'development'

  const hostname = window.location.hostname
  if (hostname.includes('demo')) return 'demo'
  if (hostname.includes('staging')) return 'staging'
  if (hostname.includes('localhost') || hostname.includes('127.0.0.1'))
    return 'development'
  return 'production'
}

// Global feature flag manager instance
let featureFlagManager: FeatureFlagManager | null = null

export function getFeatureFlagManager(): FeatureFlagManager {
  if (!featureFlagManager) {
    const environment = getEnvironment()
    featureFlagManager = new FeatureFlagManager(environment)
  }
  return featureFlagManager
}

// Convenience functions for easy access
export function isEnabled(featureName: keyof FeatureFlags): boolean {
  return getFeatureFlagManager().isEnabled(featureName)
}

export function getFlag<K extends keyof FeatureFlags>(
  featureName: K,
  defaultValue?: FeatureFlags[K]
): FeatureFlags[K] {
  return getFeatureFlagManager().getFlag(featureName, defaultValue)
}

// React Hook for feature flags
export function useFeatureFlags(): {
  flags: FeatureFlags
  isEnabled: (featureName: keyof FeatureFlags) => boolean
  getFlag: <K extends keyof FeatureFlags>(
    featureName: K,
    defaultValue?: FeatureFlags[K]
  ) => FeatureFlags[K]
  syncWithBackend: () => Promise<void>
} {
  const manager = getFeatureFlagManager()

  return {
    flags: manager.getAllFlags(),
    isEnabled: featureName => manager.isEnabled(featureName),
    getFlag: (featureName, defaultValue) => manager.getFlag(featureName, defaultValue),
    syncWithBackend: () => manager.syncWithBackend(),
  }
}

// Feature flag component wrappers
interface FeatureGateProps {
  feature: keyof FeatureFlags
  fallback?: React.ReactNode
  children: React.ReactNode
}

export function FeatureGate({ feature, fallback = null, children }: FeatureGateProps) {
  const enabled = isEnabled(feature)
  return enabled
    ? React.createElement(React.Fragment, null, children)
    : React.createElement(React.Fragment, null, fallback)
}

interface ConditionalFeatureProps {
  feature: keyof FeatureFlags
  enabled?: React.ReactNode
  disabled?: React.ReactNode
}

export function ConditionalFeature({
  feature,
  enabled = null,
  disabled = null,
}: ConditionalFeatureProps) {
  const isFeatureEnabled = isEnabled(feature)
  return isFeatureEnabled
    ? React.createElement(React.Fragment, null, enabled)
    : React.createElement(React.Fragment, null, disabled)
}

// Utility functions for demo mode
export function isDemoMode(): boolean {
  return isEnabled('demoMode')
}

export function shouldShowBetaFeatures(): boolean {
  return isEnabled('showBetaBadges') && !isDemoMode()
}

export function shouldShowIncompleteFeatures(): boolean {
  return isEnabled('showIncompleteFeatures') && !isDemoMode()
}

// Export default configuration for external access
export { DEFAULT_FLAGS }

export default getFeatureFlagManager
