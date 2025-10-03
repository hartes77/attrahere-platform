/**
 * Core types for ML Code Quality Platform
 * These match the backend API schemas for consistency
 */

// Pattern detection types
export type PatternSeverity = 'low' | 'medium' | 'high' | 'critical'
export type PatternType =
  | 'data_leakage'
  | 'gpu_memory'
  | 'magic_numbers'
  | 'reproducibility'
  | 'model_architecture'
  | 'feature_engineering'
  | 'evaluation_metrics'

// Code location information
export interface CodeLocation {
  file: string
  startLine: number
  endLine: number
  startColumn: number
  endColumn: number
}

// ML Pattern detected by analysis
export interface MLPattern {
  id: string
  type: PatternType
  severity: PatternSeverity
  location: CodeLocation
  message: string
  explanation: string
  suggestedFix?: string
  confidence: number // 0-100 for RLHF
  context?: {
    codeSnippet: string
    fixSnippet?: string
    surroundingLines?: string[]
  }
  references?: string[]
}

// Analysis result from backend
export interface AnalysisResult {
  id: string
  file_path: string
  patterns: MLPattern[]
  summary: {
    total_patterns: number
    severity_counts: Record<PatternSeverity, number>
    overall_score: number
    analysis_duration_ms: number
  }
  metadata?: {
    ml_constructs_found: string[]
    functions_analyzed: number
    classes_analyzed: number
    lines_of_code: number
  }
  async_processing?: boolean
  task_id?: string
  status_url?: string
}

// RLHF Feedback types
export type FeedbackType =
  | 'helpful'
  | 'not_helpful'
  | 'false_positive'
  | 'false_negative'
  | 'suggestion'

export interface UserFeedback {
  id?: string
  pattern_id: string
  feedback_type: FeedbackType
  accuracy_rating?: number // 1-5
  usefulness_rating?: number // 1-5
  user_comment?: string
  suggested_fix?: string
  timestamp?: string
}

// API request/response types
export interface AnalysisRequest {
  code: string
  file_path: string
  options?: {
    pattern_types?: PatternType[]
    severity_filter?: PatternSeverity[]
    include_suggestions?: boolean
    async_processing?: boolean
  }
}

export interface APIResponse<T> {
  success: boolean
  data?: T
  error?: {
    message: string
    code: string
    details?: unknown
  }
  meta?: {
    request_id: string
    timestamp: string
    processing_time_ms: number
  }
}

// Fix application types
export interface FixSuggestion {
  id: string
  pattern_id: string
  type: 'replace' | 'insert' | 'delete' | 'refactor'
  description: string
  original_code: string
  fixed_code: string
  confidence: number
  impact_assessment?: {
    breaking_changes: boolean
    performance_impact: 'positive' | 'negative' | 'neutral'
    readability_impact: 'improved' | 'degraded' | 'neutral'
  }
}

export interface ApplyFixRequest {
  pattern_id: string
  fix_id: string
  preview_only?: boolean
}

export interface ApplyFixResult {
  success: boolean
  changes_applied: number
  backup_created?: string
  warnings?: string[]
  errors?: string[]
}

// Dashboard and analytics types
export interface ProjectStats {
  total_files_analyzed: number
  total_patterns_found: number
  severity_distribution: Record<PatternSeverity, number>
  most_common_patterns: Array<{
    type: PatternType
    count: number
    avg_confidence: number
  }>
  improvement_over_time?: Array<{
    date: string
    pattern_count: number
    overall_score: number
  }>
}

// User session and preferences
export interface UserPreferences {
  theme: 'dark' | 'light' | 'system'
  editor_settings: {
    font_size: number
    font_family: string
    tab_size: number
    show_line_numbers: boolean
  }
  analysis_settings: {
    auto_analyze: boolean
    severity_threshold: PatternSeverity
    preferred_fix_types: PatternType[]
  }
  notifications: {
    analysis_complete: boolean
    new_patterns: boolean
    fix_suggestions: boolean
  }
}

// WebSocket/Real-time types
export interface RealtimeAnalysisEvent {
  type: 'analysis_start' | 'analysis_progress' | 'analysis_complete' | 'analysis_error'
  data: {
    analysis_id: string
    progress?: number // 0-100
    patterns_found?: number
    result?: AnalysisResult
    error?: string
  }
  timestamp: string
}

// Component prop types for reusability
export interface PatternCardProps {
  pattern: MLPattern
  showActions?: boolean
  onFeedback?: (feedback: UserFeedback) => void
  onApplyFix?: (fixId: string) => void
  className?: string
}

export interface CodeViewerProps {
  code: string
  language?: string
  patterns?: MLPattern[]
  highlightedLines?: number[]
  onPatternClick?: (pattern: MLPattern) => void
  readOnly?: boolean
  className?: string
}
