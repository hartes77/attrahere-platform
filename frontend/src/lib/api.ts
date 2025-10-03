// lib/api.ts - Legacy API client (placeholder)
// This is a placeholder file to fix import errors
// The actual analytics API client is in analytics-api.ts

export const apiClient = {
  // Legacy placeholder methods
  get: async (url: string) => {
    console.warn('Using legacy API client placeholder')
    return Promise.resolve({ data: null })
  },
  
  isAuthenticated: () => {
    console.warn('Using legacy authentication check')
    return false // For now, always return false
  },
  
  getMe: async () => {
    console.warn('Using legacy getMe')
    return Promise.resolve({ 
      id: 'demo-user',
      name: 'Demo User', 
      email: 'demo@attrahere.com' 
    })
  },
  
  logout: () => {
    console.warn('Using legacy logout')
    // Placeholder logout
  }
}