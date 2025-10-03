// lib/api.ts - Legacy API client (placeholder)
// This is a placeholder file to fix import errors
// The actual analytics API client is in analytics-api.ts

// TypeScript placeholder types
export interface AuditLog {
  id: string
  timestamp: string
  action: string
  user: string
  username: string
  resource: string
  ip_address: string
  details: Record<string, any>
}

export interface User {
  id: string
  name: string
  username: string
  email: string
  role: string
  is_active: boolean
  created_at: string
  last_login?: string
}

export interface UserCreate {
  username: string
  email: string
  role: string
  password: string
}

export const apiClient = {
  // Legacy placeholder methods
  get: async (url: string) => {
    console.warn('Using legacy API client placeholder')
    return Promise.resolve({ data: null })
  },
  
  post: async (url: string, data: any) => {
    console.warn('Using legacy API client POST placeholder')
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
      email: 'demo@attrahere.com',
      role: 'admin' // Default to admin for development
    })
  },
  
  logout: () => {
    console.warn('Using legacy logout')
    // Placeholder logout
  },
  
  getAuditLogs: async (limit: number = 50) => {
    console.warn('Using legacy getAuditLogs')
    // Return mock audit logs for development
    const mockLogs: AuditLog[] = [
      {
        id: '1',
        timestamp: new Date().toISOString(),
        action: 'login',
        user: 'demo@attrahere.com',
        username: 'demo',
        resource: '/admin/dashboard',
        ip_address: '127.0.0.1',
        details: { message: 'User logged in successfully', success: true }
      }
    ]
    return Promise.resolve(mockLogs.slice(0, limit))
  },
  
  listUsers: async () => {
    console.warn('Using legacy listUsers')
    const mockUsers: User[] = [
      {
        id: '1',
        name: 'Demo User',
        username: 'demo',
        email: 'demo@attrahere.com',
        role: 'admin',
        is_active: true,
        created_at: new Date().toISOString(),
        last_login: new Date().toISOString()
      }
    ]
    return Promise.resolve(mockUsers)
  },
  
  createUser: async (userData: UserCreate) => {
    console.warn('Using legacy createUser')
    const newUser: User = {
      id: Math.random().toString(36),
      name: userData.username,
      username: userData.username,
      email: userData.email,
      role: userData.role,
      is_active: true,
      created_at: new Date().toISOString()
    }
    return Promise.resolve(newUser)
  },
  
  deleteUser: async (userId: string) => {
    console.warn('Using legacy deleteUser')
    return Promise.resolve({ success: true })
  }
}