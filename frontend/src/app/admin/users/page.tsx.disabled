'use client'

import React, { useState, useEffect } from 'react'
import { apiClient, User, UserCreate } from '@/lib/api'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import {
  BarChart3,
  FileText,
  Settings,
  LogOut,
  Users,
  Plus,
  Trash2,
  Shield,
  AlertCircle,
  User as UserIcon,
  Mail,
  Calendar,
  Brain,
} from 'lucide-react'

export default function UsersManagementPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [isAdmin, setIsAdmin] = useState(false)
  const [users, setUsers] = useState<User[]>([])
  const [loadingUsers, setLoadingUsers] = useState(false)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Form state
  const [formData, setFormData] = useState<UserCreate>({
    username: '',
    email: '',
    password: '',
    role: 'researcher',
  })

  useEffect(() => {
    const checkAuth = async () => {
      if (!apiClient.isAuthenticated()) {
        setIsAuthenticated(false)
        setIsLoading(false)
        return
      }

      try {
        const currentUser = await apiClient.getMe()
        setIsAuthenticated(true)
        setIsAdmin(currentUser.role === 'admin')

        if (currentUser.role === 'admin') {
          await loadUsers()
        }
      } catch (error) {
        console.error('Auth check failed:', error)
        setIsAuthenticated(false)
      }
      setIsLoading(false)
    }

    checkAuth()
  }, [])

  const loadUsers = async () => {
    setLoadingUsers(true)
    try {
      const usersData = await apiClient.listUsers()
      setUsers(usersData)
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to load users')
    }
    setLoadingUsers(false)
  }

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess(null)

    try {
      const newUser = await apiClient.createUser(formData)
      setUsers([...users, newUser])
      setSuccess(`User ${newUser.username} created successfully`)
      setShowCreateForm(false)
      setFormData({
        username: '',
        email: '',
        password: '',
        role: 'researcher',
      })
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to create user')
    }
  }

  const handleDeleteUser = async (userId: string, username: string) => {
    if (!confirm(`Are you sure you want to delete user "${username}"?`)) {
      return
    }

    try {
      await apiClient.deleteUser(userId)
      setUsers(users.filter(u => u.id !== userId))
      setSuccess(`User ${username} deleted successfully`)
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to delete user')
    }
  }

  const handleLogout = async () => {
    await apiClient.logout()
    setIsAuthenticated(false)
  }

  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case 'admin':
        return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'researcher':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'developer':
        return 'bg-green-500/20 text-green-400 border-green-500/30'
      default:
        return 'bg-slate-500/20 text-slate-400 border-slate-500/30'
    }
  }

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'admin':
        return <Shield className="h-4 w-4" />
      case 'researcher':
        return <FileText className="h-4 w-4" />
      case 'developer':
        return <Users className="h-4 w-4" />
      default:
        return <UserIcon className="h-4 w-4" />
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-300">Loading...</p>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
        <div className="text-center max-w-2xl">
          <Shield className="h-24 w-24 text-sky-500 mx-auto mb-8" />
          <h1 className="text-4xl font-bold text-sky-500 mb-6">Access Required</h1>
          <p className="text-slate-300 text-lg mb-8">
            Please log in to access the user management dashboard
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

  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
        <div className="text-center max-w-2xl">
          <AlertCircle className="h-24 w-24 text-red-500 mx-auto mb-8" />
          <h1 className="text-4xl font-bold text-red-500 mb-6">Access Denied</h1>
          <p className="text-slate-300 text-lg mb-8">
            Only administrators can access user management
          </p>
          <Link href="/">
            <Button className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-lg rounded-xl">
              Back to Dashboard
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-slate-900 text-slate-100">
      {/* Sidebar */}
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
          <Link href="/analyze" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <UserIcon className="h-5 w-5" />
                <span>Analyze</span>
              </div>
            </div>
          </Link>
          <div className="card-relief-strong text-sky-500 px-5 py-4 rounded-xl border-sky-500/30">
            <div className="flex items-center gap-3">
              <Users className="h-5 w-5" />
              <span className="font-semibold">Users</span>
            </div>
          </div>
          <Link href="/admin/analytics" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200 bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
              <div className="flex items-center gap-3">
                <Brain className="h-5 w-5 text-purple-400" />
                <span className="text-purple-300">AI Analytics</span>
              </div>
            </div>
          </Link>
          <Link href="/admin/audit" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <FileText className="h-5 w-5" />
                <span>Audit Logs</span>
              </div>
            </div>
          </Link>
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

      {/* Main Content */}
      <main className="flex-1 p-10 space-y-10">
        {/* Header Section */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-slate-100 mb-2">
                User Management
              </h1>
              <p className="text-slate-300 text-lg">
                Manage users, roles, and permissions
              </p>
            </div>
            <Button
              onClick={() => setShowCreateForm(true)}
              className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-lg rounded-2xl flex items-center gap-3 font-bold button-relief"
            >
              <Plus className="h-5 w-5" />
              Add User
            </Button>
          </div>
        </div>

        {/* Messages */}
        {error && (
          <div className="section-elevated p-6">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-6 w-6 text-red-500 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-red-500 font-bold text-lg">Error</p>
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

        {success && (
          <div className="section-elevated p-6">
            <div className="flex items-center gap-3">
              <Shield className="h-6 w-6 text-green-500 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-green-500 font-bold text-lg">Success</p>
                <p className="text-slate-300">{success}</p>
              </div>
              <button
                onClick={() => setSuccess(null)}
                className="text-green-500 hover:text-green-400 text-2xl font-bold px-3"
              >
                ×
              </button>
            </div>
          </div>
        )}

        {/* Create User Form */}
        {showCreateForm && (
          <div className="section-elevated p-8">
            <h2 className="text-2xl font-bold text-slate-100 mb-6">Create New User</h2>
            <form onSubmit={handleCreateUser} className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-slate-300 text-sm font-medium mb-2">
                    Username
                  </label>
                  <input
                    type="text"
                    value={formData.username}
                    onChange={e =>
                      setFormData({ ...formData, username: e.target.value })
                    }
                    className="w-full px-4 py-3 code-editor-relief bg-slate-800 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 transition-all"
                    placeholder="Enter username"
                    required
                  />
                </div>
                <div>
                  <label className="block text-slate-300 text-sm font-medium mb-2">
                    Email
                  </label>
                  <input
                    type="email"
                    value={formData.email}
                    onChange={e => setFormData({ ...formData, email: e.target.value })}
                    className="w-full px-4 py-3 code-editor-relief bg-slate-800 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 transition-all"
                    placeholder="Enter email"
                    required
                  />
                </div>
                <div>
                  <label className="block text-slate-300 text-sm font-medium mb-2">
                    Password
                  </label>
                  <input
                    type="password"
                    value={formData.password}
                    onChange={e =>
                      setFormData({ ...formData, password: e.target.value })
                    }
                    className="w-full px-4 py-3 code-editor-relief bg-slate-800 text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 transition-all"
                    placeholder="Enter password"
                    required
                  />
                </div>
                <div>
                  <label className="block text-slate-300 text-sm font-medium mb-2">
                    Role
                  </label>
                  <select
                    value={formData.role}
                    onChange={e =>
                      setFormData({
                        ...formData,
                        role: e.target.value as 'admin' | 'researcher' | 'developer',
                      })
                    }
                    className="w-full px-4 py-3 code-editor-relief bg-slate-800 text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-500 transition-all"
                  >
                    <option value="researcher">Researcher</option>
                    <option value="developer">Developer</option>
                    <option value="admin">Admin</option>
                  </select>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <Button
                  type="submit"
                  className="bg-sky-500 hover:bg-sky-600 text-white px-6 py-3 rounded-xl button-relief"
                >
                  Create User
                </Button>
                <Button
                  type="button"
                  onClick={() => setShowCreateForm(false)}
                  className="bg-slate-600 hover:bg-slate-700 text-white px-6 py-3 rounded-xl button-relief"
                >
                  Cancel
                </Button>
              </div>
            </form>
          </div>
        )}

        {/* Users List */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold text-slate-100 mb-2">
                Users ({users.length})
              </h2>
              <p className="text-slate-300">Manage platform users and their roles</p>
            </div>
            {loadingUsers && (
              <div className="w-6 h-6 border-2 border-sky-500 border-t-transparent rounded-full animate-spin"></div>
            )}
          </div>

          <div className="grid gap-6">
            {users.map(user => (
              <div
                key={user.id}
                className="card-relief p-6 hover:scale-102 transition-all duration-300"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-slate-700 rounded-full flex items-center justify-center">
                      {getRoleIcon(user.role)}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-slate-100 mb-1">
                        {user.username}
                      </h3>
                      <div className="flex items-center gap-2 mb-2">
                        <Mail className="h-4 w-4 text-slate-400" />
                        <span className="text-slate-400">{user.email}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span
                          className={`px-3 py-1 rounded-full text-sm font-medium border ${getRoleBadgeColor(user.role)}`}
                        >
                          {user.role.toUpperCase()}
                        </span>
                        <div className="flex items-center gap-1 text-slate-400 text-sm">
                          <Calendar className="h-3 w-3" />
                          {new Date(user.created_at).toLocaleDateString()}
                        </div>
                        <div
                          className={`text-sm ${user.is_active ? 'text-green-400' : 'text-red-400'}`}
                        >
                          {user.is_active ? 'Active' : 'Inactive'}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="text-slate-400 text-xs font-mono bg-slate-800 px-2 py-1 rounded">
                      ID: {user.id.slice(0, 8)}...
                    </div>
                    <Button
                      onClick={() => handleDeleteUser(user.id, user.username)}
                      className="bg-red-600 hover:bg-red-700 text-white p-2 rounded-lg button-relief"
                      disabled={user.role === 'admin'}
                      title={
                        user.role === 'admin'
                          ? 'Cannot delete admin users'
                          : 'Delete user'
                      }
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  )
}
