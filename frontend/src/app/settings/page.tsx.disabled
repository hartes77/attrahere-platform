'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { apiClient } from '@/lib/api'
import {
  ArrowLeft,
  Settings,
  User,
  Bell,
  Shield,
  Palette,
  Save,
  Eye,
  EyeOff,
  Mail,
  Phone,
  Calendar,
  BarChart3,
  FileText,
  Code2,
  Brain,
  Users,
  LogOut,
  CheckCircle,
  AlertTriangle,
} from 'lucide-react'

interface UserSettings {
  profile: {
    username: string
    email: string
    fullName: string
    role: string
    joinDate: string
  }
  preferences: {
    notifications: boolean
    emailAlerts: boolean
    darkMode: boolean
    autoFix: boolean
    analysisAlerts: boolean
    weeklyReports: boolean
  }
  privacy: {
    shareAnalytics: boolean
    publicProfile: boolean
    dataRetention: string
  }
}

export default function SettingsPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [currentUser, setCurrentUser] = useState<any>(null)
  const [settings, setSettings] = useState<UserSettings | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')

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

        // Mock user settings (in real app, fetch from API)
        setSettings({
          profile: {
            username: user.username || 'admin',
            email: user.email || 'admin@attrahere.com',
            fullName: user.fullName || 'Administrator',
            role: user.role || 'admin',
            joinDate: '2025-01-15',
          },
          preferences: {
            notifications: true,
            emailAlerts: true,
            darkMode: true,
            autoFix: false,
            analysisAlerts: true,
            weeklyReports: false,
          },
          privacy: {
            shareAnalytics: false,
            publicProfile: false,
            dataRetention: '1year',
          },
        })
      } catch (error) {
        console.error('Auth check failed:', error)
        apiClient.logout()
        setIsAuthenticated(false)
        setCurrentUser(null)
      }
      setIsLoading(false)
    }
    checkAuth()
  }, [])

  const handleLogout = () => {
    apiClient.logout()
    setIsAuthenticated(false)
  }

  const updateSetting = (section: keyof UserSettings, key: string, value: any) => {
    if (!settings) return

    setSettings({
      ...settings,
      [section]: {
        ...settings[section],
        [key]: value,
      },
    })
  }

  const handleSaveSettings = async () => {
    setIsSaving(true)
    try {
      // In real app, send to API
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate API call
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch (error) {
      console.error('Failed to save settings:', error)
    }
    setIsSaving(false)
  }

  const handlePasswordChange = async () => {
    if (newPassword !== confirmPassword) {
      alert('Passwords do not match')
      return
    }
    if (newPassword.length < 6) {
      alert('Password must be at least 6 characters')
      return
    }

    try {
      // In real app, send to API
      await new Promise(resolve => setTimeout(resolve, 1000))
      setNewPassword('')
      setConfirmPassword('')
      alert('Password updated successfully')
    } catch (error) {
      console.error('Failed to update password:', error)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-sky-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-300">Loading settings...</p>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-12">
        <div className="text-center max-w-2xl">
          <Settings className="h-24 w-24 text-sky-500 mx-auto mb-8" />
          <h1 className="text-4xl font-bold text-sky-500 mb-4">Access Required</h1>
          <p className="text-slate-300 text-xl mb-8">
            Please log in to access settings.
          </p>
          <Link href="/login">
            <Button className="bg-sky-500 hover:bg-sky-600 text-white px-8 py-4 text-xl rounded-xl">
              Login
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  if (!settings) return null

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

          {/* Analysis Section */}
          {currentUser &&
            (currentUser.role === 'admin' || currentUser.role === 'researcher') && (
              <>
                <div className="pt-4 pb-2">
                  <div className="text-slate-500 text-xs uppercase tracking-wide font-medium px-3">
                    Code Analysis
                  </div>
                </div>
                <Link href="/analyze" className="block">
                  <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
                    <div className="flex items-center gap-3">
                      <Code2 className="h-5 w-5" />
                      <span>Single File Analysis</span>
                    </div>
                  </div>
                </Link>
                <Link href="/analyze-natural" className="block">
                  <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200 bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
                    <div className="flex items-center gap-3">
                      <Brain className="h-5 w-5 text-purple-400" />
                      <span className="text-purple-300">Natural Language Analysis</span>
                    </div>
                  </div>
                </Link>
              </>
            )}

          <Link href="/patterns" className="block">
            <div className="card-relief text-slate-400 hover:text-sky-500 px-5 py-4 rounded-xl hover:border-sky-500/20 transition-all duration-200">
              <div className="flex items-center gap-3">
                <FileText className="h-5 w-5" />
                <span>Patterns</span>
              </div>
            </div>
          </Link>

          {/* Admin Section */}
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
            </>
          )}

          <div className="card-relief-strong text-sky-500 px-5 py-4 rounded-xl border-sky-500/30">
            <div className="flex items-center gap-3">
              <Settings className="h-5 w-5" />
              <span className="font-semibold">Settings</span>
            </div>
          </div>
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
              <h1 className="text-4xl font-bold text-slate-100 mb-2">Settings</h1>
              <p className="text-slate-300 text-lg">
                Manage your account and preferences
              </p>
            </div>
            <div className="text-right">
              <p className="text-slate-400 text-sm">User Role</p>
              <p className="text-slate-100 text-xl font-bold capitalize">
                {settings.profile.role}
              </p>
            </div>
          </div>
        </div>

        {/* Profile Settings */}
        <div className="section-elevated p-8">
          <div className="flex items-center gap-3 mb-6">
            <User className="h-6 w-6 text-sky-500" />
            <h2 className="text-2xl font-bold text-slate-100">Profile Information</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-slate-300 text-sm font-medium mb-2">
                Username
              </label>
              <Input
                value={settings.profile.username}
                onChange={e => updateSetting('profile', 'username', e.target.value)}
                className="bg-slate-800 border-slate-600 text-slate-100"
              />
            </div>
            <div>
              <label className="block text-slate-300 text-sm font-medium mb-2">
                Full Name
              </label>
              <Input
                value={settings.profile.fullName}
                onChange={e => updateSetting('profile', 'fullName', e.target.value)}
                className="bg-slate-800 border-slate-600 text-slate-100"
              />
            </div>
            <div>
              <label className="block text-slate-300 text-sm font-medium mb-2">
                Email
              </label>
              <Input
                type="email"
                value={settings.profile.email}
                onChange={e => updateSetting('profile', 'email', e.target.value)}
                className="bg-slate-800 border-slate-600 text-slate-100"
              />
            </div>
            <div>
              <label className="block text-slate-300 text-sm font-medium mb-2">
                Role
              </label>
              <Input
                value={settings.profile.role}
                disabled
                className="bg-slate-700 border-slate-600 text-slate-400"
              />
            </div>
          </div>
        </div>

        {/* Password Change */}
        <div className="section-elevated p-8">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="h-6 w-6 text-sky-500" />
            <h2 className="text-2xl font-bold text-slate-100">Security</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-slate-300 text-sm font-medium mb-2">
                New Password
              </label>
              <div className="relative">
                <Input
                  type={showPassword ? 'text' : 'password'}
                  value={newPassword}
                  onChange={e => setNewPassword(e.target.value)}
                  placeholder="Enter new password"
                  className="bg-slate-800 border-slate-600 text-slate-100 pr-10"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-0 top-0 h-full px-3 text-slate-400 hover:text-slate-200"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
            <div>
              <label className="block text-slate-300 text-sm font-medium mb-2">
                Confirm Password
              </label>
              <Input
                type={showPassword ? 'text' : 'password'}
                value={confirmPassword}
                onChange={e => setConfirmPassword(e.target.value)}
                placeholder="Confirm new password"
                className="bg-slate-800 border-slate-600 text-slate-100"
              />
            </div>
          </div>

          <div className="mt-4">
            <Button
              onClick={handlePasswordChange}
              disabled={!newPassword || !confirmPassword}
              className="bg-orange-600 hover:bg-orange-700 text-white"
            >
              Update Password
            </Button>
          </div>
        </div>

        {/* Preferences */}
        <div className="section-elevated p-8">
          <div className="flex items-center gap-3 mb-6">
            <Bell className="h-6 w-6 text-sky-500" />
            <h2 className="text-2xl font-bold text-slate-100">Preferences</h2>
          </div>

          <div className="space-y-6">
            {/* Notifications Toggle */}
            <div className="flex items-center justify-between p-4 bg-slate-800 rounded-lg">
              <div>
                <h3 className="text-slate-100 font-medium">Analysis Notifications</h3>
                <p className="text-slate-400 text-sm">
                  Get notified when analysis completes
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.preferences.notifications}
                  onChange={e =>
                    updateSetting('preferences', 'notifications', e.target.checked)
                  }
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sky-500"></div>
              </label>
            </div>

            {/* Email Alerts Toggle */}
            <div className="flex items-center justify-between p-4 bg-slate-800 rounded-lg">
              <div>
                <h3 className="text-slate-100 font-medium">Email Alerts</h3>
                <p className="text-slate-400 text-sm">
                  Receive email notifications for important updates
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.preferences.emailAlerts}
                  onChange={e =>
                    updateSetting('preferences', 'emailAlerts', e.target.checked)
                  }
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sky-500"></div>
              </label>
            </div>

            {/* Auto-fix Toggle */}
            <div className="flex items-center justify-between p-4 bg-slate-800 rounded-lg">
              <div>
                <h3 className="text-slate-100 font-medium">Auto-fix Patterns</h3>
                <p className="text-slate-400 text-sm">
                  Automatically apply suggested fixes when possible
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.preferences.autoFix}
                  onChange={e =>
                    updateSetting('preferences', 'autoFix', e.target.checked)
                  }
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sky-500"></div>
              </label>
            </div>
          </div>
        </div>

        {/* Privacy Settings */}
        <div className="section-elevated p-8">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="h-6 w-6 text-sky-500" />
            <h2 className="text-2xl font-bold text-slate-100">Privacy & Data</h2>
          </div>

          <div className="space-y-6">
            <div className="flex items-center justify-between p-4 bg-slate-800 rounded-lg">
              <div>
                <h3 className="text-slate-100 font-medium">Share Analytics</h3>
                <p className="text-slate-400 text-sm">
                  Help improve Attrahere by sharing anonymous usage data
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.privacy.shareAnalytics}
                  onChange={e =>
                    updateSetting('privacy', 'shareAnalytics', e.target.checked)
                  }
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-sky-500"></div>
              </label>
            </div>

            <div className="p-4 bg-slate-800 rounded-lg">
              <h3 className="text-slate-100 font-medium mb-2">Data Retention</h3>
              <p className="text-slate-400 text-sm mb-3">
                How long should we keep your analysis data?
              </p>
              <select
                value={settings.privacy.dataRetention}
                onChange={e =>
                  updateSetting('privacy', 'dataRetention', e.target.value)
                }
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100"
              >
                <option value="3months">3 Months</option>
                <option value="6months">6 Months</option>
                <option value="1year">1 Year</option>
                <option value="2years">2 Years</option>
                <option value="indefinite">Indefinite</option>
              </select>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="section-elevated p-8">
          <div className="flex items-center justify-between">
            <div>
              {saveSuccess && (
                <div className="flex items-center gap-2 text-green-400">
                  <CheckCircle className="h-5 w-5" />
                  <span>Settings saved successfully!</span>
                </div>
              )}
            </div>
            <Button
              onClick={handleSaveSettings}
              disabled={isSaving}
              className="bg-green-600 hover:bg-green-700 text-white flex items-center gap-2 px-8 py-3"
            >
              {isSaving ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Save className="h-4 w-4" />
              )}
              {isSaving ? 'Saving...' : 'Save Settings'}
            </Button>
          </div>
        </div>
      </main>
    </div>
  )
}
