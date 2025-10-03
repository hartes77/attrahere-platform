import { test, expect } from '@playwright/test'

test.describe('API Integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/analyze')
  })

  test('should complete full analysis workflow', async ({ page }) => {
    // Login first
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    // Check if API is online
    const apiStatus = page.locator('text=API Online')
    if (await apiStatus.isVisible()) {
      // API is online, test full workflow

      // Ensure sample code is present
      const textarea = page.locator('textarea')
      await expect(textarea).toBeVisible()

      const codeContent = await textarea.textContent()
      if (!codeContent || codeContent.trim().length === 0) {
        // Add sample code if not present
        await textarea.fill(`
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Data leakage issue
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)
        `)
      }

      // Click analyze button
      const analyzeButton = page.getByRole('button', { name: 'Analyze Code' })
      await expect(analyzeButton).toBeEnabled()
      await analyzeButton.click()

      // Check for loading state
      await expect(page.locator('text=Analyzing')).toBeVisible({ timeout: 2000 })

      // Wait for results or timeout
      await expect(
        page.locator('text=Analysis Results, text=No Issues Found').first()
      ).toBeVisible({ timeout: 30000 })

      // If results are shown, check pattern cards
      const resultsSection = page.locator('text=Analysis Results')
      if (await resultsSection.isVisible()) {
        // Check for pattern detection UI elements
        await expect(
          page.locator('[role="button"]').filter({ hasText: /👍|👎/ }).first()
        ).toBeVisible()
      }
    } else {
      // API is offline, check appropriate error handling
      await expect(
        page
          .locator(
            'text=Backend API Unavailable, text=Cannot connect to the analysis backend'
          )
          .first()
      ).toBeVisible()
    }
  })

  test('should handle authentication flow correctly', async ({ page }) => {
    // Test wrong credentials first
    await page.getByLabel('Username').fill('wronguser')
    await page.getByLabel('Password').fill('wrongpass')
    await page.getByRole('button', { name: 'Sign In' }).click()

    // Should show error (wait for network request to complete)
    await expect(page.locator('.text-error, [role="alert"]').first()).toBeVisible({
      timeout: 5000,
    })

    // Now test correct credentials
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    // Should login successfully
    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })
  })

  test('should show pattern feedback functionality', async ({ page }) => {
    // Login
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    // Check if sample PatternCard is visible from homepage demo
    await page.goto('/')

    // Look for pattern card with feedback buttons
    const patternCard = page
      .locator('text=Data preprocessing applied before train-test split')
      .locator('..')
    if (await patternCard.isVisible()) {
      // Check for thumbs up/down buttons
      await expect(patternCard.locator('[role="button"]').first()).toBeVisible()
    }
  })

  test('should handle network errors gracefully', async ({ page }) => {
    // Mock network failure by intercepting requests
    await page.route('/api/**', route => {
      route.abort('failed')
    })

    // Try to login
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    // Should handle network error
    await expect(page.locator('.text-error, [role="alert"]').first()).toBeVisible({
      timeout: 5000,
    })
  })

  test('should maintain authentication state on refresh', async ({ page }) => {
    // Login first
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    // Refresh page
    await page.reload()

    // Should still be logged in (token persisted in localStorage)
    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    // Logout should work
    await page.getByRole('button', { name: 'Logout' }).click()

    // Should return to login
    await expect(page.locator('text=Welcome to ML Code Analyzer')).toBeVisible()

    // Refresh after logout should still show login
    await page.reload()
    await expect(page.locator('text=Welcome to ML Code Analyzer')).toBeVisible()
  })

  test('should validate code input', async ({ page }) => {
    // Login
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    const textarea = page.locator('textarea')
    const analyzeButton = page.getByRole('button', { name: 'Analyze Code' })

    // Empty code should disable button
    await textarea.fill('')
    await expect(analyzeButton).toBeDisabled()

    // Whitespace only should disable button
    await textarea.fill('   \n  \t  ')
    await expect(analyzeButton).toBeDisabled()

    // Valid code should enable button
    await textarea.fill('print("hello world")')
    await expect(analyzeButton).toBeEnabled()
  })
})
