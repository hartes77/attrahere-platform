import { test, expect } from '@playwright/test'

test.describe('Analyze Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/analyze')
  })

  test('should load analyze page successfully', async ({ page }) => {
    // Check header elements
    await expect(page.locator('text=ML Code Analyzer')).toBeVisible()
    await expect(page.locator('text=AI-powered ML pattern detection')).toBeVisible()

    // Check API status indicator (might be online or offline)
    await expect(page.locator('text=API').first()).toBeVisible()
  })

  test('should show login form when not authenticated', async ({ page }) => {
    // Check login form elements
    await expect(page.locator('text=Welcome to ML Code Analyzer')).toBeVisible()
    await expect(page.locator('text=Sign in to start analyzing')).toBeVisible()

    // Check form fields
    await expect(page.getByLabel('Username')).toBeVisible()
    await expect(page.getByLabel('Password')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Sign In' })).toBeVisible()

    // Check demo credentials hint
    await expect(page.locator('text=Demo credentials: admin / admin123')).toBeVisible()
  })

  test('should login successfully with correct credentials', async ({ page }) => {
    // Fill login form
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')

    // Submit form
    await page.getByRole('button', { name: 'Sign In' }).click()

    // Wait for login to complete and page to update
    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('text=Upload or paste your Python ML code')).toBeVisible()
  })

  test('should show error with incorrect credentials', async ({ page }) => {
    // Fill login form with wrong credentials
    await page.getByLabel('Username').fill('wrong')
    await page.getByLabel('Password').fill('wrong')

    // Submit form
    await page.getByRole('button', { name: 'Sign In' }).click()

    // Check for error message
    await expect(page.locator('[role="alert"], .text-error').first()).toBeVisible({
      timeout: 5000,
    })
  })

  test('should show code analyzer after login', async ({ page }) => {
    // Login first
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    // Wait for analyzer to appear
    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    // Check analyzer elements
    await expect(page.locator('text=Python Code')).toBeVisible()
    await expect(page.locator('textarea')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Analyze Code' })).toBeVisible()
  })

  test('should have pre-filled sample code', async ({ page }) => {
    // Login first
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    // Check that textarea has sample code
    const textarea = page.locator('textarea')
    await expect(textarea).toBeVisible()

    const content = await textarea.textContent()
    expect(content).toContain('StandardScaler')
    expect(content).toContain('train_test_split')
  })

  test('should enable analyze button only when code is present', async ({ page }) => {
    // Login first
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    const analyzeButton = page.getByRole('button', { name: 'Analyze Code' })
    const textarea = page.locator('textarea')

    // Button should be enabled with sample code
    await expect(analyzeButton).toBeEnabled()

    // Clear textarea
    await textarea.fill('')

    // Button should be disabled with empty code
    await expect(analyzeButton).toBeDisabled()

    // Add some code
    await textarea.fill('print("hello")')

    // Button should be enabled again
    await expect(analyzeButton).toBeEnabled()
  })

  test('should show logout functionality', async ({ page }) => {
    // Login first
    await page.getByLabel('Username').fill('admin')
    await page.getByLabel('Password').fill('admin123')
    await page.getByRole('button', { name: 'Sign In' }).click()

    await expect(page.locator('text=Code Analysis')).toBeVisible({ timeout: 10000 })

    // Check logout button is visible
    await expect(page.getByRole('button', { name: 'Logout' })).toBeVisible()

    // Click logout
    await page.getByRole('button', { name: 'Logout' }).click()

    // Should return to login form
    await expect(page.locator('text=Welcome to ML Code Analyzer')).toBeVisible()
  })

  test('should handle API offline state gracefully', async ({ page }) => {
    // Check for API offline warning (if backend is not running)
    const apiWarning = page.locator('text=Backend API Unavailable')

    // This test is flexible - it handles both online and offline states
    if (await apiWarning.isVisible()) {
      await expect(apiWarning).toBeVisible()
      await expect(
        page.locator('text=Cannot connect to the analysis backend')
      ).toBeVisible()
    }
  })

  test('should be responsive on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    await page.reload()

    // Check main elements are still visible
    await expect(page.locator('text=ML Code Analyzer')).toBeVisible()
    await expect(page.locator('text=Welcome to ML Code Analyzer')).toBeVisible()

    // Login form should be usable
    await expect(page.getByLabel('Username')).toBeVisible()
    await expect(page.getByLabel('Password')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Sign In' })).toBeVisible()
  })
})
