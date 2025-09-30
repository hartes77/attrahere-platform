import { test, expect } from '@playwright/test'

test.describe('Homepage', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should load homepage successfully', async ({ page }) => {
    // Check main title is visible
    await expect(
      page.locator('h1').filter({ hasText: 'ML Code Quality Platform' })
    ).toBeVisible()

    // Check subtitle/description is present
    await expect(
      page.locator(
        'text=AI-powered code analysis specifically designed for Machine Learning projects'
      )
    ).toBeVisible()
  })

  test('should display design system showcase', async ({ page }) => {
    // Check color palette section
    await expect(page.locator('text=Design System Showcase')).toBeVisible()

    // Check if color cards are present
    await expect(page.locator('text=Primary - Trust & Tech')).toBeVisible()
    await expect(page.locator('text=Secondary - AI & Growth')).toBeVisible()
    await expect(page.locator('text=Warning - RLHF Feedback')).toBeVisible()
    await expect(page.locator('text=Error - Critical Issues')).toBeVisible()
  })

  test('should display button variants', async ({ page }) => {
    // Check button showcase section
    await expect(page.locator('text=Button Variants')).toBeVisible()

    // Check different button variants are present
    await expect(page.getByRole('button', { name: 'Primary' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Secondary' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Outline' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Ghost' })).toBeVisible()
  })

  test('should display ML pattern detection example', async ({ page }) => {
    // Check pattern card showcase
    await expect(page.locator('text=ML Pattern Detection Example')).toBeVisible()

    // Check if PatternCard is visible with sample data
    await expect(
      page.locator('text=Data preprocessing applied before train-test split')
    ).toBeVisible()
    await expect(
      page.locator('text=StandardScaler is fitted on the entire dataset')
    ).toBeVisible()
  })

  test('should display key features section', async ({ page }) => {
    // Check features section
    await expect(page.locator('text=Key Features')).toBeVisible()

    // Check individual feature cards
    await expect(page.locator('text=AI-Powered Analysis')).toBeVisible()
    await expect(page.locator('text=Real-time Feedback')).toBeVisible()
    await expect(page.locator('text=Enterprise Security')).toBeVisible()
    await expect(page.locator('text=Performance Insights')).toBeVisible()
  })

  test('should display statistics', async ({ page }) => {
    // Check stats section
    await expect(page.locator('text=94.1%')).toBeVisible()
    await expect(page.locator('text=Test Success Rate')).toBeVisible()
    await expect(page.locator('text=<1s')).toBeVisible()
    await expect(page.locator('text=Analysis Response Time')).toBeVisible()
    await expect(page.locator('text=7+')).toBeVisible()
    await expect(page.locator('text=ML Pattern Types')).toBeVisible()
  })

  test('should have working start analysis button', async ({ page }) => {
    const startButton = page.getByRole('link').filter({ hasText: 'Start Analysis' })
    await expect(startButton).toBeVisible()
    await expect(startButton).toBeEnabled()

    // Click should navigate to analyze page
    await startButton.click()
    await expect(page).toHaveURL('/analyze')
  })

  test('should have responsive design', async ({ page }) => {
    // Test on mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    await page.reload()

    // Main title should still be visible
    await expect(
      page.locator('h1').filter({ hasText: 'ML Code Quality Platform' })
    ).toBeVisible()

    // Start Analysis button should be visible
    await expect(
      page.getByRole('link').filter({ hasText: 'Start Analysis' })
    ).toBeVisible()
  })
})
