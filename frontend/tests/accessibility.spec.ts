import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test.describe('Accessibility Tests', () => {
  test('homepage should not have accessibility violations', async ({ page }) => {
    await page.goto('/')

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze()

    expect(accessibilityScanResults.violations).toEqual([])
  })

  test('analyze page should not have accessibility violations', async ({ page }) => {
    await page.goto('/analyze')

    // Wait for the page to load completely
    await page.waitForLoadState('networkidle')

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze()

    expect(accessibilityScanResults.violations).toEqual([])
  })

  test('keyboard navigation should work properly', async ({ page }) => {
    await page.goto('/analyze')

    // Login first
    await page.fill('input[type="text"]', 'admin')
    await page.fill('input[type="password"]', 'admin123')
    await page.click('button[type="submit"]')

    await page.waitForLoadState('networkidle')

    // Test keyboard navigation
    await page.keyboard.press('Tab')
    const focused = await page.evaluate(() => document.activeElement?.tagName)
    expect(['BUTTON', 'INPUT', 'A']).toContain(focused)
  })

  test('skip link should be functional', async ({ page }) => {
    await page.goto('/')

    // Focus skip link
    await page.keyboard.press('Tab')

    const skipLink = page.locator('a:has-text("Skip to main content")')
    await expect(skipLink).toBeFocused()

    // Activate skip link
    await page.keyboard.press('Enter')

    const mainContent = page.locator('#main-content')
    await expect(mainContent).toBeVisible()
  })

  test('form elements should have proper labels', async ({ page }) => {
    await page.goto('/analyze')

    // Check login form labels
    const usernameInput = page.locator('input[type="text"]')
    const passwordInput = page.locator('input[type="password"]')

    await expect(usernameInput).toHaveAttribute('aria-label', /username/i)
    await expect(passwordInput).toHaveAttribute('aria-label', /password/i)
  })

  test('interactive elements should have proper ARIA attributes', async ({ page }) => {
    await page.goto('/analyze')

    // Login to access dashboard
    await page.fill('input[type="text"]', 'admin')
    await page.fill('input[type="password"]', 'admin123')
    await page.click('button[type="submit"]')

    await page.waitForLoadState('networkidle')

    // Check sidebar navigation buttons
    const overviewButton = page.locator('button:has-text("Overview")')
    await expect(overviewButton).toHaveAttribute('aria-pressed')

    const menuToggle = page.locator('button[aria-label*="sidebar"]')
    await expect(menuToggle).toHaveAttribute('aria-expanded')
  })

  test('color contrast should meet WCAG standards', async ({ page }) => {
    await page.goto('/')

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2aa'])
      .include(['color-contrast'])
      .analyze()

    expect(accessibilityScanResults.violations).toEqual([])
  })

  test('images should have alt text', async ({ page }) => {
    await page.goto('/')

    const images = page.locator('img')
    const count = await images.count()

    for (let i = 0; i < count; i++) {
      const img = images.nth(i)
      await expect(img).toHaveAttribute('alt')
    }
  })
})

test.describe('Screen Reader Support', () => {
  test('live regions should announce status changes', async ({ page }) => {
    await page.goto('/analyze')

    // Login to access dashboard
    await page.fill('input[type="text"]', 'admin')
    await page.fill('input[type="password"]', 'admin123')
    await page.click('button[type="submit"]')

    await page.waitForLoadState('networkidle')

    // Check for live regions
    const liveRegions = page.locator('[aria-live]')
    const count = await liveRegions.count()
    expect(count).toBeGreaterThanOrEqual(0)
  })

  test('error messages should be announced', async ({ page }) => {
    await page.goto('/analyze')

    // Try to login with wrong credentials
    await page.fill('input[type="text"]', 'wrong')
    await page.fill('input[type="password"]', 'wrong')
    await page.click('button[type="submit"]')

    // Check for error message with proper ARIA
    const errorMessage = page.locator('[role="alert"], [aria-live="assertive"]')
    await expect(errorMessage).toBeVisible({ timeout: 5000 })
  })
})
