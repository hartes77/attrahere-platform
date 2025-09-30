'use client'

import { PrivateBetaGuard } from '@/components/PrivateBetaGuard'
import { ROICalculator } from '@/components/roi-calculator'

export default function ROICalculatorPage() {
  return (
    <PrivateBetaGuard>
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="container mx-auto px-4">
          <ROICalculator />
        </div>
      </div>
    </PrivateBetaGuard>
  )
}
